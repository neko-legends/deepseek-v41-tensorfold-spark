#!/usr/bin/env python3
"""Per-layer numerics gate for GLM-5.3-Flash-FP4-Local: the checkpoint's own HF decoder layers (transformers 5.20.1,
Glm5NextTextDecoderLayer, eager) against tensorfold's per-layer residual-stream dumps (engine/zig-glm, TF_GLM_DUMP).

Weights are dequantized the way ModelOpt defines each format (no tensorfold code involved):
  MXFP8      e4m3 * 2^(ue8m0 - 127), a scale per 32 inputs
  FP8_PB_WO  e4m3 * weight_scale, one fp32 scale per 128 x 128 block ([n/128, 1, k/128, 1])
  NVFP4      e2m1 (low nibble first) * e4m3 block scale (16 inputs) * weight_scale_2; csf_32x2_v1 scales are expanded
             by FlashInfer B12X's own _csf_blocks.py (--b12x DIR), not by tensorfold's decoder
Each layer gets tensorfold's own input streams h_i (TF_GLM_DUMP=DIR: DIR/NNN_h.bin, bf16 [rows][hc][hidden] entering
layer NNN, windows appended), so errors do not compound; its output is compared with h_{i+1}. Routed experts are read
on demand (only the ones the router picks). The reference runs on the CPU in fp32 (--dtype bf16 for bf16 blocks).
Keep the prompt under 2048 tokens: then every DSA query attends to all its keys and indexer ties cannot differ.

  layer_ref.py MODEL_DIR DUMP_DIR --rows N --layers 0,1,3 --b12x DIR      (gate: tensorfold vs the reference)
  layer_ref.py MODEL_DIR - --ids 1,2,3 --layers 0,1,2,3 --b12x DIR        (self test: the reference chain alone)
MODEL_DIR may be `hf:` to read tensors by HTTP range (GLM_HEADERS / GLM_TENSORS, tools/glm/hf_tensor.py).
"""
import argparse, json, math, os, sys
import numpy as np
import torch
import torch.nn.functional as F

E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)
DT = {"BF16": torch.bfloat16, "F32": torch.float32, "U8": torch.uint8, "F8_E4M3": torch.float8_e4m3fn,
      "U16": torch.uint16, "U32": torch.uint32, "I32": torch.int32}
PRE = "model.language_model."


class Ckpt:
    """Tensors by full name from a local checkpoint folder, or by HTTP range (`hf:`)."""

    def __init__(self, d):
        self.d, self.open = d, {}
        if d == "hf:":
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import hf_tensor
            self.hf = hf_tensor
            self.map = hf_tensor.H["index"]["weight_map"]
        else:
            self.hf = None
            self.map = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]

    def has(self, name):
        return name in self.map

    def raw(self, name):
        if self.hf is not None:
            t = self.hf.meta(name)[2]
            buf = bytearray(self.hf.fetch(name))
            return torch.frombuffer(buf, dtype=DT[t["dtype"]]).reshape(t["shape"]) if buf else torch.empty(t["shape"], dtype=DT[t["dtype"]])
        from safetensors import safe_open
        f = self.map[name]
        if f not in self.open:
            self.open[f] = safe_open(os.path.join(self.d, f), "pt")
        return self.open[f].get_tensor(name)


class Dequant:
    def __init__(self, ck, b12x):
        self.ck, self.cb = ck, None
        if b12x:
            sys.path.insert(0, b12x)
            import _csf_blocks
            self.cb = _csf_blocks

    def scales_nvfp4(self, pre, n, k):
        if self.ck.has(pre + ".weight_scale"):
            return self.ck.raw(pre + ".weight_scale").float()
        if self.cb is None:
            raise SystemExit(f"{pre}: csf_32x2_v1 scales need --b12x (FlashInfer's _csf_blocks.py)")
        g = lambda f, dt: self.ck.raw(pre + ".weight_scale.csf32_" + f).numpy().view(dt).ravel()
        plane = self.cb.CsfBlockPlane(n, k // 16, "nvfp4", g("bases", np.uint8), g("codes", np.uint8),
                                      g("page_bases", "<u4"), g("offsets", "<u2"), g("exceptions", "<u2"))
        plane.validate()
        bases, fields, pos, vals = self.cb.logical_csf_components(plane)
        logical = (bases[:, None].astype(np.uint16) + fields).astype(np.uint8)
        logical.ravel()[pos] = vals
        return torch.from_numpy(logical.copy()).view(torch.float8_e4m3fn).float()

    def weight(self, pre):
        """`pre`.weight as fp32 [n, k], whatever its stored format."""
        w = self.ck.raw(pre + ".weight")
        if w.dtype == torch.float8_e4m3fn:
            s = self.ck.raw(pre + ".weight_scale")
            n, k = w.shape
            if s.dtype == torch.uint8 and s.dim() == 2:  # MXFP8
                return w.float() * torch.exp2(s.float() - 127).repeat_interleave(32, dim=1)[:, :k]
            if s.dtype == torch.float32 and s.dim() == 4:  # FP8_PB_WO
                s2 = s[:, 0, :, 0]
                return w.float() * s2.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]
            raise SystemExit(f"{pre}: e4m3 with a {s.dtype} {tuple(s.shape)} scale")
        if w.dtype == torch.uint8:  # NVFP4
            n, kb = w.shape
            codes = torch.stack([E2M1[(w & 0xF).long()], E2M1[(w >> 4).long()]], dim=-1).reshape(n, kb * 2)
            bs = self.scales_nvfp4(pre, n, kb * 2)
            return codes * bs.repeat_interleave(16, dim=1) * self.ck.raw(pre + ".weight_scale_2").float()
        return w.float()


class LazyExperts(torch.nn.Module):
    """Glm5NextTextExperts' eager forward with each picked expert's weights read on demand."""

    def __init__(self, dq, layer, cfg, dtype):
        super().__init__()
        self.dq, self.layer, self.n, self.limit, self.dtype, self.cache = dq, layer, cfg.num_local_experts, cfg.swiglu_limit, dtype, {}

    def get(self, e):
        if e not in self.cache:
            p = f"{PRE}layers.{self.layer}.mlp.experts.{e}."
            gu = torch.cat([self.dq.weight(p + "gate_proj"), self.dq.weight(p + "up_proj")], 0).to(self.dtype)
            self.cache[e] = (gu, self.dq.weight(p + "down_proj").to(self.dtype))
        return self.cache[e]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        for e in sorted(set(top_k_index.flatten().tolist())):
            if e >= self.n:
                continue
            tok, pos = torch.where(top_k_index == e)
            gu, down = self.get(e)
            gate, up = F.linear(hidden_states[tok], gu).chunk(2, dim=-1)
            cur = F.silu(gate.clamp(max=self.limit)) * up.clamp(min=-self.limit, max=self.limit)
            cur = F.linear(cur, down) * top_k_weights[tok, pos, None]
            final.index_add_(0, tok, cur.to(final.dtype))
        return final


def load_layer(M, dq, cfg, i, dtype):
    with torch.device("meta"):
        blk = M.Glm5NextTextDecoderLayer(cfg, i)
    blk = blk.to_empty(device="cpu")
    p = f"{PRE}layers.{i}."
    sd = {}
    for k in blk.state_dict():
        if k.startswith("mlp.experts."):
            continue
        site, _, rest = k.partition(".")
        if site in ("attn_hc", "ffn_hc"):
            sd[k] = dq.ck.raw(p + f"hc_{site[:-3]}_{rest}").float()
            continue
        if k == "self_attn.conv1d.weight":
            sd[k] = torch.cat([dq.ck.raw(p + f"self_attn.{x}_conv1d.weight").float() for x in "qkv"], 0)
            continue
        name = p + k.replace("self_attn.forget_gate.", "self_attn.")
        if not dq.ck.has(name):
            continue
        t = dq.ck.raw(name)
        sd[k] = dq.weight(name[:-len(".weight")]) if t.dtype in (torch.float8_e4m3fn, torch.uint8) else t.float()
    missing = [k for k in blk.state_dict() if not k.startswith("mlp.experts.") and k not in sd]
    if missing:
        raise SystemExit(f"layer {i}: no checkpoint tensor for {missing}")
    if hasattr(blk.mlp, "experts"):
        blk.mlp.experts = LazyExperts(dq, i, cfg, dtype)
    blk.load_state_dict({k: v for k, v in sd.items()}, strict=False)
    return blk.to(dtype).eval()


def stats(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    cos = float(F.cosine_similarity(a, b, dim=0))
    return cos, float((a - b).abs().max()), float((a - b).norm() / max(b.norm(), 1e-30))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model"), ap.add_argument("dump")
    ap.add_argument("--rows", type=int, help="rows of the first window (the prompt) in the dumps")
    ap.add_argument("--ids", help="self test: token ids, comma separated (no dumps)")
    ap.add_argument("--layers", default="0,1,2,3")
    ap.add_argument("--b12x", help="folder with FlashInfer B12X's _csf_blocks.py")
    ap.add_argument("--dtype", default="fp32", choices=("fp32", "bf16"))
    ap.add_argument("--config", help="config.json (default MODEL_DIR/config.json)")
    a = ap.parse_args()
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig
    from transformers.models.glm5_next import modeling_glm5_next as M
    torch.set_grad_enabled(False)
    dtype = torch.float32 if a.dtype == "fp32" else torch.bfloat16
    cfg_path = a.config or os.path.join(a.model, "config.json")
    cfg = Glm5NextConfig(**json.load(open(cfg_path))).text_config
    cfg._attn_implementation = "eager"
    ck = Ckpt(a.model)
    dq = Dequant(ck, a.b12x)
    layers = [int(x) for x in a.layers.split(",")]
    S, D = cfg.hc_mult, cfg.hidden_size
    if a.ids:
        ids = torch.tensor([[int(x) for x in a.ids.split(",")]])
        rows = ids.shape[1]
        emb = ck.raw(PRE + "embed_tokens.weight")[ids[0]].to(dtype)
        h = emb[None, :, None, :].expand(1, rows, S, D).contiguous()
    else:
        rows = a.rows

    def dumped(i):
        raw = np.fromfile(os.path.join(a.dump, f"{i:03d}_h.bin"), dtype=np.uint16, count=rows * S * D)
        return torch.from_numpy(raw.astype(np.int32) << 16).view(torch.float32).reshape(1, rows, S, D).to(dtype)

    mask = torch.ones(1, rows, dtype=torch.bool)
    pos = torch.arange(rows)[None]
    worst = 1.0
    for i in layers:
        blk = load_layer(M, dq, cfg, i, dtype)
        x = h if a.ids else dumped(i)
        out, _ = blk(x, attention_mask=mask, position_ids=pos, position_embeddings=None, past_key_values=None)
        kind = cfg.layer_types[i]
        if a.ids:
            print(f"layer {i:2d} {kind:17s} out rms {float(out.float().pow(2).mean().sqrt()):.4g} finite {bool(torch.isfinite(out).all())}"
                  f" delta-cos(in,out) {stats(out, x)[0]:.4f}")
            h = out
            continue
        want = dumped(i + 1)
        cos, mx, rel = stats(want, out)
        per = [stats(want[0, :, s], out[0, :, s])[0] for s in range(S)]
        worst = min(worst, cos)
        print(f"layer {i:2d} {kind:17s} cos {cos:.6f} rel {rel:.3e} max|d| {mx:.3e} streams " + " ".join(f"{c:.5f}" for c in per))
    if not a.ids:
        print(f"worst cos {worst:.6f}")


if __name__ == "__main__":
    main()
