#!/usr/bin/env python3
"""Vision-tower numerics reference for GLM-5.3-Flash-FP4-Local: the checkpoint's own HF tower (transformers 5.20.1,
Glm5NextVisionModel) on one image, on the CPU, with every stage a CUDA port must match dumped as raw files.

Preprocessing is HF's (Glm5NextImageProcessorPil from processor_config.json; Pillow bicubic, zero pad, CLIP norm,
block-major 2x2 patch rows). Weights are dequantized the ModelOpt way by layer_ref.py's Dequant (no tensorfold code):
  MXFP8   attn.qkv / attn.proj                          e4m3 * 2^(ue8m0 - 127), a scale per 32 inputs
  NVFP4   mlp.* / merger.{proj,gate,up,down} (W4A16)    e2m1 (low nibble first) * e4m3 per 16 * weight_scale_2
  BF16    everything else (patch_embed, biases, norms, downsample, merger LayerNorm)
then cast to --dtype (fp32 -> bf16 once; rope inv_freq stays fp32 as from_pretrained leaves it).

  vision_ref.py MODEL_DIR IMAGE OUT_DIR [--dtype bf16|fp32] [--max-tokens N] [--stages] [--proc DIR] [--attn sdpa|eager]
               [--pydeps DIR]
MODEL_DIR may be `hf:` to read tensors by HTTP range (GLM_HEADERS / GLM_TENSORS, tools/glm/hf_tensor.py); then --proc
DIR must hold config.json (vision_config) and processor_config.json (default: MODEL_DIR). Pillow is imported lazily
(--pydeps DIR: a folder with it, as `python -I` ignores PYTHONPATH).

OUT_DIR: pixels.f32 (float32 [P][1176], = pixel_values), meta.json {grid_thw, P, tokens, image [w,h], canvas [h,w],
content [h,w], dtype, attn, stages: {name: {file, shape, dtype}}}, and one raw little-endian NAME.bin per stage (bf16, or
f32 for --dtype fp32): patch_embed [P,1024], block00/11/23 [P,1024] (--stages: every block), post_layernorm [P,1024],
downsample [P/4,4096] (= last_hidden_state), rows [P/4,4096] (merger = pooler_output, the <|image|> rows).
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layer_ref import Ckpt, Dequant  # noqa: E402

PRE = "model.visual."


def preprocess(proc_dir, image, max_tokens):
    from PIL import Image
    from transformers.models.glm5_next.image_processing_pil_glm5_next import Glm5NextImageProcessorPil, smart_resize
    cfg = json.load(open(os.path.join(proc_dir, "processor_config.json")))["image_processor"]
    cfg.pop("image_processor_type", None)
    if max_tokens:
        cfg["max_image_tokens"] = max_tokens
    ip = Glm5NextImageProcessorPil(**cfg)
    img = Image.open(image).convert("RGB")
    out = ip(images=[img], return_tensors="np")
    w, h = img.size
    T, f = cfg["temporal_patch_size"], cfg["patch_size"] * cfg["merge_size"] * cfg.get("patch_expand_factor", 1)
    th, tw = smart_resize(num_frames=T, height=h, width=w, temporal_factor=T, factor=f,
                          min_pixels=cfg["min_image_tokens"], max_pixels=cfg["max_image_tokens"])
    scale = min(th / h, tw / w)  # Glm5NextImageProcessorPil.resize's content size (then zero pad to th x tw)
    if T * h * w >= T * f * f * cfg["min_image_tokens"]:
        scale = min(1.0, scale)
    content = [max(1, min(th, math.floor(h * scale))), max(1, min(tw, math.floor(w * scale)))]
    return np.ascontiguousarray(out["pixel_values"], dtype=np.float32), out["image_grid_thw"], [w, h], [th, tw], content


def load_tower(M, ck, dq, vcfg, dtype):
    with torch.device("meta"):
        m = M.Glm5NextVisionModel(vcfg)
    used, raw = set(), ck.raw
    ck.raw = lambda n: used.add(n) or raw(n)
    sd = {}
    for k in m.state_dict():
        name = PRE + k
        if not ck.has(name):
            raise SystemExit(f"no checkpoint tensor {name}")
        sd[k] = (dq.weight(name[:-len(".weight")]) if k.endswith(".weight") else ck.raw(name).float()).to(dtype)
    m.load_state_dict(sd, strict=True, assign=True)
    inv, _ = M.Glm5NextVisionRotaryEmbedding.compute_axial_rope_parameters(vcfg)
    m.rotary_pos_emb.inv_freq = inv
    m.rotary_pos_emb.original_inv_freq = inv.clone()
    left = sorted(n for n in ck.map if n.startswith(PRE) and n not in used)
    print(f"tower: {len(sd)} HF tensors loaded from {len(used)} checkpoint tensors "
          f"({sum(1 for n in ck.map if n.startswith(PRE))} {PRE}*), {sum(v.numel() for v in sd.values()) / 1e6:.1f} M params")
    if left:
        raise SystemExit(f"{len(left)} unused checkpoint tensors: {left[:8]}")
    return m.eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model"), ap.add_argument("image"), ap.add_argument("out")
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--max-tokens", type=int, help="override processor_config's max_image_tokens")
    ap.add_argument("--stages", action="store_true", help="save the output of every block (default 0, 11, 23)")
    ap.add_argument("--proc", help="folder with config.json and processor_config.json (default MODEL_DIR)")
    ap.add_argument("--attn", default="sdpa", choices=("sdpa", "eager"), help="HF attention implementation (default sdpa, HF's)")
    ap.add_argument("--pydeps", help="extra import folder (Pillow), since `python -I` ignores PYTHONPATH")
    a = ap.parse_args()
    if a.pydeps:
        sys.path.append(a.pydeps)
    from transformers.models.glm5_next import modeling_glm5_next as M
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
    torch.set_grad_enabled(False)
    dtype = torch.float32 if a.dtype == "fp32" else torch.bfloat16
    proc = a.proc or a.model
    os.makedirs(a.out, exist_ok=True)

    pv, grid, size, canvas, content = preprocess(proc, a.image, a.max_tokens)
    P, tokens = pv.shape[0], int(grid.prod()) // 4
    pv.tofile(os.path.join(a.out, "pixels.f32"))
    print(f"image {size[0]}x{size[1]} -> content {content[0]}x{content[1]} canvas {canvas[0]}x{canvas[1]} "
          f"grid_thw {grid[0].tolist()} P {P} tokens {tokens}")

    vcfg = Glm5NextVisionConfig(**json.load(open(os.path.join(proc, "config.json")))["vision_config"])
    vcfg._attn_implementation = a.attn
    ck = Ckpt(a.model)
    t0 = time.time()
    m = load_tower(M, ck, Dequant(ck, None), vcfg, dtype)
    print(f"load {time.time() - t0:.1f} s")

    keep = set(range(vcfg.depth)) if a.stages else {0, 11, vcfg.depth - 1}
    mods = [("patch_embed", m.patch_embed)] + [(f"block{i:02d}", b) for i, b in enumerate(m.blocks) if i in keep]
    mods += [("post_layernorm", m.post_layernorm), ("downsample", m.downsample), ("rows", m.merger)]
    got = {}
    for name, mod in mods:
        mod.register_forward_hook(lambda _m, _i, o, name=name: got.__setitem__(name, o.detach().reshape(-1, o.shape[1] if o.dim() == 4 else o.shape[-1])))
    t0 = time.time()
    out = m(hidden_states=torch.from_numpy(pv), grid_thw=torch.from_numpy(grid))
    dt = time.time() - t0
    assert torch.equal(got["rows"], out.pooler_output) and torch.equal(got["downsample"], out.last_hidden_state)

    ext = "bf16" if dtype == torch.bfloat16 else "f32"  # meta dtype; files are NAME.bin
    stages = {}
    print(f"forward {dt:.1f} s ({a.dtype}, {a.attn})")
    for name, _ in mods:
        t = got[name].contiguous()
        f = f"{name}.bin"
        (t.view(torch.uint16) if dtype == torch.bfloat16 else t).numpy().tofile(os.path.join(a.out, f))
        stages[name] = {"file": f, "shape": list(t.shape), "dtype": ext}
        print(f"  {name:15s} {str(list(t.shape)):13s} rms {float(t.float().pow(2).mean().sqrt()):.6g} "
              f"max|x| {float(t.float().abs().max()):.4g} finite {bool(torch.isfinite(t).all())}")
    json.dump({"grid_thw": grid[0].tolist(), "P": P, "tokens": tokens, "image": size, "canvas": canvas, "content": content,
               "dtype": ext, "attn": a.attn, "max_tokens": a.max_tokens, "pixels": {"file": "pixels.f32", "shape": list(pv.shape), "dtype": "f32"},
               "stages": stages}, open(os.path.join(a.out, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
