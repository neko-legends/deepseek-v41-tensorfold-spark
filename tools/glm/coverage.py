#!/usr/bin/env python3
"""Header-only coverage of engine/zig-glm's loader (families/glm/cuda_weights.zig) on GLM-5.3-Flash-FP4-Local: every
tensor name the loader reads, by its rules, against the checkpoint's index (hf_headers.py output). Reports names the
loader would miss (a load error) and language-model names it never reads (left unused). Vision tensors are out of
scope (no CUDA vision yet) and listed as a count.

  coverage.py HEADERS_JSON CONFIG_JSON
"""
import json, sys

P = "model.language_model."


def main():
    H = json.load(open(sys.argv[1]))
    names = set(H["index"]["weight_map"])
    t = json.load(open(sys.argv[2]))["text_config"]
    L, E, dense = t["num_hidden_layers"], t["n_routed_experts"], t["first_k_dense_replace"]
    mla = set(t["linear_attn_config"]["full_attn_layers"])
    want = set()

    def fp8(m):  # get() widens an e4m3 weight with its _scale; a bf16 one is read as is
        want.add(m + ".weight")
        if m + ".weight_scale" in names:
            want.add(m + ".weight_scale")

    def nvfp4(m):  # nvfp4Linear: codes, scale_2, csf planes or plain scales, input_scale when stored
        want.update({m + ".weight", m + ".weight_scale_2"})
        if m + ".weight_scale.csf32_bases" in names:
            want.update(m + ".weight_scale.csf32_" + x for x in ("bases", "codes", "page_bases", "offsets", "exceptions"))
        else:
            want.add(m + ".weight_scale")
        if m + ".input_scale" in names:
            want.add(m + ".input_scale")

    for i in range(L + t.get("num_nextn_predict_layers", 0)):
        p = f"{P}layers.{i}."
        mtp = i == L
        want.update({p + "input_layernorm.weight", p + "post_attention_layernorm.weight"})
        if not mtp:
            want.update(p + f"hc_{s}_{x}" for s in ("attn", "ffn") for x in ("fn", "base", "scale"))
        a = p + "self_attn."
        if i in mla or mtp:
            for x in ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "kv_b_proj", "o_proj", "indexer.wk", "indexer.weights_proj", "indexer.wq_b"):
                fp8(a + x)
            want.update(a + x for x in ("q_a_layernorm.weight", "kv_a_layernorm.weight", "indexer.k_norm.weight", "indexer.k_norm.bias",
                                        "indexer.index_kpool_compress_gate", "indexer.index_kpool_compress_ape"))
        else:
            for x in ("q_proj", "k_proj", "v_proj", "f_a_proj", "g_a_proj", "b_proj", "f_b_proj", "g_b_proj", "o_proj"):
                fp8(a + x)
            want.update(a + x for x in ("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight", "A_log", "dt_bias", "o_norm.weight"))
        m = p + "mlp."
        if i < dense:
            for x in ("gate_proj", "up_proj", "down_proj"):
                fp8(m + x)
        else:
            want.update({m + "gate.weight", m + "gate.e_score_correction_bias"})
            for e in range(E):
                for x in ("gate_proj", "up_proj", "down_proj"):
                    nvfp4(f"{m}experts.{e}.{x}")
            for x in ("gate_proj", "up_proj", "down_proj"):
                fp8(m + "shared_experts." + x)
        if mtp:
            want.update(p + x for x in ("enorm.weight", "hnorm.weight", "eh_proj.weight", "shared_head.norm.weight"))
    want.update({P + "embed_tokens.weight", P + "norm.weight"})
    fp8("lm_head")
    missing = sorted(want - names)
    text = {n for n in names if not n.startswith("model.visual.")}
    unused = sorted(text - want)
    print(f"checkpoint {len(names)} names ({len(names) - len(text)} vision), loader reads {len(want)}")
    print(f"missing (load would fail): {len(missing)}")
    for n in missing[:20]:
        print("  ", n)
    print(f"text names never read: {len(unused)}")
    for n in unused[:20]:
        print("  ", n)
    return 1 if missing or unused else 0


if __name__ == "__main__":
    sys.exit(main())
