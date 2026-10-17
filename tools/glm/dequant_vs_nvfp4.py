# Dequantized FP4-Local tensors vs the same tensors of nvidia/GLM-5.3-Flash-NVFP4 (bf16 attention / shared experts,
# NVFP4 experts with plain scales): semantics check of MXFP8, FP8_PB_WO and csf NVFP4 dequantization.
import importlib, os, sys
import torch
# Usage: GLM_WORK=DIR dequant_vs_nvfp4.py  (DIR: headers-hf.json and headers-nvfp4.json from hf_headers.py, ref/_csf_blocks.py)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
here = os.environ["GLM_WORK"]
def ck(headers, tensors, repo):
    os.environ.update(GLM_HEADERS=headers, GLM_TENSORS=tensors, GLM_REPO=repo)
    import hf_tensor; importlib.reload(hf_tensor)
    import layer_ref; importlib.reload(layer_ref)
    c = layer_ref.Ckpt("hf:")
    return layer_ref.Dequant(c, here + "/ref")
P = "model.language_model.layers."
mods = ["3.self_attn.q_a_proj", "3.self_attn.o_proj", "0.self_attn.q_proj", "0.mlp.down_proj", "3.mlp.shared_experts.up_proj",
        "20.mlp.experts.137.gate_proj", "44.mlp.experts.287.down_proj", "45.mlp.experts.5.up_proj"]
ours = ck(here + "/headers-hf.json", here + "/tensors", "nvidia/GLM-5.3-Flash-FP4-Local")
A = {m: ours.weight(P + m) for m in mods}
theirs = ck(here + "/headers-nvfp4.json", here + "/tensors-nvfp4", "nvidia/GLM-5.3-Flash-NVFP4")
for m in mods:
    a, b = A[m], theirs.weight(P + m)
    cos = float(torch.nn.functional.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0))
    print(f"{m:34s} cos {cos:.5f}  rms ours {float(a.pow(2).mean().sqrt()):.4e} theirs {float(b.pow(2).mean().sqrt()):.4e}")
