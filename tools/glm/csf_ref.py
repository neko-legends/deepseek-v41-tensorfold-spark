# Reference decode of csf_32x2_v1 expert scales with FlashInfer B12X's own _csf_blocks.py (local-inference-lab/flashinfer
# 7e203806): CsfBlockPlane(...).validate(), logical_csf_components -> bases[:,None] + fields, exceptions written in.
# Writes <out>/<module>.e4m3 (plain [n, k/16] bytes) and prints sha256 per plane. Usage: csf_ref.py REF_DIR OUT_DIR MODULE...
import hashlib, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hf_tensor  # GLM_HEADERS / GLM_TENSORS
ref_dir, out_dir, modules = sys.argv[1], sys.argv[2], sys.argv[3:]
sys.path.insert(0, ref_dir)
import _csf_blocks as cb
os.makedirs(out_dir, exist_ok=True)
for m in modules:
    w = hf_tensor.meta(m + ".weight")[2]["shape"]
    rows, cols = w[0], w[1] * 2 // 16
    get = lambda f, dt: np.frombuffer(hf_tensor.fetch(m + ".weight_scale.csf32_" + f), dtype=dt)
    plane = cb.CsfBlockPlane(rows, cols, "nvfp4", get("bases", "u1"), get("codes", "u1"), get("page_bases", "<u4"),
                             get("offsets", "<u2"), get("exceptions", "<u2"))
    plane.validate()
    bases, fields, positions, values = cb.logical_csf_components(plane)
    logical = (bases[:, None].astype(np.uint16) + fields).astype(np.uint8)
    assert int((bases[:, None].astype(np.uint16) + fields).max()) <= 255
    logical.ravel()[positions] = values
    raw = logical.tobytes()
    open(os.path.join(out_dir, m + ".e4m3"), "wb").write(raw)
    f = logical.view(np.uint8)
    nan = int(np.sum((f & 0x7F) == 0x7F)); neg = int(np.sum(f >> 7))
    print(f"{hashlib.sha256(raw).hexdigest()}  {m} [{rows},{cols}] exceptions={len(values)} nan={nan} negative={neg}")
