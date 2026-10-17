# Synthetic csf_32x2_v1 planes made by B12X's own encoder (_from_logical_components) for the Zig unit tests:
# per case, a little-endian file: u32 rows, cols, n_exc; bases[rows]; codes; page_bases u32[]; offsets u16[]; exceptions u16[]; logical[rows*cols].
import os, struct, sys
import numpy as np
ref_dir, out_dir = sys.argv[1], sys.argv[2]
sys.path.insert(0, ref_dir)
import _csf_blocks as cb
os.makedirs(out_dir, exist_ok=True)
rng = np.random.default_rng(53)
for rows, cols in [(32, 1), (32, 2), (64, 5), (160, 9), (256, 8), (128, 4), (288, 70)]:
    bases = rng.integers(0, 241, rows, dtype=np.uint8)
    logical = (bases[:, None].astype(np.int32) + rng.integers(0, 16, (rows, cols))).astype(np.uint8)
    out = rng.random((rows, cols)) < 0.07
    logical[out] = rng.integers(0, 256, int(out.sum()), dtype=np.uint8)
    offsets = logical.astype(np.int16) - bases[:, None]
    exc = (offsets < 0) | (offsets >= 16)
    positions = np.flatnonzero(exc)
    values = logical.ravel()[positions]
    offsets[exc] = 0
    p = cb._from_logical_components(bases, offsets.astype(np.uint8), positions, values, "nvfp4")
    p.validate()
    b, f, pos, v = cb.logical_csf_components(p)
    back = (b[:, None] + f).astype(np.uint8); back.ravel()[pos] = v
    assert np.array_equal(back, logical)
    blob = struct.pack("<III", rows, cols, p.exceptions.size) + p.bases.tobytes() + p.codes.tobytes() + \
        p.page_bases.tobytes() + p.offsets.tobytes() + p.exceptions.tobytes() + logical.tobytes()
    open(os.path.join(out_dir, f"csf_{rows}x{cols}.bin"), "wb").write(blob)
    print(rows, cols, "exceptions", p.exceptions.size, "pages", p.page_bases.size, "bytes", len(blob))
