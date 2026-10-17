# engine/zig-glm — vendored upstream TensorFold v1.0.12 (ashhart/TensorFold, tag v1.0.12)

Pinned: tag `v1.0.12` (tag object 9264fe89, commit 833bbfcb, 2026-10-16). `ash/main` at vendoring time (6ee828c2) has
the same `zig/`, `build.zig` and `build.zig.zon` (the eight commits past the tag are docs only).

Base for the GLM-5.3-Flash FP4-Local port (nvidia/GLM-5.3-Flash-FP4-Local, TP-4 on four DGX Sparks). It carries
upstream's native CUDA GLM-5.3-Flash family as shipped (commit 2675b30b and follow-ups 00536217, ee39bfdc, 231053e8:
`zig/src/families/glm/cuda_*.zig`, `zig/kernels/cuda/glm_{kda,glue,ops,quant}.cu`, entry `cuda_native.zig`, registered
in `zig/src/native/cuda.zig`): KDA, the DSA indexer, mHC, MTP, prompt cache and NCCL tensor parallelism. All of that is
upstream's work (TensorFold, ashhart and contributors; LICENSE / NOTICE / THIRD_PARTY_NOTICES.md as shipped).

Dropped from the vendored copy (not needed to build the engine): `app/`, `assets/`, `.github/`, `.gitignore`,
`CONTRIBUTING.md`, `RELEASE-NOTES-1.0.0.md`, `RUNBOOK.md`, `SECURITY.md`, `ZIG-PREVIEW.md`, `setup.py` — the same set
engine/zig-nemotron dropped.

Ours on top (listed in GLM-PORT-LOG.md): `zig/src/fabric/tp/` + `zig/src/fabric/tp_verbs.zig` + `zig/build/tp.zig`
(the RoCE/NCCL TP collectives of the DeepSeek TP-4 tree, as carried in engine/zig-nemotron), and the FP4-Local format
work (MIXED_PRECISION admit, csf_32x2_v1 scales, MXFP8 / FP8 block weights, W4A16 MTP experts).
The DeepSeek engine (../zig) and the Nemotron engine (../zig-nemotron) are untouched.
