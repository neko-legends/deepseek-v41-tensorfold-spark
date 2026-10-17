# GLM-5.3-Flash FP4-Local on TensorFold (Zig, CUDA, TP-4 on four DGX Sparks) — port log (branch glm53-tensorfold-20261017)

Base: upstream TensorFold v1.0.12 vendored here (VENDORED.md). Target: `nvidia/GLM-5.3-Flash-FP4-Local`
(`Glm5NextForConditionalGeneration`, model_type `glm5_next`): 45 layers = 34 KDA (linear attention) + 11 DSA-MLA
(layers 3, 7, ..., 43), dense MLP on layers 0-2, 288 routed experts top-8 + 1 shared on layers 3-44, mHC (hc_mult 4,
Sinkhorn 20), one MTP layer (index 45, DSA + MoE), vocab 154,880, hidden 4,096. 177.4 GB in 40 shards plus
`model-inputscales.safetensors` (36,288 expert `input_scale`) and `tensors/native-QB.safetensors` (214 BF16 KDA gate,
DSA indexer and MLA kv_b tensors). 295,041 tensors (547 vision).

## Who did what

**Upstream (TensorFold, ashhart and contributors; v1.0.12, commit 2675b30b and follow-ups 00536217, ee39bfdc,
231053e8):** the whole GLM-5.3-Flash CUDA family: config, TP shares (`cuda_shard.zig`), the loader for
nvidia/GLM-5.3-Flash-NVFP4 (`cuda_weights.zig`), every kernel (KDA chain, DSA indexer and sparse/dense latent attention,
mHC pre/post, router, grouped NVFP4 experts, B16 matmuls, sampler, MTP head and 4-bit draft head), the forward, prompt
cache, lanes, NCCL TP over hosts (`TF_GLM_RANK` / `TF_GLM_WORLD` / `NCCL_COMM_ID`) and the server. Validated upstream
on 4 layers at TP=2 only; the full checkpoint was never served (CHANGELOG #804).

**Ours (this branch):** making that family read the FP4-Local checkpoint, and the checks around it:

| piece | file | what |
|---|---|---|
| MIXED_PRECISION admit | `zig/src/core/modelopt.zig` | `admit` accepts `quant_algo: MIXED_PRECISION` when every `quantized_layers` entry is NVFP4 g16 (plain or `csf_32x2_v1`), W4A16_NVFP4 g16, MXFP8 1x32 or FP8_PB_WO 128x128; anything else refused by name. FP4-Local's config is a fixture (42 / 77 / 205 / 170 entries). |
| csf_32x2_v1 decoder | `zig/src/core/csf.zig` | port of FlashInfer B12X's decode (local-inference-lab/flashinfer `flashinfer/experimental/b12x/loader/_csf_blocks.py` @7e203806: `CsfBlockPlane`, `logical_csf_components`) to the plain e4m3 [n, k/16] plane upstream's NVFP4 path takes. Structure checks (sizes, page prefixes, sorted in-bounds records) refuse a bad plane. |
| FP8 widening | `zig/src/core/fp8w.zig` | MXFP8 (e4m3 x 2^(ue8m0-127) per 32, exact in bf16) and FP8_PB_WO (e4m3 x fp32 per 128x128 block, RNE to bf16) to bf16 rows. |
| loader | `zig/src/families/glm/cuda_weights.zig` | `get()` widens any e4m3 tensor by its `_scale` (lm_head: the rank's rows only); routed experts read via `nvfp4Linear` (csf or plain scales, `input_scale` optional); the MTP layer's W4A16 experts are read as stored (upstream quantizes BF16 ones); MXFP8 dense layers widened to bf16 [gate \| up] + down. |
| dense MLP on B16 | `cuda_mlp.zig`, `cuda_state.zig` | `Layer.b16` runs layers 0-2 on the B16 matmuls exactly as the shared expert runs (gu, SwiGLU, down fp32 partials); `mm_part` + `b16Shapes` sized for it. |
| dump hook | `cuda_forward.zig` | `TF_GLM_DUMP=DIR`: residual streams entering each layer (bf16 [rows][4][4096], rank 0, unpieced loop) appended to `DIR/NNN_h.bin`. |
| TP collectives | `zig/src/fabric/tp/`, `tp_verbs.zig`, `zig/build/tp.zig` | our RoCE / NCCL TP layer from the DeepSeek TP-4 tree (as carried in engine/zig-nemotron); imported by the glm module, not yet called (upstream's NCCL path serves first). |
| host checks | `zig build test-glm` | family host tests + csf / fp8w / modelopt tests; the loader and the forward are forced through semantic analysis on the host. |
| tools | `../../tools/glm/` | `csf_check.sh` (bit-exact vs B12X on real planes), `coverage.py` (loader reads vs index), `layer_ref.py` (HF per-layer gate), `dequant_vs_nvfp4.py`, `hf_headers.py` / `hf_tensor.py` (headers and tensors by HTTP range). |

## Tensor map: FP4-Local -> what upstream's loader expects -> what we do (TP-4 cut, kernel)

| tensors (count) | FP4-Local stores | upstream expects (NVFP4 repo) | ours at load | TP-4 cut | kernel (owner) |
|---|---|---|---|---|---|
| embed_tokens | BF16 [154880, 4096] | BF16 | as is | whole | embed (up) |
| lm_head | MXFP8 e4m3 + U8 [154880, 128] | BF16 | rows lo..hi widened (exact) | vocab rows / 4 = 38,720 | B16 matmul (up) |
| norm.weight, post_attention_layernorm x45 | **F32** | BF16 | rounded to bf16 (upstream's kernels take bf16) — see risks | whole | rmsnorm (up) |
| input_layernorm x46, q_a/kv_a layernorm, o_norm, k_norm w/b | BF16 | BF16 | as is | whole | (up) |
| hc_{attn,ffn}_{fn,base,scale} x45 | fn BF16, base/scale **F32** | BF16 | fn bf16, base/scale f32 (as upstream widens) | whole | hcPre/hcPost (up) |
| KDA q/k/v_proj x34 | MXFP8 [8192, 4096] | BF16 | widened (exact) | heads: 16 of 64 (2,048 rows) | B16 `kda.proj` (up) |
| KDA o_proj x34 | MXFP8 [4096, 8192] | BF16 | widened | columns 2,048 | B16 (up) |
| KDA f_a/g_a [128,4096], b [64,4096], f_b/g_b [8192,128] | BF16 (native-QB) | BF16 | as is | f_a/g_a whole; b, f_b, g_b by heads | (up) |
| KDA q/k/v_conv1d [8192,1,4] | **BF16** | F32 | widened to f32 (exact) | by heads | KDA chain (up) |
| KDA A_log [64], dt_bias [8192] | F32 | F32 | as is | by heads | (up) |
| MLA q_a_proj, kv_a_proj_with_mqa x11 | FP8_PB_WO + F32 [nb,1,kb,1] | BF16 | widened (RNE) | whole | B16 `dsa.proj` (up) |
| MLA q_b_proj x11 | FP8_PB_WO [16384, 1536] | BF16 | widened | heads 16 (4,096 rows) | B16 (up) |
| MLA kv_b_proj x11 | BF16 (native-QB) [32768, 512] | BF16 | as is | per head wk / wv | absorb (up) |
| MLA o_proj x11 | FP8_PB_WO [4096, 16384] | BF16 | widened | columns 4,096 | B16 (up) |
| DSA indexer wk / weights_proj / wq_b / kpool gate, ape x11 | BF16 (native-QB) | BF16 | as is | whole | indexer (up) |
| dense gate/up/down x3 (layers 0-2) | MXFP8 [12288, 4096] / [4096, 12288] | **NVFP4** | widened to bf16 [gate\|up], down | 3,072 of 12,288 | **B16 (ours: new use)** |
| router gate.weight / e_score_correction_bias x43 | BF16 / F32 | BF16 / F32 | as is | whole | router (up) |
| routed experts x42 layers x288 x3 | NVFP4 U8 codes + **csf_32x2_v1** scales + scale_2 + input_scale | NVFP4 plain e4m3 scales | csf decoded on the host to plain e4m3 (bit-exact vs B12X), then upstream's packers | each expert's 2,048 -> 512 (gate/up rows, down cols) | grouped NVFP4, bf16 activations (up) |
| shared experts x42 | FP8_PB_WO | BF16 | widened | 512 of 2,048 | B16 (up) |
| MTP layer 45: attention + indexer + kv_b | MXFP8 (g32) | BF16 | widened (exact) | as the DSA layers | (up) |
| MTP layer 45: shared experts | MXFP8 | BF16 | widened | 512 | B16 (up) |
| MTP layer 45: routed experts x288 | **W4A16_NVFP4** plain e4m3 scales, no input_scale | BF16 (quantized on the GPU) | read as stored (no requantization) | 512 | grouped NVFP4 (up) |
| MTP enorm / hnorm / eh_proj / shared_head.norm | BF16 | BF16 | as is | whole | (up); draft head = q4 of the rank's head rows (up) |
| model.visual.* (547) | MXFP8 attn, W4A16 MLP/merger, BF16 rest | — | not read (no CUDA vision yet) | — | — |

`tools/glm/coverage.py` mirrors these rules over the real index: 294,494 text tensors read, 0 missing, 0 left unread.

## Per-rank memory at TP-4 (weights, from the real headers; `tools/glm` estimate)

| | now (FP8 widened to bf16) | with native FP8 (later) |
|---|---|---|
| routed experts NVFP4 (incl. MTP) | 40.82 GiB | 40.82 GiB |
| FP8 tensors, split (KDA/MLA/shared/dense) | 3.36 GiB | 1.68 GiB |
| embedding bf16 (whole on every rank) | 1.18 GiB | 1.18 GiB |
| other bf16/f32 whole + split, FP8 whole (q_a/kv_a, indexer...) | 0.77 GiB | 0.67 GiB |
| lm_head rows + MTP 4-bit draft head | 0.38 GiB | 0.23 GiB |
| **total** | **46.5 GiB** | **44.6 GiB** |

A Spark has ~112 GiB usable unified memory: about 60 GiB per rank stays for KV / KDA states, buffers, the prompt cache
and the OS, so widening FP8 costs memory we have. It costs decode bandwidth, though: the widened KDA q/k/v/o alone are
~2.3 GB a token a rank in bf16 vs ~1.1 GB in FP8 (more than the ~1.2 GB of routed experts a token reads), so native
FP8 matmuls are the first speed item once numerics pass.

## Verified (host only, no GPU)

- csf_32x2_v1: 48 random real expert planes (layers 3-44, gate/up/down) decode bit-exact (sha256) vs B12X's own
  decoder (`tools/glm/csf_check.sh`); 7 synthetic B12X-encoded planes (clipped tiles, odd columns, 2 pages) as unit tests.
- Dequant semantics vs an independent checkpoint (`tools/glm/dequant_vs_nvfp4.py`, against nvidia/GLM-5.3-Flash-NVFP4's
  BF16 attention / shared experts and plain-scale NVFP4 experts): FP8_PB_WO q_a / o_proj, MXFP8 KDA q_proj cos 1.00000
  (rms equal to 4 digits), FP8 shared expert 0.99978, MXFP8 dense down 0.995, csf NVFP4 experts 0.996 (two 4-bit
  quantizations of QAD-related weights), W4A16 MTP experts 0.9955.
- HF reference chain (`tools/glm/layer_ref.py --ids`, transformers 5.20.1, CPU fp32) runs layers 0-3 (KDA + dense,
  DSA + MoE with csf experts on demand) from the real weights, finite.
- `zig build test-glm` 28/28, `zig build test-cuda-host` 39/39 steps; aarch64 cross build (CUDA 13.0.88 nvcc, the
  Sparks' toolkit build, sm_121) of `tensorfold-native` and `tf-cuda-test`.

## Known risks for parity

- F32 norms (final norm, 45 post-attention norms) are not bf16-representable (QAD-trained in f32); upstream's kernels
  take bf16 weights, so they are rounded. Measure in the layer gate; a f32-weight rmsnorm variant is the fix if needed.
- FP8_PB_WO widened with one RNE to bf16 (a native FP8 GEMM would multiply in fp32 instead); MXFP8 widening is exact.
- Routed experts run W4A16 (bf16 activations, upstream's kernels; `input_scale` is read and unused). B12X decodes
  with BF16 activations too but prefills with NVFP4 activations, so prompt logits may differ slightly from vLLM.
- Upstream's own CUDA GLM path was validated on 4 layers at TP=2; full-model and TP-4 behaviour are new ground.

## Not done (needs a GPU window; agenda in ../../tools/glm/WINDOW.md)

Load on one rank (`TF_GLM_LAYERS=N`), `tf-cuda-test glm-ref`, the per-layer HF gate, TP-4 full load and serving,
MTP acceptance. Then: native FP8 matmuls (fp8g exists upstream for 128x128 blocks; MXFP8 needs a UE8M0 kernel),
RoCE collectives (`fabric/tp`) in place of NCCL, CUDA vision, csf decode of only the rank's rows (load time),
our server API bits.

## Stage B (CPU, before the first GPU window)

**Server API vs the reference engine** (vLLM, `--reasoning-parser glm45 --tool-call-parser glm47`, the Phase-2 template
with its thinking switch). Upstream's server already renders GLM's template, parses GLM-4.7 calls and streams usage;
ours on top:
- thinking switch = vLLM's GLM-4.7 / DeepSeek-V3 parser rule: on when `enable_thinking` or `thinking` is on, off when
  either is given (not null) and neither is (`fields.thinkingFields`); off renders `<think></think>`.
- `TENSORFOLD_REASONING_FIELDS=reasoning_content|reasoning|both`: `both` (our DeepSeek server's default; the GLM
  launch sets it) emits vLLM's `reasoning` beside our harnesses' `reasoning_content`, in messages and stream deltas;
  the upstream default stays `reasoning_content` (the frozen goldens: 309/319, 0 unexpected). An earlier reply's
  `reasoning` reaches the template as `reasoning_content`; /v1/messages and /v1/responses read either name.
- `tools/glm/api_parity.py` (the real server on `fake_serve`, the checkpoint's tokenizer + template, against the
  reference): 16/16 prompts token-identical (thinking combos, efforts, system, history, tool specs, tool turns),
  5/5 replies parsed identically (plain, thinking, one call, two calls, thinking + call; streamed and whole; usage
  chunk). Known differences, kept: an effort the template does not name is coerced to the nearest (medium -> High;
  the reference passes it and the template falls back to Max); the reference drops a history `reasoning_content`
  (it reads only `reasoning`), ours renders it as the template intends (clear_thinking false); a literal
  `</arg_value>` inside a value ends it here (vLLM's glm47 reads it as data unless a key or the call's end follows).

**Structured output on the CUDA engine** (JSON schema, required tool calls; upstream's GLM backend had no masked
draw): `first_masked` and window masks; a masked step gathers every rank's logits columns on every rank (the
sampled draw's broadcast; `masked_flag` on verify/shared steps, a `redraw` step for the first token), rank 0
applies the grammar bits (sample.cu's `tf_grammar_mask`) and draws (sampled on the device; greedy on the host,
largest allowed value, lowest id on ties).

**CUDA vision** (`cuda_vision.zig`; plan and HF line references in the stage-B notes): HF Glm5NextVisionModel in
bf16 on rank 0 (weights bf16 at load: MXFP8 exact, W4A16 NVFP4 as code x scale x scale_2 rounded once; new vit.cu
kernels for the RMSNorms, q/k norm + axial 2-D rope, clamped SwiGLU, the 4096-wide LayerNorm, the splice;
`vision_kernel<64,8>`), image rows broadcast to every rank and spliced over the `<|image|>` rows in all four hc
copies (and the MTP head's embedding); `--vision` (default 8000 image tokens, the processor's maximum). The
processor is upstream's Metal-side code, split into `vision_config.zig` / `image_patches.zig` so both share it;
its geometry equals Glm5NextProcessor's on 15 sizes (a test). `tools/glm/vision_ref.py` is the HF CPU reference;
`tf-cuda-test glm-vision MODEL REF [IMG]` gates the tower block by block (the reference's own bf16 vs fp32 runs
agree only to cos 0.99 on the final rows: late "sink" patches hinge on near ties, so the end-to-end number is
reported, not gated). Rank 0 holds ~1.1 GiB of tower weights + ~1.3 GiB scratch at 8000 tokens.
Known gaps: `--vision` / image tokens are not in the ranks' fingerprint (every rank must be started alike); video
(upstream's video.zig frame groups) is not wired on CUDA; an image prompt passes whole (no other stream's round
between its chunks).
