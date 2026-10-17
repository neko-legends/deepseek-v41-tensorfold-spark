# GLM-5.3-Flash FP4-Local on engine/zig-glm — first GPU window agenda

Owner of the window: the orchestrator (it stops DeepSeek, holds the Sparks lock, restores DeepSeek after). Nothing here
runs outside a window. Hosts are named spark1..spark4 (rank 0..3, rank order of `sparks.sh`); `$MASTER` is rank 0's
RoCE link address, `$MODEL` = `$HOME/models/GLM-5.3-Flash-FP4-Local` (fanned out and sha256-verified on all four).

Binaries (aarch64, sm_121, CUDA 13.0.88 nvcc, stripped), staged and sha256-checked on every Spark:
`$HOME/glm53-tf/bin-e81ca0ea/{tensorfold-native,tf-cuda-test}` (`SHA256SUMS` beside them; built from branch
glm53-tensorfold-20261017 at e81ca0ea: the FP4-Local loader, the server API parity, structured output, the CUDA
image tower; `bin-9f41dec8` is the same without vision, `bin-0eb6952f` the loader alone). Libraries in `$HOME/glm53-tf/lib/` (`SHA256SUMS`): `libnccl.so.2` (NCCL 2.32.3,
the Nemotron runs' copy; upstream's GLM ranks dlopen it) and `libtfgrammar.so` (structured output, found as
`../lib` beside the binary's folder). A CPU reference venv (torch 2.14.1+cpu, transformers 5.20.1) with
`tools/glm` and B12X's `_csf_blocks.py` lives on spark2 under `$HOME/glm53-tf/{refenv,tools,b12x}`.

Driver: `tools/glm/window/glmw.sh` (subcommands pre, glmref, thin, tp, wait, gate, stop, status), configured by a
file outside the repo (`GLMW_ENV`: hosts, RoCE address/netdev/HCA, paths). Every step below is one or two of its
subcommands; reference answers were recorded before the window (`tools/glm/ref_tokens.py record`, the Phase-2 engine
serving), so the window needs no reference engine.

Common env for every step: `B=$HOME/glm53-tf/bin-0eb6952f  W=$HOME/glm53-tf/runs/$(date +%m%d-%H%M)`.

## 0. Before (2 min)
- `free -g` on each Spark after DeepSeek is down: expect >= 100 GiB available; `nvidia-smi` shows no process.
- `cd $B && sha256sum -c SHA256SUMS`.

## 1. Kernels on one GPU, synthetic (spark1, ~2 min)
- `$B/tf-cuda-test glm-ref` -> `glm-ref: every projection within ... of its float64 tolerance` (upstream's gate).
- `$B/tf-cuda-test glm-synth $W/synth && $B/tf-cuda-test glm-lanes-concurrent $W/synth $W/synth/dc.json 0 1 <idfile>`
  (optional; upstream's lane checks on a synthetic checkpoint; the idfile is any small ids file, see upstream docs).

## 2. Thin load on one rank (spark1, ~5 min) — the FP4-Local loader
```
TF_GLM_RANK=0 TF_GLM_WORLD=1 TF_GLM_LAYERS=4 LD_LIBRARY_PATH=$HOME/glm53-tf/lib \
  $B/tensorfold-native serve $MODEL --host 127.0.0.1 --port 8190 --no-update-check --context 4096 --parallel 1 --no-drafts
```
Check: config admitted (no `MIXED_PRECISION` refusal), layers 0-3 load (KDA MXFP8 widened, dense MXFP8 on B16, DSA
FP8_PB_WO, MoE csf experts), load time, `nvidia-smi` memory (~4.1 GB of experts a MoE layer at TP-1), one greedy
chat completion returns (text is meaningless at 4 layers). Then `TF_GLM_LAYERS=8` once (two MoE layers more).
Failure modes to read in the log: `csf_32x2_v1 scales do not decode`, `neither MXFP8 nor FP8_PB_WO`, `BadNvfp4`,
`checkpoint has no tensor` — each names the tensor.

## 3. Per-layer numerics gate vs HF (spark1 GPU + CPU reference, ~20 min)
Run the thin server with `TF_GLM_DUMP=$W/dump` (and `TF_GLM_LAYERS=8`), send ONE greedy request of a short prompt
(< 2,048 tokens, e.g. 40-200 tokens, `max_tokens: 1`), note `usage.prompt_tokens` = ROWS, stop the server. The dump has
`000_h.bin .. 008_h.bin` (the prompt window first, then any decode rows appended).
Reference (CPU, any Spark, a venv with torch (CPU) + transformers==5.20.1 + safetensors + numpy, and B12X's
`_csf_blocks.py` at flashinfer 7e203806 in `$W/b12x`):
```
python3 tools/glm/layer_ref.py $MODEL $W/dump --rows ROWS --layers 0,1,2,3,4,5,6,7 --b12x $W/b12x
```
Pass: per-layer cos >= 0.999 for KDA and DSA layers, >= 0.998 for MoE layers (bf16 kernels vs an fp32 reference;
Nemotron's gate gave 0.9994-0.9999). Look at layer 0's result first (KDA + dense on B16, the newest code) and at the
F32-norm effect: if cos sags evenly on every layer, try the reference with `--dtype bf16` (it then rounds the norms as
the engine does) to tell the norm rounding from a real fault.

## 4. TP-4 full load (all four, ~10 min)
On each Spark r = 0..3 (rank 0 last is not needed; NCCL waits):
```
TF_GLM_RANK=$r TF_GLM_WORLD=4 NCCL_COMM_ID=$MASTER:29660 NCCL_IB_HCA=roceP2p1s0f1 NCCL_SOCKET_IFNAME=enP2p1s0f1np1 \
  LD_LIBRARY_PATH=$HOME/glm53-tf/lib \
  $B/tensorfold-native serve $MODEL --host 0.0.0.0 --port 8190 --no-update-check --context 32768 --parallel 1 --no-drafts
```
Check: every rank loads 45 layers (expect ~46.5 GiB of weights a rank, GLM-PORT-LOG.md), load time (csf decode is
~1 min a rank on the host), rank 0 serves; greedy chat on 3 prompts gives coherent text; a 4K-token prompt; tokens/s
decode and prefill (first numbers only). Then `TF_GLM_DUMP` at TP-4 with `TF_GLM_LAYERS=45` is NOT needed: compare the
TP-4 top-1 tokens / logprobs with the Phase-2 vLLM engine on the same prompts instead (greedy, temperature 0).

## 5. MTP (all four, ~5 min)
Same as 4 without `--no-drafts` (on CUDA the MTP head loads exactly when drafts are on; `TF_GLM_MTP` is the Python engine's): loads the W4A16 MTP experts as stored; a drafted greedy
reply must equal the `"draft": false` reply; note acceptance per round.

## 5b. CUDA vision (spark1, ~5 min, then TP-4 with `GLMW_VISION=1`)
- `glmw.sh vision $HOME/glm53-tf/vision-ref/224-all $HOME/glm53-tf/vision-ref/img224.png`: the processor's patches vs
  HF's, the patch embedding, each of the 24 blocks from the reference's own input (gate cos >= 0.999), the tail
  (post norm, downsample, merger; >= 0.999), and the end-to-end rows (reported only: the reference's late blocks
  hinge on near ties between "sink" patches, bf16 vs fp32 rows 0.9916). Then the 640 dump (only blocks 0/11/23).
- TP-4 with `--vision`: the orchestrator's verify.py image line ("red square, blue circle"), an image prompt twice
  (prompt cache keyed by the image hash).

## 6. Server API on our TP-4 server (~5 min)
- `tools/glm/ref_tokens.py compare http://RANK0:8190 ref-tokens.json`: greedy tokens vs the reference's (near ties
  within 0.5 nats reported apart).
- The orchestrator's `verify.py http://RANK0:8190/v1`: thinking default/off, stream, JSON schema (structured output:
  first_masked + masked windows), tools with and without thinking (the image line fails until CUDA vision).
- `tools/glm/api_parity.py` ran before the window on `fake_serve` (template, parsers, stream framing): only the engine
  is new here.
- Prompt cache under TP: the same long prompt twice (`usage.prompt_tokens_details.cached_tokens` > 0 the second
  time, the same reply), then a shared prefix with a different question.

## Stop / restore
Stop the servers (SIGTERM, then check `nvidia-smi` is empty on all four), hand back to the orchestrator for DeepSeek.
Copy `$W` logs to the mission folder. Nothing in this agenda writes outside `$HOME/glm53-tf/`.
