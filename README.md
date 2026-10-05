# DeepSeek-V4.1-Flash on TensorFold, 4x NVIDIA DGX Spark

Serve DeepSeek-V4.1-Flash (the 2.9 bpw EXL3 pack) across **four** NVIDIA DGX Sparks, tensor-parallel (TP=4) over a
switched CX7 RoCE fabric, behind an OpenAI-compatible API: exact DSpark speculative decoding, Engram rows from local
NVMe, 4 request slots of up to 300K tokens, NVMe sessions, structured output and DSML tool calls.

This is the four-Spark fork of **[jayleaton/deepseek-v41-tensorfold-spark](https://github.com/jayleaton/deepseek-v41-tensorfold-spark)**.
The engine, the DeepSeek-V4.1-Flash family and nearly everything in this repository are Jay Leaton's work on
[TensorFold](https://github.com/ashhart/TensorFold); this fork adds what four Sparks need (`patches/0003`-`0005`), a
four-node launcher, and the measurements. **Two Sparks? Use Jay's repository**: it is the maintained two-Spark recipe
and has moved on since this fork branched (below).

> Measured on one cluster (four GB10s, one switch), one boot a configuration. Read [What is not solved](#what-is-not-solved)
> before relying on it.

## Results (four Sparks)

Weights: [`dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw).
Configuration: [`config/prod.env.example`](config/prod.env.example) + [`config/tp4.env.example`](config/tp4.env.example)
(expert pruning on, `TF_DSV41_EXPERT_TOPP=0.85`), thinking off, temperature 0, 512-token replies, isolated runs.

### Decode (2026-10-04)

| prompt | 1k | 20k | 40k | 80k | 160k | geometric mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prose, tok/s | 63.6 | 65.7 | 64.0 | 62.8 | 61.0 | **63.4** |
| code, tok/s | 104.9 | 97.0 | 96.4 | 103.1 | 99.8 | **100.2** |

Median of 3 a cell. Four streams at once: **119.2 tok/s** aggregate (`dsbench`, single-stream median 62.6).

### Reading a new prompt (2026-10-05)

| cold time to first token | 20k prompt | 160k prompt |
| --- | ---: | ---: |
| `0003` alone (2026-10-04) | 12.9-14.2 s | 97.4-99.8 s |
| + split selections and overlapped exchanges | 9.4-9.5 s | 74.6-75.2 s |
| **+ pipelined prompt reading (the shipped config)** | **5.8-6.8 s** | **38.8-39.6 s** |

Each build started with an empty session cache (every prompt read cold). With the pipeline a cold 160k-token prompt
reads at ~4,100 tokens/s; before it, TP=4 read prompts ~15-25% slower than two Sparks do (this repository's two-Spark
row: 1,953 tokens/s at 128K), because every rank recomputes the parts of a layer that tensor parallelism does not
split and waits at 43 exchanges a chunk. How the pipeline works: [docs/PREFILL_SPEED.md](docs/PREFILL_SPEED.md).
Warm repeats reuse the prefix: 0.26-0.64 s at every depth.

### Checks

- On the running server: arithmetic, forced tool call (`tool_choice=required`), tool continuation, strict JSON
  schema at T = 0 / 0.7 / 1.0, reasoning, and `scripts/canary.py` (chat, thinking, tool, json, tokenize): all pass.
- A code word hidden at 30 / 60 / 85% of 20k / 80k / 158k-token prompts: found 3 / 3 (pipelined and not).
- On the CPU (four ranks as threads, exact numerics): four ranks == one rank, the ranks agree bit for bit, a
  pipelined prompt leaves the same state as the tensor-parallel one, greedy / top-k / nucleus decoding on four ranks
  == one rank token for token (`tests/test_dsv41_tp4.py`, `test_dsv41_pipe.py`, `test_dsv41_four_sparks.py`).

Raw files: [`results/four-sparks-20261004/`](results/four-sparks-20261004/README.md),
[`results/prefill-speed-20261005/`](results/prefill-speed-20261005/README.md),
[`results/four-spark-fixes-20261005/`](results/four-spark-fixes-20261005/README.md). The same cluster's earlier deployments
(SGLang on the FP8 checkpoint, vLLM) and a dated history of every change: [spark-bench](https://github.com/neko-legends/spark-bench).

## What four Sparks change

| part | two Sparks (a rank) | four Sparks (a rank) |
| --- | --- | --- |
| query heads | 32 | 16 |
| routed + shared experts (intermediate 2,304) | 1,152 | 640 / 640 / 512 / 512 |
| vocabulary (head columns, embedding rows) | 64,640 | 32,384 / 32,384 / 32,256 / 32,256 |
| Engram hash heads | 12 | 6 (`engram-l{1,14}-r{rank}of4.bin`) |
| resident weights | ~99 GiB | ~56 GiB (+~27 GiB for the prompt pipeline's full-width layers) |

Every split stays on whole 128-wide Hadamard blocks (EXL3 rotations act per block), so the widths that do not divide
by four are uneven; every exchange sends the same size from every rank. Details: [docs/FOUR_SPARKS.md](docs/FOUR_SPARKS.md).

## Quick start

Requirements: four DGX Sparks on one switch, one IPv4 subnet per CX7 port function, Docker with the NVIDIA runtime on
each, passwordless ssh from the head (rank 0) to the three workers over the link; on each node's local NVMe ~200 GB
for the weights, ~47 GB for its Engram shards, ~50 GB for the prepared rank folder, and room for the session tier.
Nothing else on the GPUs.

```bash
git clone --recurse-submodules https://github.com/neko-legends/deepseek-v41-tensorfold-spark.git
cd deepseek-v41-tensorfold-spark
cp config/prod.env.example config/prod.env        # the engine knobs
cp config/tp4.env.example config/tp4.env          # every <placeholder>: workers, link addresses, paths

# 1. weights: the same EXL3 pack on all four nodes, same path
hf download dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw --local-dir <MODEL>

# 2. Engram shards, rank r on node r (from DeepSeek's original checkpoint: docs/FOUR_SPARKS.md)
python3 scripts/pack_engram.py --src <deepseek-ai/DeepSeek-V4.1-Flash dir> --config <MODEL>/config.json \
    --out <ENGRAM> --world 4 --rank <r>

# 3. the image (TensorFold v0.6.0 + patches/*), shipped to the workers, CUDA extensions prebuilt on all four
docker build -f docker/Dockerfile -t dsv41-tensorfold:tp4 .
bash scripts/serve4.sh ship
bash scripts/serve4.sh prebuild

# 4. serve (the first start writes the prepared rank folders; later starts load in ~35 s)
bash scripts/serve4.sh start
bash scripts/serve4.sh status | logs [R] | stop
```

Then:

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "DeepSeek-V4.1-Flash-TF",
  "messages": [{"role": "user", "content": "What is 17 * 23?"}], "reasoning_effort": "low"}'
```

Optional: `scripts/keeper4.sh` from cron restarts the four ranks after a reboot or three failed health checks.
Setup notes and lessons from the first boots: [docs/FOUR_SPARKS.md](docs/FOUR_SPARKS.md). Knobs and memory gates:
[docs/OPERATIONS.md](docs/OPERATIONS.md).

## API

OpenAI-compatible on `HOST:PORT` (`127.0.0.1:8000` by default: put your own proxy and authentication in front):
`/v1/chat/completions` (streaming, tool calls, `response_format`), `/v1/completions` (text or token ids),
`/tokenize`, `/v1/models`, `/health`, `/metrics`. Thinking follows DeepSeek-V4.1's encoding and is on by default
(`TF_DSV41_THINKING=0` turns it off); `reasoning_effort` `none` / `low` / `medium` / `high` / `max` or 1-100. The
full table: [docs/TWO_SPARKS.md#api](docs/TWO_SPARKS.md#api) (unchanged at four Sparks).

## The engine

`vendor/TensorFold` is upstream TensorFold v0.6.0, unmodified; the Dockerfile applies `patches/` in order:

| patch | from | what |
| --- | --- | --- |
| [`0001-spark-stack-060.patch`](patches/0001-spark-stack-060.patch) | jayleaton | the GLM-5.3-Flash Spark engine stack rebased onto 0.6.0: communicator, RoCE all-gather, server pieces |
| [`0002-deepseek-v41-family.patch`](patches/0002-deepseek-v41-family.patch) | jayleaton | the DeepSeek-V4.1-Flash family (engine G13, `767ad9f`) |
| [`0003-four-sparks.patch`](patches/0003-four-sparks.patch) | this fork | TP=4: whole-block uneven splits, `--tp 4 --rank 0..3`, N-way rank agreement and plan link, RoCE post rotation, `/health` draft counters, sync-free expert counts, opt-in prefill profile |
| [`0004-prefill-speed.patch`](patches/0004-prefill-speed.patch) | this fork | pipelined prompt reading across the four ranks, split selections, overlapped exchanges (all opt-in; on in `tp4.env.example`) |
| [`0005-four-spark-fixes.patch`](patches/0005-four-spark-fixes.patch) | this fork | the uneven-slice bugs jayleaton's review of the TP=4 port found (below) |

### Fixed in 0005 (2026-10-05)

Jay reviewed the TP=4 port ([PR #6](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/pull/6)) and found
bugs that only uneven slices or more than two ranks expose. All fixed, with tests in `tests/test_dsv41_four_sparks.py`
(each review bug's test fails without its fix), and checked on the four Sparks (2026-10-05; serving since 11:30):

- the four benchmark replies (20k and 160k, prose and code) are byte-identical to the 0004 build's; cold first token
  5.8-6.8 s (20k) and 38.8-39.0 s (160k), as before; decode the same as 0004 in an A/B in one window;
- the hidden code word found 3 / 3, the short gates 7 / 7;
- sampled requests past the narrow slices pass: nucleus rows, nucleus with a JSON schema (full-vocabulary
  candidates), `top_k` 32,000 and 40,000, including `top_k` 40,000 as the first request after a boot.

The fixes:

- **Candidates past the narrow vocabulary slices**: every rank took `min(count, its own width)` candidates, so a
  count above 32,256 (`top_k` above that, or masked nucleus rows, which ask for the whole vocabulary) all-gathered
  unequal sizes. On the four Sparks, the 0004 build answered a `top_k` 40,000 request with an illegal memory access
  on ranks 2 and 3 (the narrow slices) and the server stopped. Every rank now sends the widest slice's count,
  narrower slices padded with entries that sort after every real token (`pick.rank_top`).
- **Trimmed draft head** (`TF_DSV41_DRAFT_HEAD=trim`): the per-rank id lists assumed equal slices; rank 2 refused to
  boot. They now follow the 128-block split.
- **TCP plan link** (`TF_DSV41_PLAN_LINK=tcp`): followers now say their rank and world size; rank 0 refuses strays,
  duplicates and another world size, orders the connections by rank, and names a rank that never connects.
- **`--tp 4`** is refused for families that do not declare it (`CUDA_TP`), instead of starting an engine that takes
  neither path.
- **`TF_DSV41_PREFILL_ATTN_BMQ=32`** (the two-Spark production value) on a 16-head rank now runs the fused prefill
  attention with 16 heads a program instead of silently falling back to the chunk kernels.
- **NVMe session tier**: the world size is part of the directory's identity, so an entry written by two Sparks is
  never resumed by four.
- **The first very wide window** (found while testing the above on the four Sparks): with the gather fixed, a
  sampled request with `top_k` 32,000-40,000 as the server's first wide request still timed out every follower's
  Engram gate at layer 14 (4 of 4 runs); runs whose first wide request was a JSON nucleus request passed. Most
  likely cause: the window needed the large-`k` top-k and large-row sort kernels for the first time, and CUDA's lazy
  module loading loaded them while the forward still waited on host threads (the Engram gate, the RoCE exchanges).
  The candidate kernels of every width class now run once at boot on every rank (`cand_warm`), and the candidates'
  pinned buffer grows before a window's forward is queued (`cand_reserve`): since then the same requests pass in
  every run (2 of 2, one with `top_k` 40,000 as the first request after a boot). Not fully explained: the 0004 build
  once answered a first `top_k` 32,000 request without the warm-up.
- The gate scorer and DSpark delta shards (both off in production) now handle any rank count and uneven splits.

## What is not solved

- **Jay's G14-G19 are not in this fork.** This fork branched at engine G13; his newer main (image input, fail-fast
  across ranks, adaptive prefill, an admission floor, RoCE link changes, faster decode on two Sparks) assumes two
  ranks in places and needs porting to four.
- The boot memory budget (`memory.load_check`) is anchored on two-Spark measurements: conservative on ranks 0 and 1,
  skipped on ranks 2 and 3. The live floor is MemAvailable ~26 GB a node with the pipeline on.
- A pipelined prompt's bits are one rank's arithmetic, not the four-rank sum's (its own session tag): as different
  as the engine's own fast vs exact prefill kernels. Exact numerics give the same state either way.
- Short prompts gain less from the pipeline (it fills in four steps): cold 20k is 5.8-6.8 s.
- One cluster, one boot a configuration; expert pruning is lossy (delete its three lines for the unpruned model).

## Documentation

| | |
| --- | --- |
| [FOUR_SPARKS](docs/FOUR_SPARKS.md) | the TP=4 port: what changes, setup, first-boot lessons, results |
| [PREFILL_SPEED](docs/PREFILL_SPEED.md) | the prompt pipeline, split selections, overlapped exchanges, what did not work |
| [OPERATIONS](docs/OPERATIONS.md) | settings, memory gates, turning each lever off |
| [TWO_SPARKS](docs/TWO_SPARKS.md) | Jay's two-Spark README at the fork point: method, quality tables, strict mode, credits |
| [ENGINE](docs/ENGINE.md) / [ARCHITECTURE](docs/ARCHITECTURE.md) / [DECODE](docs/DECODE.md) | the patches, the model split, the decode roofline |
| [campaign/](docs/campaign/README.md) | Jay's development log, windows G1-G13 |

## Licensing

Apache License 2.0 for this project's own code, patches, scripts, benchmarks and docs ([`LICENSE`](LICENSE),
[`NOTICE`](NOTICE)): keep the copyright lines and the NOTICE attributions and state your changes. This fork's changes
are listed in [`NOTICE`](NOTICE). TensorFold (`vendor/TensorFold`, unmodified) is Apache-2.0 from 0.6.0; the code the
patches modify stays under its license. Files the patches add keep their own SPDX notice (the DeepSeek-V4.1-Flash
family: MIT, Copyright (c) 2026 Jay Leaton). Third-party code and the model weights: the full table in
[docs/TWO_SPARKS.md#licensing](docs/TWO_SPARKS.md#licensing). The uncensored weights have refusals removed; you are
responsible for how you use them.

## Credits

- **[Jay Leaton (jayleaton)](https://github.com/jayleaton)**: the DeepSeek-V4.1-Flash engine for TensorFold and the
  two-Spark recipe this fork extends, and the review that found the bugs 0005 fixes. Follow him on
  [X](https://x.com/jayleaton) / support him on [Buy Me a Coffee](https://buymeacoffee.com/jayleaton).
- [Ash Hart (ashhart) / TensorFold](https://github.com/ashhart/TensorFold): the engine, the EXL3 kernels, the drafting
  and the server.
- [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash): the model, its encoding, and the CED / CSA2 /
  mHC / DSpark / Engram designs.
- [dealignai](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw) (the uncensored pack),
  [Mia-AiLab](https://huggingface.co/Mia-AiLab) (the 2.9 bpw EXL3 pack), [turboderp / ExLlamaV3](https://github.com/turboderp-org/exllamav3)
  (EXL3), and everyone credited in [docs/TWO_SPARKS.md#credits](docs/TWO_SPARKS.md#credits).
