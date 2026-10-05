# Patch 0005 on the four Sparks (2026-10-05)

Measurements behind the README's "Fixed in 0005" section. Four GB10s, the configuration of
`config/prod.env.example` + `config/tp4.env.example`, every build started with an empty session cache. Requests through
the OpenAI API on rank 0; thinking off. Summary tables only (the raw request logs stay on the cluster).

## 1. Same replies, same speed

The depth bench (spark-bench `bench-v41-depth-ab.py` fixtures, 512-token replies, temperature 0), cold:

| prompt | 0004 build: first token | 0005 build: first token | reply identical (sha256) |
| --- | ---: | ---: | --- |
| 20k prose | 11.0 s (first request after boot) | 6.76 s | yes |
| 20k code | 5.82 s | 5.83 s | yes |
| 160k prose | 39.64 s | 38.95 s | yes |
| 160k code | 38.79 s | 38.81 s | yes |

Decode, A/B in one window (each build booted twice, alternating; 1k and 20k prompts, 3 trials: cold, then warm),
tok/s:

| | 1k prose | 1k code | 20k prose | 20k code |
| --- | --- | --- | --- | --- |
| 0004, boot 1 | 53.0 / 55.1 / 54.4 | 87.9 / 90.5 / 90.9 | 49.7 / 55.1 / 55.4 | 80.5 / 86.6 / 86.0 |
| 0005, boot 1 | 53.3 / 55.1 / 55.3 | 87.6 / 89.3 / 90.2 | 49.2 / 55.5 / 55.1 | 79.5 / 85.2 / 84.2 |
| 0004, boot 2 | 52.5 / 55.0 / 55.0 | 87.7 / 89.4 / 89.7 | 48.6 / 54.5 / 54.3 | 80.3 / 84.7 / 85.8 |
| 0005, boot 2 | 53.2 / 55.0 / 54.8 | 87.7 / 90.8 / 91.1 | 49.2 / 54.9 / 55.2 | 81.1 / 86.0 / 85.9 |

Both builds decoded ~8% slower that afternoon than in the morning's runs of the same 0004 build (20k code cold 91.6
then, 80.5 here); no throttling was reported (SM clocks 2,184-2,190 MHz, 53 C, no clock events). Not explained yet.

Checks on the 0005 build: a code word hidden at 30 / 60 / 85% of 20k / 80k / 158k-token prompts found 3 / 3; the
short gates (arithmetic, forced tool call, tool continuation, strict JSON at T = 0 / 0.7 / 1.0, reasoning) 7 / 7.

## 2. Wide sampled requests

Sampled requests (temperature 0.7-1.0) whose candidates pass the narrow vocabulary slices (32,256 columns on ranks 2
and 3); each must answer within 60-120 s.

| build | requests in order | result |
| --- | --- | --- |
| 0004 (before) | top_k 1,000; 32,000; 40,000 | 1,000 and 32,000 answer; **40,000: illegal memory access on ranks 2 and 3**, server down (the uneven candidate gather jayleaton's review found) |
| 0005 without the warm-up (3 builds) | nucleus (top_p 0.9); nucleus + min_p; top_k 40,000; ... | nucleus rows answer; **top_k 40,000 (or 32,000 as the first wide request): every follower's Engram gate timed out at layer 14** |
| 0005 without the warm-up, speculation off | JSON nucleus; top_k 32,000; 40,000 | all answer |
| 0005 without the warm-up | JSON nucleus; top_k 4,000; 8,000; 16,000; 24,000 | all answer |
| **0005 (shipped)** | nucleus; nucleus + min_p; top_k 40,000; 32,000; 20; JSON nucleus | **6 / 6** |
| **0005 (shipped), fresh boot** | top_k 40,000 first; 32,000; 1,000 | **3 / 3** |

The runs that passed without the warm-up had a JSON nucleus request first: its masked rows fetch full-vocabulary
candidates after the window, so the large-k top-k and large-row sort kernels had already run. The shipped build runs
every candidate width class once at boot (`cand_warm`). One run of the 0004 build answered a first top_k 32,000 request
without a warm-up, so CUDA's lazy module loading is the most likely cause, not a proven one.

## 3. On the CPU

`tests/test_dsv41_four_sparks.py` (20 tests: four ranks as threads, exact numerics; the thread communicator now
refuses unequal all-gather sizes like NCCL): candidates at counts 1-640 over 256 / 128 / 128 / 128 slices == one
rank's; greedy, top-k and nucleus decoding on four ranks == one rank token for token; the trimmed draft head on four
ranks; the plan link's handshake (rank order, a stray client, a duplicate rank, another world size, a missing rank);
`--tp 4` scope; the BMQ clamp; the session tier's world; DSpark delta shards; the boot warm-up. Run without the fixes,
each review bug's test fails (unequal gather sizes, the draft head's boot refusal, no handshake). The other DeepSeek
V4.1 suites pass (on the final tree: 23 files, 327 passed), except two that also fail before 0005 (`test_serve_parses_the_kv_cache_flag`: 0002 adds
`--kv-dtype fp8`; `test_shipped_ranking`: the draft vocabulary file is not published).
