# Faster prompt reading on four Sparks: raw files (2026-10-05)

Summary and method: [docs/PREFILL_SPEED.md](../../docs/PREFILL_SPEED.md). Every `*-progress.log` row is one cold
request of the published depth fixtures (`trial 0`, 512 tokens, thinking off) against a build started with an empty
session cache; `output_sha256` compares replies across builds.

| file | build |
| --- | --- |
| `tp-control-w2-progress.log`, `tp-control-w4-progress.log` | TP=4 as 0003 (two windows) |
| `tp-overlap-progress.log`, `tp-chunk4096-progress.log`, `tp-overlap-chunk4096-progress.log` | overlap, 4,096-row segments, both |
| `tp-split-progress.log`, `tp-split-overlap-progress.log` | split selections, split + overlap |
| `pipe-p2p-progress.log` | the pipeline (p2p exchange): the served configuration |
| `pipe-allgather-traced-progress.log`, `pipe-chunk4096-traced-progress.log` | the pipeline with an all-gather exchange / 4,096-row segments, round timing on |
| `pipe-round-timing.txt` | per rank: Engram, stage and exchange ms a round (all-gather and 4,096-row runs) |
| `pipe-p2p-needle.jsonl`, `tp-split-overlap-needle.jsonl` | a code word hidden at 30 / 60 / 85% of 20k / 80k / 158k tokens (the TP run shared the server with other traffic: its wall times are not comparable) |
| `pipe-p2p-gates.log` | the short gates on the pipelined server |
| `ppbench-full.txt`, `pipe-vs-tp-drift.txt` | the prototype (`ppbench.py`) over 160k tokens; the stage outputs vs the TP forward layer by layer, and the TP forward's fast vs exact kernels |
