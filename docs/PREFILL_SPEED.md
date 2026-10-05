# Faster prompt reading on four Sparks

`patches/0004-prefill-speed.patch` (after 0003) adds three opt-in ways to read a prompt faster on four DGX Sparks.
Decode is untouched by all three. Measured 2026-10-05 on the four-Spark setup of [FOUR_SPARKS.md](FOUR_SPARKS.md)
(same pack, `config/prod.env.example` knobs incl. expert pruning, thinking off, 512-token replies, the published depth
fixtures, each build with an empty session cache: every prompt below is read cold).

| cold time to first token | 20k prompt | 160k prompt |
| --- | ---: | ---: |
| 0003 as published (TP=4), 2026-10-04 | 12.9-14.2 s | 97.4-99.8 s |
| same build, 2026-10-05 control | 10.5-11.1 s | 82.5-84.8 s |
| `TF_DSV41_PREFILL_OVERLAP=1024` | 9.7 s | 81.6-82.4 s |
| `TF_DSV41_INDEX_SPLIT=256` | 9.8-10.8 s | 74.6-75.9 s |
| split + overlap | 9.4-9.5 s | 74.6-75.2 s |
| **`TF_DSV41_PREFILL_PIPE=1024`** (+ split + overlap for short prompts) | **5.8-6.8 s** | **38.8-39.6 s** |

Decode rates were unchanged in every row (prose ~54, code ~95 tok/s). Raw rows, the pipeline's round timing and the
checks below: [results/prefill-speed-20261005](../results/prefill-speed-20261005/).

## 1. Pipelined prompt reading (`TF_DSV41_PREFILL_PIPE`)

Tensor parallelism splits every layer four ways, but in this model a lot of each layer is not split: the indexer's 32
heads and its top-512 selection, the compressor and the single KV head's projections, the mHC streams. Every rank
computes those in full, and every sublayer ends in an all-gather. For decode that is the right trade; for reading a
long prompt it wastes three quarters of the replicated work.

With the switch on, each rank also loads a quarter of CED's encoder layers at full width (stages
`[0, 5) [5, 10) [10, 15) [15, 21)`; the last stage ends at layer 20's site and compressor, as the encoder pass does).
A long prompt's segments then flow through the ranks like an assembly line: in round t, rank s runs segment t - s
through its layers. Each round ends in one exchange:

- the next stage's input, point to point (NCCL p2p): the streams before the stage's last FFN post, that pending post
  (the gathered partial and its mHC coefficients; the next stage's first boundary applies it exactly as the next block
  would in one process) and the latest index source's selection;
- to every rank: the rows the round wrote into the kv sources' compressed rows and index keys, and the last stage's
  CED stash rows.

Each stage writes its layers' rings, compressed rows, index keys and carries straight into the slot's own stores.
`finish_prompt` and prompt snapshots drain the line and spread each stage's last-window SWA rows and carries, so the
slot's state is whole before anything reads it. Decode runs tensor parallel as before.

- Cost: ~27 GB more GPU memory a rank (the encoder half of the model a second time; `MemAvailable` ~50 -> ~26 GB on our
  nodes), one long prompt in the line at a time (as `long prefills 1 at a time` already was).
- Round timing at 160k (2,048 rows a segment): stages 366 / 383 / 401 / 341 ms, Engram rows ~33 ms, the exchange
  ~20-45 ms. Layer 14 (kv source + selection + Engram) makes the third stage the slowest; with whole-layer stages the
  split above is already the most even.
- Numerics: a stage's layers run at world 1, so a pipelined prompt's bits are one rank's, not the TP=4 ones (its own
  session tag, `tag_grid + 2`). How far apart, layer by layer on a prompt's first segment: 0.4% at layer 0 to 26% at
  layer 19 (relative norm of the streams), the same curve as the TP forward's own fast vs exact prefill kernels (0.5%
  -> 27%); the top-512 sparse selection turns rounding-level differences into different selections. In exact
  numerics the pipelined state equals the tensor-parallel one (`tests/test_dsv41_pipe.py`).
- Checks on the running server: a code word hidden at 30% / 60% / 85% of 20k / 80k / 158k-token prompts: 3/3 (as the
  TP server); the short gates 7/7; the p2p exchange and an all-gather exchange give byte-identical replies.

Knobs: `TF_DSV41_PREFILL_PIPE=<rows>` (segments of at least that many rows start a pipeline; 0 = off),
`TF_DSV41_PREFILL_PIPE_BOUNDS` (explicit stages, "0,5,10,15,21"), `TF_DSV41_PIPE_P2P=0` (all-gather instead of p2p),
`TF_DSV41_PIPE_TRACE=1` (a round's Engram / stage / exchange times, printed when a prompt drains).

## 2. Split selection (`TF_DSV41_INDEX_SPLIT`)

Without the pipeline, an index layer's selection of a prefill segment of at least that many rows is computed for a
quarter of the rows on each rank and gathered (a ~4 MB exchange). A row's selection depends on that row alone: same
selections, same bits. -10% at 160k.

## 3. Overlapped exchanges (`TF_DSV41_PREFILL_OVERLAP`)

Without the pipeline, a prefill segment runs as two interleaved halves; each half's all-gathers run on a high-priority
stream while the other half computes. Rows are invariant: same bits. -10% at 20k, -2% at 160k (at depth the segment is
dominated by attention over the history, not by the exchanges).

## Tried and dropped

- Larger segments (4,096 rows) for TP=4 prefill: slower (85.7 vs 83.8 s at 160k). In the pipeline they tie at 160k
  (38.6-38.9 s) and lose at 20k (the line fills more slowly).
- Two rewrites of the streaming top-512 selection (row-blocked scoring, a two-pass threshold): both pick the same
  positions, both are slower than the original kernel.

## Tests

`tests/test_dsv41_overlap.py` (overlap and split == one segment: TP4 prefill, TP2 CED replay, alone and together),
`tests/test_dsv41_pipe.py` (W = 2 and 4, a snapshot mid-prompt, the p2p path), on the CPU twin.
