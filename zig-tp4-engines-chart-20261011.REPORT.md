status: needs-review

Chart 2: every engine DeepSeek V4.1 Flash has run on at TP=4 on the four Sparks. Commit `bdc739d4` on `zig-tp4` (one new commit, no amend, no push).

## Images
- docs/img/engines-dsv41-comparison-20261011.png (3000x2625 px at dpi 150), docs/img/engines-dsv41-comparison-20261011.svg
- 1,000 px check: `render --dpi 50 --png-only --out /tmp/engines-1000` gives a 1000x875 px image. I looked at both renders: every bar value is readable at 1,000 px, the % pills are small but legible, and the doc table carries every number.

## Value table (best of reps; % vs live Python RoCE; italic = different harness, indicative; — = no like metric)
| Cell | Python NCCL | Python RoCE (live) | Zig NCCL | Zig + pipe NCCL | Zig RoCE ‡ | Zig + pipe RoCE ‡ | vLLM † (2026-09-11) | SGLang † (2026-09-14) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32K prompt (prompt tok/s) | 4,084.5 (±0%) | 4,083.6 | 2,774.4 (−32%) | 4,825.6 (+18%) | 2,778.9 (−32%) | 4,851.5 (+19%) | *1,494.9* | *2,356.9* |
| 131K prompt (prompt tok/s) | 4,333.6 (±0%) | 4,354.1 | 2,554.5 (−41%) | 5,131.0 (+18%) | 2,548.9 (−41%) | 5,148.1 (+18%) | *1,415.6* | *1,950.3* |
| 154K-token prompt (s) | 35.58 (±0%) | 35.41 | 60.36 (+70%) | 30.05 (−15%) | 60.50 (+71%) | 29.95 (−15%) | — | — |
| decode after the 154K prompt (tok/s) | 81.0 (+2%) | 79.5 | 73.8 (−7%) | 74.9 (−6%) | 90.7 (+14%) | 99.6 (+25%) | — | — |
| mixed windows aggregate (tok/s) | 165.9 (−14%) | 192.1 | 177.2 (−8%) | 175.2 (−9%) | 219.2 (+14%) | 212.2 (+10%) | — | — |
| code ×1 (tok/s) | 101.7 (−15%) | 119.1 | 111.6 (−6%) | 111.5 (−6%) | not measured | not measured | *71.4* | *76.8* |
| code ×4 aggregate (tok/s) | 176.1 (−15%) | 207.9 | 181.9 (−13%) | 182.1 (−12%) | not measured | not measured | — | — |
| code T0.7 ×4 aggregate (tok/s) | 169.2 (−15%) | 199.4 | 182.8 (−8%) | 181.9 (−9%) | not measured | not measured | — | — |
| prose ×1 (tok/s) | 55.9 (−15%) | 66.1 | 57.5 (−13%) | 57.6 (−13%) | not measured | not measured | *31.7* | *36.6* |
| prose ×4 aggregate (tok/s) | 112.9 (−13%) | 129.8 | 120.9 (−7%) | 121.0 (−7%) | not measured | not measured | — | — |

## Series × cell → value → source → harness (paths relative to /home/jun/nest/tmp/zvp-20261011/)
| series | cell | best | median | n | source | harness |
| --- | --- | ---: | ---: | ---: | --- | --- |
| python-nccl | prefill-32k | 4084.5 | 4051.4 | 3 | python-tp4-nccl/long.json | zig_vs_python |
| python-nccl | prefill-131k | 4333.6 | 4332.1 | 3 | python-tp4-nccl/long.json | zig_vs_python |
| python-nccl | ttft-154k | 35.581 | 35.594 | 3 | python-tp4-nccl/long.json | zig_vs_python |
| python-nccl | long-decode-131k | 81.01 | 71.67 | 3 | python-tp4-nccl/long.json | zig_vs_python |
| python-nccl | mixwin | 165.92 | 161.21 | 3 | python-tp4-nccl/mixwin.json | zig_vs_python |
| python-nccl | code-x1 | 101.71 | 101.43 | 3 | python-tp4-nccl/warm.json | zig_vs_python |
| python-nccl | code-x4 | 176.09 | 174.96 | 3 | python-tp4-nccl/warm.json | zig_vs_python |
| python-nccl | code-t07-x4 | 169.15 | 168.08 | 3 | python-tp4-nccl/warm.json | zig_vs_python |
| python-nccl | prose-x1 | 55.85 | 55.83 | 3 | python-tp4-nccl/warm.json | zig_vs_python |
| python-nccl | prose-x4 | 112.93 | 111.69 | 3 | python-tp4-nccl/warm.json | zig_vs_python |
| python-roce | prefill-32k | 4083.6 | 4065.7 | 3 | python-tp4-roce-live/long.json | zig_vs_python |
| python-roce | prefill-131k | 4354.1 | 4352.2 | 3 | python-tp4-roce-live/long.json | zig_vs_python |
| python-roce | ttft-154k | 35.414 | 35.429 | 3 | python-tp4-roce-live/long.json | zig_vs_python |
| python-roce | long-decode-131k | 79.48 | 79.22 | 3 | python-tp4-roce-live/long.json | zig_vs_python |
| python-roce | mixwin | 192.08 | 191.47 | 3 | python-tp4-roce-live/mixwin.json | zig_vs_python |
| python-roce | code-x1 | 119.08 | 117.92 | 3 | python-tp4-roce-live/warm.json | zig_vs_python |
| python-roce | code-x4 | 207.93 | 205.75 | 3 | python-tp4-roce-live/warm.json | zig_vs_python |
| python-roce | code-t07-x4 | 199.43 | 198.75 | 3 | python-tp4-roce-live/warm.json | zig_vs_python |
| python-roce | prose-x1 | 66.07 | 66.05 | 3 | python-tp4-roce-live/warm.json | zig_vs_python |
| python-roce | prose-x4 | 129.78 | 129.24 | 3 | python-tp4-roce-live/warm.json | zig_vs_python |
| zig-nccl-control | prefill-32k | 2774.4 | 2744.5 | 3 | w7/zig-tp4-nccl-control/long.json | zig_vs_python |
| zig-nccl-control | prefill-131k | 2554.5 | 2542.7 | 3 | w7/zig-tp4-nccl-control/long.json | zig_vs_python |
| zig-nccl-control | ttft-154k | 60.363 | 60.642 | 3 | w7/zig-tp4-nccl-control/long.json | zig_vs_python |
| zig-nccl-control | long-decode-131k | 73.77 | 71.18 | 3 | w7/zig-tp4-nccl-control/long.json | zig_vs_python |
| zig-nccl-control | mixwin | 177.25 | 176.52 | 3 | w7/zig-tp4-nccl-control/mixwin.json | zig_vs_python |
| zig-nccl-control | code-x1 | 111.61 | 111.45 | 3 | w7/zig-tp4-nccl-control/warm.json | zig_vs_python |
| zig-nccl-control | code-x4 | 181.91 | 181.56 | 3 | w7/zig-tp4-nccl-control/warm.json | zig_vs_python |
| zig-nccl-control | code-t07-x4 | 182.75 | 182.18 | 3 | w7/zig-tp4-nccl-control/warm.json | zig_vs_python |
| zig-nccl-control | prose-x1 | 57.52 | 57.18 | 3 | w7/zig-tp4-nccl-control/warm.json | zig_vs_python |
| zig-nccl-control | prose-x4 | 120.92 | 120.63 | 3 | w7/zig-tp4-nccl-control/warm.json | zig_vs_python |
| zig-nccl-pipe | prefill-32k | 4825.6 | 4823.1 | 3 | w7/zig-tp4-nccl-pipe/long.json | zig_vs_python |
| zig-nccl-pipe | prefill-131k | 5131.0 | 5123.0 | 3 | w7/zig-tp4-nccl-pipe/long.json | zig_vs_python |
| zig-nccl-pipe | ttft-154k | 30.052 | 30.099 | 3 | w7/zig-tp4-nccl-pipe/long.json | zig_vs_python |
| zig-nccl-pipe | long-decode-131k | 74.92 | 73.76 | 3 | w7/zig-tp4-nccl-pipe/long.json | zig_vs_python |
| zig-nccl-pipe | mixwin | 175.15 | 173.38 | 3 | w7/zig-tp4-nccl-pipe/mixwin.json | zig_vs_python |
| zig-nccl-pipe | code-x1 | 111.52 | 111.45 | 3 | w7/zig-tp4-nccl-pipe/warm.json | zig_vs_python |
| zig-nccl-pipe | code-x4 | 182.13 | 181.51 | 3 | w7/zig-tp4-nccl-pipe/warm.json | zig_vs_python |
| zig-nccl-pipe | code-t07-x4 | 181.86 | 180.38 | 3 | w7/zig-tp4-nccl-pipe/warm.json | zig_vs_python |
| zig-nccl-pipe | prose-x1 | 57.62 | 57.36 | 3 | w7/zig-tp4-nccl-pipe/warm.json | zig_vs_python |
| zig-nccl-pipe | prose-x4 | 120.98 | 120.44 | 3 | w7/zig-tp4-nccl-pipe/warm.json | zig_vs_python |
| zig-roce-control | prefill-32k | 2778.9 | 2758.7 | 3 | w8/zig-tp4-roce-control/long.json | zig_vs_python |
| zig-roce-control | prefill-131k | 2548.9 | 2546.5 | 3 | w8/zig-tp4-roce-control/long.json | zig_vs_python |
| zig-roce-control | ttft-154k | 60.496 | 60.551 | 3 | w8/zig-tp4-roce-control/long.json | zig_vs_python |
| zig-roce-control | long-decode-131k | 90.65 | 84.72 | 3 | w8/zig-tp4-roce-control/long.json | zig_vs_python |
| zig-roce-control | mixwin | 219.2 | 215.24 | 3 | w8/zig-tp4-roce-control/mixwin.json | zig_vs_python |
| zig-roce-pipe | prefill-32k | 4851.5 | 4842.2 | 3 | w8/zig-tp4-roce-pipe/long.json | zig_vs_python |
| zig-roce-pipe | prefill-131k | 5148.1 | 5133.5 | 3 | w8/zig-tp4-roce-pipe/long.json | zig_vs_python |
| zig-roce-pipe | ttft-154k | 29.952 | 30.037 | 3 | w8/zig-tp4-roce-pipe/long.json | zig_vs_python |
| zig-roce-pipe | long-decode-131k | 99.58 | 93.06 | 3 | w8/zig-tp4-roce-pipe/long.json | zig_vs_python |
| zig-roce-pipe | mixwin | 212.24 | 208.92 | 3 | w8/zig-tp4-roce-pipe/mixwin.json | zig_vs_python |
| vllm | code-x1 | 71.44 | 71.44 | 1 | vllm/dsv41-vllm/phase4/final-champion/bench-final-champion.json | v41bench |
| vllm | prose-x1 | 31.7 | 31.7 | 1 | vllm/dsv41-vllm/phase4/final-champion/bench-final-champion.json | v41bench |
| vllm | prefill-32k | 1494.9 | 1476.95 | 2 | vllm/dsv41-vllm/phase4/final-champion/bench-final-champion.json, vllm/dsv41-vllm/phase4/final-champion/prefill-probe.json | v41bench cold prefill / prefillprobe |
| vllm | prefill-131k | 1415.6 | 1400.05 | 2 | vllm/dsv41-vllm/phase4/final-champion/bench-final-champion.json, vllm/dsv41-vllm/phase4/final-champion/prefill-probe.json | v41bench cold prefill / prefillprobe |
| sglang | code-x1 | 76.8 | 76.47 | 3 | sglang/trial-20260914/bench/rep1/bench-sglang-r1.json, sglang/trial-20260914/bench/rep2/bench-sglang-r2.json, sglang/trial-20260914/bench/rep3/bench-sglang-r3.json | v41bench |
| sglang | prose-x1 | 36.58 | 36.35 | 3 | sglang/trial-20260914/bench/rep1/bench-sglang-r1.json, sglang/trial-20260914/bench/rep2/bench-sglang-r2.json, sglang/trial-20260914/bench/rep3/bench-sglang-r3.json | v41bench |
| sglang | prefill-32k | 2356.9 | 2283.75 | 4 | sglang/trial-20260914/bench/rep1/bench-sglang-r1.json, sglang/trial-20260914/bench/rep2/bench-sglang-r2.json, sglang/trial-20260914/bench/rep3/bench-sglang-r3.json, sglang/trial-20260914/bench/sglang-prefill/prefill-probe.json | v41bench cold prefill / prefillprobe |
| sglang | prefill-131k | 1950.3 | 1950.3 | 1 | sglang/trial-20260914/bench/sglang-prefill/prefill-probe.json | v41bench cold prefill / prefillprobe |

Reproduces Depths' table where w8 measured the cell: prefill 32K 4,084 → 4,852, 131K 4,354 → 5,148, TTFT 35.4 → 30.0 s, decode after 131K 79.5 → 99.6, mixwin 192 → 212. **It does not reproduce the code/prose cells of that table.** Depths' "Zig + pipeline (RoCE)" 129 / 218 / 227 / 138 / 66.2 are w8's **cold pass** (`w8/zig-tp4-roce-pipe/cold.json`, n = 1, the first requests after boot), set against Python's **warm** best of 3. w8 has no warm.json. In the chart those cells are "not measured". The cold numbers are kept in the data file (`cold_pass_only`) and in the doc text, labelled as cold. Cold vs cold for scale: Python RoCE 113.0 / 204.9 / 199.7 / 128.2 / 65.2 vs Zig + pipe RoCE 129.1 / 218.3 / 226.9 / 138.2 / 66.2 (code ×1 / ×4 / T0.7 ×4, prose ×4 / ×1).

## Choices
- Percentages: vs live Python TensorFold over RoCE, the baseline. They appear only on the four Zig series, as same-harness cross-session numbers. vLLM and SGLang get no % (different harness).
- Tiers: controlled = an ink box and tab around each Zig off→on pair (w7 NCCL, w8 RoCE). Cross-session = a grey dashed pill. Different harness = hollow outlined bar, sparse hatch, italic value, "other harness" in the legend.
- vLLM: the champion in `vllm/dsv41-vllm/CHAMPION.env` (greedy DSpark draft k = 5, MAX_BATCHED = 16384, MAXLEN = 430080). STATUS.md "Final (Phase 4b)" says it is the config left after B3 (greedy) and B7 (batched 16K) were accepted and every other arm (K = 4, K = 10, fusion, Engram 64 threads, NCCL variants) was rejected or skipped. The run is `phase4/final-champion`, the named champion run and the only one with both decode and prefill. n = 1 decode, n = 2 prefill.
- SGLang: `trial-20260914`, the gated deployment: 3 bench reps plus 1 prefill probe.
- Colours: python / zig / zig-pipe keep chart 1's slots, and vLLM takes violet (slot 7). SGLang is neutral ink-grey on purpose, because no sixth hue clears the all-pairs floor next to orange (magenta 12.9, red 7.1). Validator results below.

## Excluded, and why
- docs/campaign/BASELINE-RESULTS.md: vLLM at TP=2 (two Sparks). Left out.
- TTFT for vLLM and SGLang: their longest cold prompt is 145.5–145.9K tokens, not 154K (vLLM 102.8 s, SGLang 74.8 s). Footnoted, not plotted.
- ×4 cells for vLLM and SGLang: their C4 sends **one prompt ×4** (prompt_tokens [47,47,47,47]), and agg_tok_s = tokens / wall clock including TTFT. Ours uses 4 distinct prompts over the span where all four streams are live. Footnoted.
- T0.7, decode after a long prompt, mixed windows: their harness has no such cell.
- sglang/tf-claude-20261004 (2026-10-04 SGLang TP4/EP2, uncensored FP8, FOUR_SPARKS.md's comparison): a different deployment and prompt set, and its prompts are only 17–1,460 tokens, so it has no long-prompt cell.
- vllm ab-unc-20260913 (uncensored FP8 on vLLM): failed gates G1 and G3. Not a series.
- w8 warm / mix: not measured. Shown as "not measured", never 0.

## Correctness gates found
SGLang 2026-09-14 (`sglang/trial-20260914/gates/`, same day as the bench reps; file times 03:48–04:32, bench 03:56): **all pass.** It is not a no-go.
- g1-garble.log: `30/30 clean (0 garbled)`
- g2-tool.log: `TOOL_ROUNDTRIP 3/3`
- g3-reasoning.log: `REASONING_PARSER PASS`
- g4-needle-32k/400k/1m.log: `"answer": "COPPER-LANTERN-8315", "pass": true`
- WORLD.md: "The Sep-10 DSpark-on-GB10 corruption is fixed by the 2026-09-11 `lmsysorg/sglang:dev-dsv41` image … G0 (our deterministic tool-result continuation repro) passes 3/3 clean."

So the corruption Jun remembers is real, but it belongs to the **2026-09-10 image**, not the benched one. vLLM's phase-3 STATUS also says "HEADLINE: DSpark corruption was SGLang-side, not model-bound." The uncensored rerun (gates-unc/trial.log) also reads `VERDICT: ALL GATES PASS`.

vLLM (`vllm/dsv41-vllm/STATUS.md`): "Gates on final world: ALL PASS incl NIAH 32k x3 + 400k (397,753 tok, TTFT 323 s)". The phase-3 entry reads "Tool round-trip (phase-1 SGLang corruption repro): PASS CLEAN". The gate log `phase4/gates-final-champion.log` is **not in the mirror**, so this rests on STATUS.md.

## Layout check (full size)
layout check: 188 text artists, none outside the 1440x1260 pt figure, no pair overlapping or under 3.0 pt apart
  closest to the edge: '● Controlled: Zig pipelined prefill off → on, sa' at 10.1 pt; next 'DeepSeek V4.1 Flash on 4× DGX Spark: every engin' at 14.4 pt
  tightest pair: '120' / 'Time to first token', 3.5 pt apart
(This chart raises chart 1's MIN_GAP_PT from 1.5 to 3.0. At 1.5 pt, "4,084 4,084" read as one number. Chart 1 is unchanged and still passes.)

## Checks
check: python3 bench/charts/engines_comparison.py build; cmp before/after → BUILD-IDEMPOTENT (byte-identical data file)
check: uv run --with matplotlib python bench/charts/engines_comparison.py render → layout check passed (188 artists, ≥3.0 pt), wrote png + svg
check: render --dpi 50 --png-only --out /tmp/engines-1000 → layout check passed; PNG header 1000x875
check: PNG header of docs/img/engines-dsv41-comparison-20261011.png → 3000x2625
check: mutation (MIN_GAP_PT=12 copy of the script) → "layout check FAILED … '4,084' / '4,084'", exit=1 (it fails loudly)
check: uv run --with matplotlib python bench/charts/prefill_pipe_comparison.py render --png-only --out /tmp/chart1-check → chart 1 layout check still passes (110 artists)
check: uvx ruff check --line-length 120 bench/charts/engines_comparison.py → All checks passed!
check: node validate_palette.js "#eb6834,#1baf7a,#2a78d6,#4a3aa7,#52514e" --mode light --pairs all → CVD worst 9.2 PASS, normal-vision worst 16.3 PASS, chroma FAIL only on #52514e (the grey is intentional)
check: python3 bench/charts/engines_comparison.py table → the table above; w8 short cells "not measured", vLLM/SGLang non-matching cells "—"
check: grep cold lines in w8/zig-tp4-roce-pipe/bench.log → "cold code T0.0 x1: 129.13 … x4: 218.33 … T0.7 x4: 226.87 … prose T0.0 x1: 66.24 … x4: 138.17" (the source of Depths' code/prose numbers)
check: cat sglang/trial-20260914/gates/*.log → G1 30/30 clean, G2 3/3, G3 PASS, G4 3× pass
check: grep "Gates on final world" vllm/dsv41-vllm/STATUS.md → "ALL PASS incl NIAH 32k x3 + 400k"
check: python3 max prompt_tokens over sglang/tf-claude-20261004/{bench,ab}/*.json → 17–1,460 (n=129)
check: git log --oneline -1 → bdc739d4 docs: chart 2 — every engine on four Sparks …; git status → only the untracked REPORT files

## Beyond-scope findings
- **The pipelined prefill changes greedy (T0) outputs.** w7/compare-control-vs-pipe.w7.md: `T0 replies equal (zig-tp4-nccl-pipe vs zig-tp4-nccl-control): 40/46`. w8: `10/16`. Same-engine control-vs-control runs (compare-w4b-vs-w6-control.md, compare-zig-nccl-vs-roce.w4b.md) are `40/40`. I did not check which cells the mismatches fall in. If the pipeline is meant to be numerically identical (ZIG-PREFILL-PIPE claims call-for-call identity), this needs a look before the w8 review closes. Worth pursuing: yes.
- **Depths' table mixes cold n=1 with warm best-of-3** for the RoCE code/prose cells (details above). Any message to Jun that quotes "code x1 119 → 129" should be corrected, or labelled cold vs warm. Worth pursuing: yes.
- **SGLang's 1M needle gate reports a nonsense prefill rate**: `{"prompt_tokens": 993435, "ttft_s": 2.7, "prefill_tok_s": 365698.1, … "pass": true}` (gates/g4-needle-1m.log). That is almost certainly a radix-cache hit on the 400K needle's filler. The answer is still correct, but the number must never be quoted as a prefill rate. Worth pursuing: no (note only).
- **SGLang is not deterministic at T0** across reps of the same prompt (tf-claude-20261004/ab/sglang-0-A.json, story rep 0 vs rep 1: different text). ARCH-LEVERAGE.md already notes SGLang "output is not bitwise stable across batch composition". Worth pursuing: no.
- The boot logs do not record the TensorFold weights pack by name (only "55.1 GiB" a rank). "EXL3 2.9 bpw" on the labels comes from config/tp4-zig.env.example / ZIG-TP4-PLAN.md (like-for-like with Python's pack), not from the run's own log. Worth pursuing: maybe (have the harness record MODEL_DIR).
