status: needs-review

Commit 92fdb5e1 on zig-tp4, parent 70204bae. It is not pushed: Depths pushes. This report replaces the earlier "blocked" report, which was written before the mirror existed.

Images: docs/img/prefill-pipe-comparison-20261011.png (2080x2112) and docs/img/prefill-pipe-comparison-20261011.svg
Script: bench/charts/prefill_pipe_comparison.py (`render`; `add --run-dir ...` appends or replaces a series)
Data: bench/charts/data/prefill-pipe-20261011.json (each value has its source file, median and n)
Doc: docs/ZIG-RESULTS.md, new section "TP=4: Python vs Zig vs Zig + pipelined prefill (2026-10-11)"

## Data (best of 3, no failed streams; TTFT is the lowest of 3) → source

Artifact root: forge:~/zigtp4-run/out/zvp, mirrored at /home/jun/nest/tmp/zvp-20261011.

| cell | Python NCCL | Python RoCE | Zig off (w7, NCCL) | Zig + pipe (w7, NCCL) | file in each run dir |
|---|---:|---:|---:|---:|---|
| prefill 32K tok/s | 4084.5 | 4083.6 | 2774.4 | 4825.6 | long.json (long-32768 prompt_tok_s) |
| prefill 131K tok/s | 4333.6 | 4354.1 | 2554.5 | 5131.0 | long.json (long-131072 prompt_tok_s) |
| TTFT 154K-token prompt, s | 35.581 | 35.414 | 60.363 | 30.052 | long.json (long-131072 ttft_s, min) |
| code T0 x1 | 101.71 | 119.08 | 111.61 | 111.52 | warm.json |
| code T0 x4 aggregate | 176.09 | 207.93 | 181.91 | 182.13 | warm.json |
| prose T0 x4 aggregate | 112.93 | 129.78 | 120.92 | 120.98 | warm.json |
| decode after 32K | 80.85 | 89.22 | 74.85 | 85.34 | long.json (mean_tok_s) |
| mix: 131K decode | 67.64 | 94.57 | 80.93 | 79.57 | mix.json |
| mix: code x4 per stream | 64.25 | 74.88 | 64.54 | 63.30 | mix.json |
| mixwin aggregate | 165.92 | 192.08 | 177.25 | 175.15 | mixwin.json |

Run directories: python-tp4-nccl/ (salt zvp-20261010b), python-tp4-roce-live/ (salt zvp-20261010a), w7/zig-tp4-nccl-control/ and w7/zig-tp4-nccl-pipe/ (both salt zvp-20261011w7).

The Python pipelined-prefill gate figures named in the brief (4,084 tok/s @32K, 4,354 @131K, TTFT 35.5 s @154K) are not a separate artifact. They are these Python runs' own long cells: 4084.5 is python-tp4-nccl's 32K best, and 4354.1 and 35.414 come from python-tp4-roce-live. The values were found and are sourced, so nothing is labelled "source not mirrored".

For "Zig, before today" I used window 7's control boot (same window, same image, pipeline off), not the w4b run. It is the only one-variable-changed comparison. compare-w4b-vs-w6-control.md shows w4b and the control within about 1 % on prefill.

check: python3 bench/charts/prefill_pipe_comparison.py add (x4 run dirs) → 10 cells each; per-series values printed and match the table above
check: compare against /home/jun/nest/tmp/zvp-20261011/w7/compare-control-vs-pipe.w7.md → long-32768 2774.4/4825.6 (+73.9 %), long-131072 2554.5/5131.0 (+100.9 %), decode 74.85/85.34, mix 64.54/63.3 and 80.93/79.57, mixwin 177.25/175.15, code/prose best values: all equal
check: compare against /home/jun/nest/tmp/zvp-20261011/compare-nccl.w4b.md (python-tp4-nccl column) → 4084.5, 4333.6, 80.85, 64.25, 67.64, 165.92, code x1 101.7, code x4 176.1, prose x4 112.9: all equal
check: grep "long ... pass" */bench.log → python-roce 131K pass 2 4354.1 tok/s TTFT 35.414 s; w7 pipe 32K pass 0 4825.6 TTFT 7.872 (the brief's spot-check); control 2726.2 TTFT 13.934 (pass 0)
check: python3 (failed_streams sum over warm/long/mix/mixwin for all four run dirs) → 0 everywhere; salts zvp-20261010b / zvp-20261010a / zvp-20261011w7 / zvp-20261011w7
check: grep "tp: rank 0 of 4 up" zig-tp4-nccl-{pipe,control}-boot-r0.w7.log → both "(nccl, NCCL 23203 ...)"; the pipe log has "pipeline: prompts of 1024+ rows through stages { 0, 5, 10, 15, 21 }"
check: python3 (percentages from the data file) → on vs off +73.9 / +100.9 / −50.2 %; Zig+pipe vs Python NCCL +18.1 / +18.4 / −15.5 %; Python RoCE vs NCCL decode +10.4…+39.8 %. These match the doc table and text (the "10–40 %" wording was corrected from my first draft)
check: node dataviz validate_palette.js "#eb6834,#1baf7a,#2a78d6" --mode light --pairs all → ALL CHECKS PASS (worst CVD ΔE 9.2, normal-vision 24.0). Aqua contrast is 2.74:1, a WARN that the relief rule covers: every bar has a value label and the doc has the table
check: uv run --with matplotlib python bench/charts/prefill_pipe_comparison.py render → writes PNG+SVG. I opened the PNG after each of 4 iterations; the final one has no overlaps (brackets clear of the % labels, footnotes clear of the decode axis), labels are legible and engines are told apart by colour (orange/aqua/blue) with RoCE also hatched
check: md5sum before/after re-render → ff0a40b8… both times (deterministic output)
check: copy of data file + add --run-dir w8/zig-tp4-roce-pipe → "5 cells", renders a 5-series chart with no layout breakage; I viewed the PNG (/tmp/w8test.png); the committed data file was not touched
check: uvx ruff check bench/charts/ → All checks passed!
check: grep -niE "price|cost|\$" in the new doc section and bench/charts → no matches (no pricing)
check: git show --stat HEAD → 5 files (script, data, PNG, SVG, ZIG-RESULTS.md); no engine code; no amend/rebase/push

Design notes: there are three panels in one figure, so prefill (~5,000) and decode (~100–200) never share an axis, and TTFT is its own panel labelled "lower is better ↓". Colour shows the engine and hatching shows RoCE; every legend entry names its transport and window. The bracket "● same window" marks the only controlled comparison. The small % on each Zig bar compares against Python over the same transport and is stated as cross-session in the note line and the footnote.

## Beyond-scope findings (noticed, not requested)
- The mirror already holds a partial window 8 run, w8/zig-tp4-roce-pipe/ (long.json and mixwin.json, no warm/mix yet). Its prefill is 4851.5 / 5148.1 tok/s, TTFT 29.95 s, mixwin 212.1. I did not chart it because the brief says to wait for window 8. A dry run shows `add` handles it. worth-pursuing: yes (once w8 completes, append it with `add` and regenerate)
- Python decode is 10–40 % faster over RoCE than over NCCL (e.g. code x4 207.9 vs 176.1). Against Python over RoCE, the Zig NCCL decode cells are behind (code x4 182.1 vs 207.9). The chart shows this, but any headline "Zig beats Python on decode" holds only against Python over NCCL. worth-pursuing: yes (window 8's Zig RoCE decode is the fair comparison)
- In bench/zig_vs_python.py the mix-code cell records aggregate_tok_s = None, so its "best" is the mean per-stream rate, not an aggregate. compare-*.md presents it next to aggregates without saying so. I labelled it "per stream". worth-pursuing: maybe
- The decode figure after a long prompt varies a lot between passes (Python NCCL 61.4–80.9, Zig pipe 68.0–85.3), so the +14 % on/off figure for that cell is probably noise. worth-pursuing: maybe (more passes)

## Round 2 (presentation only): 2cb723c8 on top of 92fdb5e1, not pushed

status: needs-review

No value, series, source path or footnote changed. bench/charts/data/prefill-pipe-20261011.json is byte-identical (md5 ab41af6e…).

Mechanical layout check, now built into `render`. It does a full canvas.draw(), takes every visible text artist's window extent plus its bbox patch, and fails (keeping <out>.FAILED.png) if any of them leaves the figure, or if two of them overlap or are closer than 1.5 pt.
- Before, on 92fdb5e1's layout (112 text artists, 936x950 pt): nothing leaves the figure. The closest to an edge is the footnote block, 11.4 pt from the bottom. Among the decode labels the vision model named, "+6%" is 27.1 pt and "175" is 28.2 pt from the right edge; the title is 22.5 pt from the top. So the reported clipping was a misread. One real overlap: "−50% · same window ●" with the TTFT panel's "seconds (lower is better ↓)". (My first measurement, taken without a draw, showed bogus legend overlaps from pre-layout positions, which is why the check draws first.)
- After (110 text artists, 936x1066 pt): nothing leaves the figure and no pair overlaps or is under 1.5 pt apart. Closest to an edge: the TTFT tab "● −50% pipeline off → on", 7.9 pt from the right. Tightest pair: decode "+20%" / "+18%", 1.9 pt apart.

Visual changes:
- Controlled comparison: the Zig off/on NCCL pair is boxed in solid ink, with an inverted ink tab "● +74% pipeline off → on / same window, one variable" (and the same for +101 % and −50 %). Cross-session %: grey italic text in a dashed outline pill on each Zig bar. On the prefill and TTFT panels, where a pill sits inside the controlled box, it reads "+18% / vs Python". A key row shows a sample of each mark next to its sentence from the data file's percent_note. The two classes differ in fill (inverted vs outline), line (solid vs dashed), weight (bold vs italic), colour (ink vs grey) and placement (one tab per pair above the box vs one pill per bar). No bar has a delta that could be read as the controlled one. Python bars carry no %. In decode there is no box (the on/off deltas there were never charted), so every % there is a cross-session pill.
- Legend: "Python TensorFold · NCCL (2026-10-10)", "Python TensorFold · RoCE (2026-10-10)", "Zig, pipelined prefill off · NCCL (window 7, 2026-10-11)", "Zig + pipelined prefill · NCCL (window 7, 2026-10-11)". These labels come from the data file. I did not rename the control series to "Zig · NCCL" because that would change the data.
- Sizes: footnotes 9.5 → 12 pt, legend 10.5 → 12.5, unit labels 10.5 → 12, ticks 12 → 13, decode % 8.5–9.5 → 10. The figure is now 13 x 14.8 in. At 1,000 px wide the footnotes are about 12.8 px. I viewed it: footnotes, legend and key read clearly; the decode pills are the smallest text and still readable.
- Labels and the box are stacked in points (the y limit is solved from them), so they cannot collide whatever the scale. With 5+ series, neighbouring decode pills are staggered and the panels move down for a third legend row.
- Palette: the claim holds, now with its metric stated. ΔE 9.2 is the dataviz validator's OKLab ΔE×100 under Machado 2009 severity 1.0 (deutan; tritan 9.6). An independent check (colorspacious full-severity CVD, CIEDE2000) gives worst pairs orange↔aqua 16.7 deutan / 14.7 protan and aqua↔blue 14.6 tritan. Normal vision is ≥ 44.8. Aqua's 2.74:1 contrast on the surface is still a WARN, covered by the value labels.

Images: docs/img/prefill-pipe-comparison-20261011.png (2080x2368, md5 e825bcc4…) and .svg. Phone check (not committed): /tmp/chart2/phone-1000px.png (1000x1138). Before: /tmp/chart2/before.png. Window 8 dry run: /tmp/chart2/w8test.png.

check: measure.py (draw, then text extents) on 92fdb5e1's script → 0 outside the figure; min edge margin 11.4 pt (footnotes); 1 overlapping pair ("−50% · same window ●" / "seconds (lower is better ↓)")
check: uv run --with matplotlib python bench/charts/prefill_pipe_comparison.py render → "110 text artists, none outside the 936x1066 pt figure, no pair overlapping or under 1.5 pt apart"; closest to an edge 7.9 pt; tightest pair 1.9 pt
check: the check fails when it should → intermediate renders failed with 10, then 1, then 2 offenders (e.g. "0.9 pt apart (< 1.5): '+10%' / '+10%'") until fixed
check: render twice + md5sum -c → PNG e825bcc4… both times (deterministic)
check: git diff --quiet -- bench/charts/data; md5sum → data file untouched, ab41af6e…
check: render --dpi 76.95 --png-only → 1000x1138 PNG; viewed: legible
check: node validate_palette.js "#eb6834,#1baf7a,#2a78d6" --mode light --pairs all → PASS, worst CVD ΔE 9.2 (deutan), tritan 9.6, normal 24.0, aqua contrast WARN 2.74
check: uv run --with colorspacious --with scikit-image python /tmp/chart2/cvd.py → CIEDE2000 worst 14.6 (tritan aqua/blue), deutan 16.7, protan 14.7
check: copy of data + add --run-dir w8/zig-tp4-roce-pipe, render → 5 cells; "125 text artists … no pair overlapping"; viewed
check: uvx ruff check bench/charts/ → All checks passed!
check: git log --oneline -2 → 2cb723c8 on 92fdb5e1; 3 files (script, PNG, SVG); no amend, rebase or push

### Beyond-scope findings (round 2)
- The vision model's "clipped" labels were a misread (see the numbers above), but the one real overlap it did not report (TTFT tab vs unit label) was there. The render-time check now catches both kinds. worth-pursuing: no (done)
- The percent_note still says "Bracketed % ●"; the mark is now a box with a tab. The meaning still holds and the key shows the mark, but Depths may want to reword it to "Boxed % ●" in the data file. worth-pursuing: maybe
- When window 8 is appended, the subtitle and footnotes in the data file still describe only window 7 (visible in /tmp/chart2/w8test.png). They need editing with the data, as the docstring says. worth-pursuing: yes, at window 8
