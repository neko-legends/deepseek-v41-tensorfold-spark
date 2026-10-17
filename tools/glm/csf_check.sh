#!/usr/bin/env bash
# Bit-exact check of engine/zig-glm's csf_32x2_v1 decoder (zig/src/core/csf.zig) against FlashInfer B12X's own decoder
# on real routed-expert scale planes of nvidia/GLM-5.3-Flash-FP4-Local, fetched by HTTP range (headers + scale tensors
# only, a few hundred KB a plane). Needs python3 with numpy, curl, and zig 0.17.0 (ZIG).
#   tools/glm/csf_check.sh [PLANES]     (default 48 random planes over layers 3..44 and all three projections)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ZIG="${ZIG:-zig}"
PY="${PY:-python3}"
WORK="${GLM_WORK:-${TMPDIR:-/tmp}/glm-csf-check}"
B12X_SHA=7e203806a82b588b9113f5a0866a028564847741 # local-inference-lab/flashinfer main, 2026-10-17
N="${1:-48}"
mkdir -p "$WORK/ref" "$WORK/tensors"
export GLM_HEADERS="$WORK/headers.json" GLM_TENSORS="$WORK/tensors"
[ -s "$GLM_HEADERS" ] || "$PY" -I tools/glm/hf_headers.py nvidia/GLM-5.3-Flash-FP4-Local > "$GLM_HEADERS" 2>/dev/null
[ -s "$WORK/ref/_csf_blocks.py" ] || curl -sSfL -o "$WORK/ref/_csf_blocks.py" \
    "https://raw.githubusercontent.com/local-inference-lab/flashinfer/$B12X_SHA/flashinfer/experimental/b12x/loader/_csf_blocks.py"
"$ZIG" build-exe -OReleaseSafe --dep csf -Mroot=engine/zig-glm/zig/tools/glm/csf_decode.zig \
    -Mcsf=engine/zig-glm/zig/src/core/csf.zig --name csf_decode -femit-bin="$WORK/csf_decode"
mods=$("$PY" -I -c "
import random; r = random.Random(53)
for _ in range($N):
    print(f'model.language_model.layers.{r.randrange(3, 45)}.mlp.experts.{r.randrange(288)}.{r.choice([\"gate_proj\", \"up_proj\", \"down_proj\"])}')
" | sort -u)
"$PY" -I tools/glm/csf_ref.py "$WORK/ref" "$WORK/ref-out" $mods | awk '{print $1"  "$2}' | sort -k2 > "$WORK/ref.sha256"
: > "$WORK/zig.sha256"
for shape in "2048 256 gate_proj" "2048 256 up_proj" "4096 128 down_proj"; do
    set -- $shape
    sel=$(printf '%s\n' $mods | grep "\.$3\$" || true)
    [ -n "$sel" ] && "$WORK/csf_decode" "$GLM_TENSORS" "$1" "$2" $sel 2>> "$WORK/zig.sha256"
done
sort -k2 -o "$WORK/zig.sha256" "$WORK/zig.sha256"
planes=$(wc -l < "$WORK/ref.sha256")
if diff -q "$WORK/ref.sha256" "$WORK/zig.sha256" > /dev/null && [ "$planes" -gt 0 ]; then
    echo "csf_check: $planes planes bit-exact (B12X _csf_blocks.py @${B12X_SHA:0:8} vs zig core/csf.zig)"
else
    diff "$WORK/ref.sha256" "$WORK/zig.sha256" | head; echo "csf_check: MISMATCH"; exit 1
fi
