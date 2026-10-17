#!/usr/bin/env bash
# glmw.sh — the GLM-5.3-Flash FP4-Local GPU window on four Sparks (tools/glm/WINDOW.md), run from the workstation.
# Configuration from $GLMW_ENV (a file kept outside the repo: host names and fabric addresses never go in here):
#   GLMW_HOSTS="h0 h1 h2 h3"   rank order       GLMW_MASTER=<rank 0's RoCE IPv4>   GLMW_IFACE=<RoCE netdev>
#   GLMW_HCA=<RoCE HCA>        GLMW_RANGE=<RoCE /24>   GLMW_BIN=<dir with tensorfold-native, tf-cuda-test>
#   GLMW_LIB=<dir with libnccl.so.2, libtfgrammar.so>   GLMW_MODEL=<checkpoint dir>   GLMW_PORT=8190   GLMW_REF=<CPU host>
#   GLMW_VENV=<reference venv on GLMW_REF>   GLMW_TOOLS=<tools/glm copy on GLMW_REF>   GLMW_B12X=<_csf_blocks.py dir>
# Subcommands (each prints what it ran; logs go to $GLMW_RUNS/<stamp>/ on the hosts):
#   pre                         memory, GPU processes, binaries' and libraries' sha256 on every host
#   glmref                      upstream's synthetic kernel gate on rank 0's GPU (tf-cuda-test glm-ref)
#   vision REF [IMG]            the CUDA image tower stage by stage vs tools/glm/vision_ref.py's dumps in REF (host 0)
#   thin LAYERS [dump]          one rank (TF_GLM_WORLD=1) on host 0 with LAYERS layers, --no-drafts; dump: TF_GLM_DUMP
#   tp [drafts] [CONTEXT]       all four ranks over NCCL (RoCE), 45 layers; drafts: MTP on
#   wait                        until rank 0 lists the model (prints the load time)
#   gate ROWS LAYERS            the last thin run's dump to GLMW_REF, then tools/glm/layer_ref.py there
#   stop                        SIGTERM every tensorfold-native, then check no GPU process is left
#   status                      processes, GPU memory, each rank's last log lines
set -uo pipefail
: "${GLMW_ENV:?set GLMW_ENV to the window configuration file}"
# shellcheck disable=SC1090
. "$GLMW_ENV"
read -r -a H <<<"$GLMW_HOSTS"
PORT=${GLMW_PORT:-8190}
STAMP_FILE=${GLMW_STAMP_FILE:-/tmp/glmw.stamp}
say() { echo "[glmw $(date +%T)] $*"; }
on() { local h=$1; shift; ssh -o BatchMode=yes -o ConnectTimeout=10 -n "$h" "$@"; }
run_dir() { [ -f "$STAMP_FILE" ] || date +%m%d-%H%M >"$STAMP_FILE"; echo "${GLMW_RUNS:-glm53-tf/runs}/$(cat "$STAMP_FILE")"; }
NCCL_ENV="NCCL_NET=IB NCCL_IB_DISABLE=0 NCCL_IB_ROCE_VERSION_NUM=2 NCCL_IB_ADDR_FAMILY=AF_INET NCCL_IB_ADDR_RANGE=$GLMW_RANGE \
NCCL_IB_HCA=$GLMW_HCA NCCL_SOCKET_IFNAME=$GLMW_IFACE NCCL_CROSS_NIC=1 NCCL_NVLS_ENABLE=0 NCCL_IB_MERGE_NICS=0 NCCL_CUMEM_ENABLE=0 \
NCCL_IGNORE_CPU_AFFINITY=1 NCCL_DEBUG=${NCCL_DEBUG:-WARN}"
SERVE_ENV="LD_LIBRARY_PATH=$GLMW_LIB TENSORFOLD_REASONING_FIELDS=both"

start_rank() { # host rank world layers dump drafts context
  local h=$1 r=$2 w=$3 layers=$4 dump=$5 drafts=$6 ctx=$7 d; d=$(run_dir)
  local env="$SERVE_ENV TF_GLM_RANK=$r TF_GLM_WORLD=$w"
  [ "$layers" != 0 ] && env+=" TF_GLM_LAYERS=$layers"
  [ "$dump" = dump ] && env+=" TF_GLM_DUMP=\$HOME/$d/dump"
  [ "$w" -gt 1 ] && env+=" NCCL_COMM_ID=$GLMW_MASTER:29660 $NCCL_ENV"
  local flags="--host 0.0.0.0 --port $PORT --no-update-check --context $ctx --parallel 1 --name GLM-5.3-Flash-FP4"
  [ "$drafts" = drafts ] || flags+=" --no-drafts"
  [ -n "${GLMW_VISION:-}" ] && flags+=" --vision"  # GLMW_VISION=1: the image tower on rank 0
  say "$h rank $r/$w: $env tensorfold-native serve MODEL $flags"
  on "$h" "mkdir -p ~/$d/dump && cd ~/$d && date +%s.%N > r$r.start && \
    env $env setsid nohup $GLMW_BIN/tensorfold-native serve $GLMW_MODEL $flags > r$r.log 2>&1 < /dev/null & echo started"
}

case "${1:-status}" in
pre)
  for h in "${H[@]}"; do
    on "$h" "echo \$(hostname): avail \$(awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo) GiB; \
      nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; \
      (cd $GLMW_BIN && sha256sum -c SHA256SUMS | tr '\n' ' '); (cd $GLMW_LIB && sha256sum -c SHA256SUMS | tr '\n' ' '); echo"
  done ;;
glmref)
  d=$(run_dir); on "${H[0]}" "mkdir -p ~/$d && cd ~/$d && LD_LIBRARY_PATH=$GLMW_LIB $GLMW_BIN/tf-cuda-test glm-ref 2>&1 | tee glm-ref.log | tail -20" ;;
vision)
  d=$(run_dir); on "${H[0]}" "mkdir -p ~/$d && cd ~/$d && LD_LIBRARY_PATH=$GLMW_LIB $GLMW_BIN/tf-cuda-test glm-vision $GLMW_MODEL ${2:?ref dir} ${3:-} 2>&1 | tee glm-vision.log | tail -40" ;;
thin)
  start_rank "${H[0]}" 0 1 "${2:?layers}" "${3:-}" nodrafts "${GLMW_CONTEXT:-4096}" ;;
tp)
  drafts=${2:-nodrafts}; ctx=${3:-32768}
  # rank 0 first (its NCCL_COMM_ID root listens), the others at once; each ssh in the background (a rank's start may
  # hold its ssh session until it exits)
  for r in 0 1 2 3; do start_rank "${H[$r]}" "$r" 4 0 "" "$drafts" "$ctx" & sleep 3; done; sleep 5 ;;
wait)
  d=$(run_dir); t0=$(date +%s)
  for _ in $(seq 1 360); do
    if curl -s -m 5 "http://${H[0]}:$PORT/v1/models" | grep -q GLM-5.3-Flash-FP4; then say "rank 0 serving after $(( $(date +%s) - t0 )) s of waiting"; exit 0; fi
    on "${H[0]}" "pgrep -f '[t]ensorfold-native serve' >/dev/null" || { say "rank 0 exited"; on "${H[0]}" "tail -30 ~/$d/r0.log"; exit 1; }
    sleep 10
  done; say "not serving after 3600 s"; exit 1 ;;
gate)
  rows=${2:?rows}; layers=${3:?layers}; d=$(run_dir)
  on "${H[0]}" "cd ~/$d && tar cf - dump" | on "$GLMW_REF" "mkdir -p ~/$d && cd ~/$d && tar xf -"
  on "$GLMW_REF" "cd ~/$d && $GLMW_VENV/bin/python -I $GLMW_TOOLS/layer_ref.py $GLMW_MODEL dump --rows $rows --layers $layers --b12x $GLMW_B12X 2>&1 | tee gate-$layers.log" ;;
stop)
  for h in "${H[@]}"; do on "$h" "pkill -TERM -f '[t]ensorfold-native serve' ; true"; done
  sleep 5
  for h in "${H[@]}"; do
    for _ in $(seq 1 30); do on "$h" "pgrep -f '[t]ensorfold-native serve' >/dev/null" || break; sleep 2; done
    on "$h" "pkill -KILL -f '[t]ensorfold-native serve'; echo \$(hostname): left \$(pgrep -fc '[t]ensorfold-native serve'); nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader"
  done ;;
status)
  d=$(run_dir)
  for r in "${!H[@]}"; do
    on "${H[$r]}" "echo \$(hostname) r$r: \$(pgrep -fc '[t]ensorfold-native serve') proc, avail \$(awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo) GiB; \
      nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; tail -${TAIL:-5} ~/$d/r$r.log 2>/dev/null"
  done ;;
*) sed -n 2,20p "$0"; exit 2 ;;
esac
