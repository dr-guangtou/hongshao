#!/bin/bash
# usage: queue.sh MAXJOBS PROBE_VALUE — the three exp85 starts (adopted_probe, adopted_zero, adopted_far) from the
# adopted mean with q_c at PROBE_VALUE / 0 / 2 x PROBE_VALUE, at most MAXJOBS fits at once (the machine takes TWO:
# 11.5 GB each). Each fit is launched in its own session (launch.py) so it survives this shell and the Claude Code process.
cd /Users/shuang/Dropbox/work/project/massive/hongshao
export HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 PYTHONPATH=.
MAX=$1; PROBE=$2
OUT=experiments/exp85_compact_expansion/outputs
TAG=_q_e-q_c_delay0.15_fix-tau_d0.15
NAMES=(adopted_probe adopted_zero adopted_far)
for k in 0 1 2; do
  NAME=${NAMES[$k]}
  if [ -f $OUT/stage1_fit${TAG}_start_${NAME}.npz ]; then echo "skip $NAME (done)"; continue; fi
  if pgrep -f "starts $k:$k --knobs q_e,q_c" > /dev/null; then echo "skip $NAME (running)"; continue; fi
  while [ $(pgrep -f "stage1_fit.py --starts" | wc -l) -ge $((2 * MAX)) ]; do sleep 60; done   # uv wrapper + worker = 2 per fit
  PID=$(python3 experiments/exp82_delay_expansion/launch.py $OUT/fit_q_c_s$k.log -- \
    uv run python -u experiments/exp80_deposit_size_law/stage1_fit.py --starts $k:$k --knobs q_e,q_c \
    --delay 0.15 --fix tau_d=0.15 --start-adopted q_c=$PROBE --outdir $OUT)
  echo "$(date +%H:%M) launched $NAME pid $PID"
  sleep 90
done
echo "$(date +%H:%M) queue drained (last jobs still running)"
