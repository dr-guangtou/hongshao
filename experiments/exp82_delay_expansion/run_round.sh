#!/bin/bash
# usage: run_round.sh TAU [--delay-exp]   — three starts (tuned, baseline, far) at tau_d held, each fully detached
cd /Users/shuang/Dropbox/work/project/massive/hongshao
export HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 PYTHONPATH=.
TAU=$1; shift; FORM="$*"; TAG=$(echo "$FORM" | grep -q exp && echo "_exp" || echo "")
OUT=experiments/exp82_delay_expansion/outputs
for k in 0 1 2; do
  PID=$(python3 experiments/exp82_delay_expansion/launch.py $OUT/fit_tau${TAU}${TAG}_s$k.log -- \
    uv run python -u experiments/exp80_deposit_size_law/stage1_fit.py --starts $k:$k --delay $TAU --fix tau_d=$TAU $FORM --outdir $OUT)
  echo "tau $TAU$TAG start $k pid $PID"
done
