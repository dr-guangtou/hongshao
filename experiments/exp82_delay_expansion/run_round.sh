#!/bin/bash
# usage: run_round.sh TAU [--delay-exp]   — three starts (tuned, baseline, far) at tau_d held
cd /Users/shuang/Dropbox/work/project/massive/hongshao
export HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 PYTHONPATH=.
TAU=$1; shift; FORM="$*"; TAG=$(echo "$FORM" | grep -q exp && echo "_exp" || echo "")
OUT=experiments/exp82_delay_expansion/outputs
for k in 0 1 2; do
  nohup uv run python -u experiments/exp80_deposit_size_law/stage1_fit.py --starts $k:$k --delay $TAU --fix tau_d=$TAU $FORM --outdir $OUT \
    > $OUT/fit_tau${TAU}${TAG}_s$k.log 2>&1 &
  PID=$!; caffeinate -i -w $PID > /dev/null 2>&1 &
  echo "tau $TAU$TAG start $k pid $PID"
done
