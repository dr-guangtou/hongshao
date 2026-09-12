#!/bin/bash
# usage: queue.sh MAXJOBS [--delay-exp] — run the tau_d grid x three starts with at most MAXJOBS fits at once
cd /Users/shuang/Dropbox/work/project/massive/hongshao
export HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 PYTHONPATH=.
MAX=$1; shift; FORM="$*"; TAG=$(echo "$FORM" | grep -q exp && echo "_exp" || echo "")
OUT=experiments/exp82_delay_expansion/outputs
TAUS="${TAUS:-0.10 0.15 0.20 0.30}"
for TAU in $TAUS; do for k in 0 1 2; do
  TAUG=$(python3 -c "print(f'{float(\"$TAU\"):g}')")
  NAME=$(case $k in 0) echo tuned;; 1) echo baseline;; 2) echo far;; esac)
  if [ -f $OUT/stage1_fit_delay${TAUG}${TAG}_fix-tau_d${TAUG}_start_${NAME}.npz ]; then echo "skip tau $TAU$TAG $NAME (done)"; continue; fi
  while [ $(pgrep -f "stage1_fit.py --starts" | wc -l) -ge $((2 * MAX)) ]; do sleep 60; done   # uv wrapper + worker = 2 per fit
  PID=$(python3 experiments/exp82_delay_expansion/launch.py $OUT/fit_tau${TAU}${TAG}_s$k.log -- \
    uv run python -u experiments/exp80_deposit_size_law/stage1_fit.py --starts $k:$k --delay $TAU --fix tau_d=$TAU $FORM --outdir $OUT)
  echo "$(date +%H:%M) launched tau $TAU$TAG $NAME pid $PID"
  sleep 90
done; done
echo "$(date +%H:%M) queue drained (last jobs still running)"
