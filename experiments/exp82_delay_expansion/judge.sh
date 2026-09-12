#!/bin/bash
# usage: judge.sh TAU [--delay-exp] [--figures] [--best NAME] — merge the grid point's starts and judge it
cd /Users/shuang/Dropbox/work/project/massive/hongshao
export HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 PYTHONPATH=.
TAU=$1; shift
FORM=""; TABLES="--tables-only"; BEST=""
while [ $# -gt 0 ]; do case "$1" in --delay-exp) FORM="--delay-exp";; --figures) TABLES="";; --best) BEST="--best $2"; shift;; esac; shift; done
EXP=$(echo "$FORM" | grep -q exp && echo "_exp" || echo "")
TAUG=$(python3 -c "print(f'{float(\"$TAU\"):g}')")   # the fit scripts format tau_d with %g (0.10 -> 0.1)
OUT=experiments/exp82_delay_expansion/outputs
uv run python -u experiments/exp80_deposit_size_law/stage1_fit.py --merge --delay $TAU --fix tau_d=$TAU $FORM --outdir $OUT > $OUT/merge_tau${TAU}${EXP}.log 2>&1
uv run python -u experiments/exp80_deposit_size_law/stage1_eval.py $TABLES --fit-tag _delay${TAUG}${EXP}_fix-tau_d${TAUG} --delay $FORM $BEST --outdir $OUT > $OUT/eval_tau${TAU}${EXP}.log 2>&1
echo "judged tau $TAU$EXP"
