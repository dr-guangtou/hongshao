#!/bin/bash
# usage: judge.sh [--figures] [--best NAME] [--also NAMES] — merge the exp85 starts and judge the named one (never the
# merged file's best: the loss's favourite may be the gate-rejected basin) against the adopted mean and the baseline
cd /Users/shuang/Dropbox/work/project/massive/hongshao
export HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 PYTHONPATH=.
TABLES="--tables-only"; BEST=""; ALSO=""; BESTNAME=""
while [ $# -gt 0 ]; do case "$1" in --figures) TABLES="";; --best) BEST="--best $2"; BESTNAME="$2"; shift;; --also) ALSO="--also $2"; shift;; esac; shift; done
OUT=experiments/exp85_compact_expansion/outputs
TAG=_q_e-q_c_delay0.15_fix-tau_d0.15
uv run python -u experiments/exp80_deposit_size_law/stage1_fit.py --merge --knobs q_e,q_c --delay 0.15 --fix tau_d=0.15 --outdir $OUT > $OUT/merge.log 2>&1
uv run python -u experiments/exp80_deposit_size_law/stage1_eval.py $TABLES --knobs q_e,q_c --fit-tag $TAG --delay $BEST $ALSO --name exp85 --outdir $OUT > $OUT/eval${BESTNAME:+_$BESTNAME}.log 2>&1
echo "judged exp85 $BEST"
