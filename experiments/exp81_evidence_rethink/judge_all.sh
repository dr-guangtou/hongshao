#!/bin/zsh
# merge + judge the six controlled fits (tables only; figures for the adopted one later)
cd /Users/shuang/Dropbox/work/project/massive/hongshao
export HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 PYTHONPATH=.
FIT=experiments/exp80_deposit_size_law/stage1_fit.py
EVAL=experiments/exp80_deposit_size_law/stage1_eval.py
OUT=experiments/exp80_deposit_size_law/outputs
judge() {  # $1 = label, $2.. = fit args (merge), then eval args via env EVALARGS
  uv run python -u $FIT --merge "${@:2}" > $OUT/merge_$1.log 2>&1
  if [[ "$EVALARGS" == *"--best"* ]]; then B=""; else B="--best tuned"; fi
  uv run python -u $EVAL --tables-only $B $EVALARGS > $OUT/stage1_eval_$1.log 2>&1
  echo "judged $1"
}
case "$1" in
  qfix)
    for q in 0.08 0.127 0.2; do
      EVALARGS="--fit-tag _fix-q_e$q" judge "fix_q$q" --fix q_e=$q &
    done; wait ;;
  delay)
    EVALARGS="--fit-tag _noknob_delay0.15 --knobs none --delay --best baseline" judge "delay_alone" --knobs none --delay 0.15 &
    EVALARGS="--fit-tag _delay0.15_fix-q_e0.127 --delay" judge "delay_qfix" --delay 0.15 --fix q_e=0.127 &
    EVALARGS="--fit-tag _delay0.15 --delay" judge "delay_qfree" --delay 0.15 &
    wait ;;
esac
