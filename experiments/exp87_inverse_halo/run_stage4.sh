#!/bin/bash
# usage: run_stage4.sh CELL [CELL ...] — the symbolic-regression driver on the named cells, one after another
cd /Users/shuang/Dropbox/work/project/massive/hongshao
export PYTHONPATH=. OMP_NUM_THREADS=1
uv run python -u experiments/exp87_inverse_halo/stage4_driver.py "$@"
