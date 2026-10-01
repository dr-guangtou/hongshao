#!/bin/bash
# usage: run_stage3.sh BLOCK [BLOCK ...] — run the named Stage 3 blocks one after another in this process group
cd /Users/shuang/Dropbox/work/project/massive/hongshao
export PYTHONPATH=. OMP_NUM_THREADS=2
for B in "$@"; do
  uv run python -u experiments/exp87_inverse_halo/stage3_ladder.py --block "$B"
done
echo "RUN_STAGE3 DONE $*"
