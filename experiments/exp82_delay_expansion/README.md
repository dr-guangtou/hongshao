# exp82 — the deposition delay and the expansion exponent together (2026-09-12)

Plan: `doc/plans/2026-09-12-exp82-delay-expansion.md`. The scripts are
exp80's `stage1_fit.py` / `stage1_eval.py` run with `--outdir` into this
directory (`run_round.sh`, `judge.sh`); the model is `size_law.predict_law`
with a delay spec (`--delay TAU --fix tau_d=TAU`, `--delay-exp` for the
exponential arrival) and the expansion exponent q_e free.

Results are appended below as the rounds finish.

## The grid (step arrival, tau_d held; q_e free; three starts each; judged tables-only, the 'tuned' start named unless noted)

| tau_d | starts: loss (q_e) | basin | loss (full nodes) | offset / width of 15 | R50 z=1.5 / 2 | R80 z=2 | slope z=2 | M(<2) z=0.4 / 1.5 / 2 | leak z=0.7 / 1 / 1.5 / 2 | 50–100 kpc z=2 fit \| mh-c |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| baseline | — | — | 15.56 | 12 / 0 | +0.037 / +0.054 | +0.039 | 0.33 | +5.6 / −9.8 / −12.6 | −0.013 / −0.061 / −0.038 / −0.007 | +29.6 \| −0.5 |
| 0.10 | tuned 13.09 (0.130), baseline 13.10 (0.110), far 13.09 (0.145) | one | 13.06 | 13 / 0 | +0.020 / +0.033 | −0.017 | — | +0.4 / −7.6 / −8.8 | −0.048 / −0.071 / −0.047 / −0.039 | +10.0 \| −14.0 |
