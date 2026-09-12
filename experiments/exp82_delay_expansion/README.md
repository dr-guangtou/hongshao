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
| 0.15 (tuned basin) | tuned 12.75 (0.152) | two | 12.70 | **14 / 1** | +0.016 / +0.027 | −0.020 | — | +0.5 / −6.2 / −6.9 | −0.043 / −0.039 / −0.015 / −0.026 | +7.5 \| −17.7 |
| 0.15 (wide basin) | baseline 12.27 (0.385), far 12.30 (0.520) | " | 12.26 | 13 / 0 (R50 widths 0.52 → 0.25) | +0.031 / +0.032 | −0.014 | — | +5.8 / −8.3 / −9.8 | −0.042 / −0.051 / −0.031 / −0.029 | +0.4 \| −21.4 |
| 0.20 (wide basin, all starts) | tuned 12.16 (0.557), baseline 12.15 (0.488), far 12.14 (0.445) | one | 12.29 | 15 / 0 (R50 widths 0.46 → 0.21) | +0.018 / +0.001 | −0.040 | — | +5.4 / −6.7 / −7.1 | −0.042 / −0.034 / −0.019 / −0.013 | −8.9 \| −32.4 |
| 0.30 (wide basin, all starts) | tuned 12.69 (0.598), baseline 12.70 (0.568), far 12.70 (0.543) | one | 12.69 | 12 / 0 (R50 widths 0.41 → 0.20; R80 z=2 −0.067 OFF) | +0.018 / +0.011 | −0.067 | — | +5.1 / −9.0 / −9.5 | −0.046 / −0.026 / −0.014 / +0.010 | −15.6 \| −36.6 |

**The grid, read.** The loss's minimum in tau_d is near 0.20 (12.15) and
the loss prefers, at every tau_d ≥ 0.15, the wide-split family (d_split
≈ 1.1–1.3, q_e 0.4–0.6) that narrows every size distribution to 0.2–0.5
of the truth's and returns the z = 0.4 centre to +5 per cent — the family
every earlier judge rejected. The baseline's structure (d_split 0.27, n_c
0.7, q_e settling at 0.11–0.15) survives only at tau_d ≤ 0.15, and at 0.15
only from the start that carries Stage 0 C's size constants. **The
gate-chosen grid point is tau_d = 0.15, the tuned-start basin** (14 of 15
offsets, one width pass, the centre +0.5 / −6.2 / −6.9 per cent, widths
0.73 → 0.35 of the truth's, loss 12.70): it beats tau_d = 0.10 (13, leak
worse at every epoch) and the wide basins on the widths and the centre at
equal or better offset counts. Its known costs stand: the mh-complete
progenitors' 50–100 kpc shell at z ≥ 1.5 (−12 / −18 per cent; the
cumulative M(<103) there +3.4 / −1.4), and the future-dependence gate at
z = 0.7 (−0.043 against the baseline's −0.013, the truth −0.005).

## The chosen model's checks (`outputs/residual_check_tau0.15_tuned.log`)

- **The mechanism holds on the fitted model.** The residual's correlation
  with the halo's recent growth at fixed halo mass is |ρ| ≤ 0.13 at every
  cell (the baseline +0.27 to +0.58); the residual's out-of-fold
  halo-predictability at z ≥ 1 is R² 0.02–0.06 (the baseline 0.22–0.35),
  i.e. at the halo-information ceiling; the rms falls from 0.189 → 0.165
  dex at z = 2, 5 kpc and 0.167 → 0.139 at z = 2, 103 kpc.
- **What is left.** At z = 0.4 the residual correlates with the EARLY mass
  (+0.37 at 5 kpc, +0.33 at 103 kpc; R² 0.22 / 0.15; rms at 5 kpc 0.136 →
  0.146): early-formed haloes are now too heavy at low redshift — exp83's
  (S3) target.
- **The decline.** For the 42 per cent of galaxies whose true central mass
  fell from z = 2 to 0.4 (−0.066 dex at 4.9 kpc), the model's change is
  +0.039 (the baseline +0.061): a third of the mismatch removed by the
  expansion, the decliner span 0.147 → 0.120 dex; the non-decliners' span
  0.082 → 0.105 (the expansion acts on every galaxy). A subpopulation
  mechanism is still owed (C8, S3).

## The full judge with figures on the chosen model (`outputs/eval_tau0.15.log`, `figures/qa/*exp80_*`; the model is labelled `exp80` there — the judge script's name)

Parameters: **14 in the model (exp63's twelve, tau_d, q_e); 13 fitted at
each grid point with tau_d held; tau_d chosen by the gates on the grid.**
Loss 12.70 (baseline 15.56, the old loss-best basin 14.63). Size gate at
fixed stellar mass 14 of 15 offsets and 1 width (R20 at z = 0.4, 0.73 of
the truth's scatter); at fixed halo mass 14 / 0. R50 offsets −0.013 /
−0.009 / −0.008 / +0.016 / +0.027 dex (baseline −0.016 / −0.027 / −0.010 /
+0.037 / +0.054); R20 at z = 2 +0.054 (baseline +0.092); R80 at z = 2 −0.020
(baseline +0.039). The centre M(<2 kpc) +0.5 / −6.2 / −6.9 per cent at
z = 0.4 / 1.5 / 2 (baseline +5.6 / −9.8 / −12.6). The mass planes
(`qa_planes_exp80_exp80.png`): no knee; the outer-vs-inner slopes a little
shallower than the truth at z ≤ 1.5 (M(30–50) | M(<30) 1.26 against 1.43 at
z = 0.4; the baseline 1.33) and right at z = 2 (1.59 against 1.42;
M(50–100) 1.60 against 1.66). The future-dependence gate −0.043 / −0.039 /
−0.015 / −0.026 dex per dex at z = 0.7 / 1 / 1.5 / 2 (baseline −0.013 /
−0.061 / −0.038 / −0.007; the truth −0.005 / +0.061 / +0.033 / +0.004):
worse at z = 0.7 and z = 2 by 0.03, better at z = 1 and 1.5. The
mh-complete profile: M(<10) +3.2 / +4.2 / +8.1 / +8.6 / +3.2 per cent
(baseline +2.4 / +6.4 / +7.8 / −0.5 at z = 0.7 … 2); M(<103) +1.1 / +5.1 /
+3.4 / −1.4 (baseline −0.3 / +4.5 / +8.6 / −1.0); the 50–100 kpc shell at
z = 1.5 / 2 −11.8 / −17.7 (baseline +4.4 / −0.5).

**Verdict.** Against the baseline the chosen model improves the loss by 18
per cent, the size offsets (12 → 14 of 15) with the first width pass, the
centre at every epoch, the z = 2 outer sizes and the mass–size slope, and
removes the formation-time residual the model class had always carried,
with one physical parameter (the delay) and one geometric one (the
expansion). Its costs are the mh-complete progenitors' 50–100 kpc shell at
z ≥ 1.5 and a 0.03 dex-per-dex rise of the future-growth dependence at
z = 0.7 and z = 2. **Recommended for adoption as the baseline mean** (the
user's call; `rebaseline.adopted_baseline()` and the CLAUDE.md rule would
then point at `outputs/stage1_fit_delay0.15_fix-tau_d0.15_start_tuned.npz`
with `size_law.predict_law` as the engine).
