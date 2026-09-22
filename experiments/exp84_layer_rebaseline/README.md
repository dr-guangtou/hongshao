# exp84 — the stochastic layer re-baselined on the adopted mean, with an outer size component (S4)

Branch `exp84-layer-rebaseline` (from master `17f2127`). Plan
`doc/plans/2026-09-22-exp84-layer-rebaseline.md` (approved 2026-09-22).
Everything below is committed scripts + gitignored outputs; every stage's
numbers are in `outputs/stage*.log`.

**Status: Stage 2 judged held out (two rounds); Stage 3 packaging ready;
adoption is the user's call.**

## What was built

The v1 layer (exp60) ran on exp57's X3 problem and the official-curve step
engine; neither is the adopted mean's. This experiment rebuilds the layer
on `rebaseline.adopted_mean()` (exp82: tau_d 0.15 + q_e 0.152, the measured
history) — the half-day rebuild owed since exp74 — with the components the
user chose (2026-09-22):

- **amplitude**: v1's cross-epoch-correlated Gaussian at 103 kpc, widths
  calibrated in quadrature to the mean's residual (Stage 0: 0.105–0.133 dex,
  nearest-epoch correlation 0.66);
- **the compact deposit size `log_f_c` and the extended deposit size
  `log_f_e`** (the outer component), drawn PER GALAXY AND PER EPOCH from a
  Gaussian whose 10×10 correlation (5 epochs × 2 axes) is fitted to the
  anatomy — the deviation acts at the observed epoch's evaluation through
  `size_law.predict_law(size_dev=)` (a zero deviation nests bit for bit);
- X3 dropped; the decline targets (S1) reported, not gated.

The scripts: `predictor.py` (the mean on the measured history; one call on
2356 galaxies = 1.2–2.4 s at 11 GB from the build), `stage0_targets.py`,
`stage1_anatomy.py`, `sampler.py`, `stage2_judge.py`, `stage3_adopt.py`;
`hongshao/qa.py::evaluate_draws` scores tiers 2d and 2e on drawn
populations through the same code path as a mean (C16 as a procedure).

## Stage 0 — the targets on the adopted mean (`outputs/stage0_targets.log`)

- Amplitude residual at 103 kpc: half 16–84 width 0.109 / 0.106 / 0.105 /
  0.109 / 0.133 dex (z = 0.4 → 2), nearest-epoch correlation 0.66.
- **The persistence diagnostic** (Spearman of the size residual at fixed
  M*(<148) across neighbouring epochs): the TRUTH 0.68 / 0.71 / 0.59 for
  R20 / R50 / R80; the mean's own 0.90 / 0.81 / 0.71 — a galaxy's size rank
  is less persistent in TNG than the halo's, so the layer's deviation must
  itself be less than fully persistent (decision 2).
- The null row: the mean scores offset 14/15, width 1/15 at fixed M* (0/15
  at fixed Mh); S1: 42.3 per cent of centres decline in the truth, 4.9 per
  cent in the mean.

## Stage 1 — the anatomy (`outputs/stage1_anatomy.log`, 222 calls, 8.9 min)

Per galaxy and per epoch, the deviation of each size that minimises that
galaxy-epoch's loss, by the SMALLEST-SUFFICIENT rule (the smallest |δ|
within 1 per cent of the best: a compact deposit pushed below the 2 kpc
grid sits on a flat valley, and plain argmin put 37 per cent of the axis on
the −1 dex edge).

- The two sizes are the shape axes: `log_f_e` 34.8 per cent median gain
  (its degenerate partner `b_e` 36.0), `log_f_c` 23.1 (`b_c` 23.3); the
  extended shape `c_e` is a third candidate at 30.7; the amplitude
  parameters lead at ~50 (the amplitude draw's job). The joint (δ_c, δ_e)
  reaches 46.5 per cent; 13.5 per cent of galaxy-epochs sit at a grid edge
  (the anatomy's ceiling).
- Widths (half 16–84): δ_c 0.41–0.62 dex, δ_e 0.21–0.29 dex. Column
  medians within ±0.04 dex except δ_c at z = 0.4 (+0.14: the centre wants
  larger compact deposits at low z — a mean-model residual, NOT applied).
- Correlation structure: nearest-epoch 0.43 (c), 0.57 (e); c × e −0.22 to
  −0.35 (the two sizes trade against each other).
- What a conditioning would see: |ρ| ≤ 0.29 (δ_e with the early-mass
  fraction +0.29 and f_form −0.25 at z = 0.4; δ_c with log Mh −0.29 at
  z = 1). Reported for S6; not used.

## Stage 2, round 1 — held out, both swaps, 8 realizations each (`outputs/stage2_judge.log`)

Tier 2d on the DRAWS (offset = median dlog R, pass ≤ 0.05; width = the
model's size scatter at fixed mass over the truth's, pass within 20 per
cent), the v1 row from exp73's record of the v1 layer on ITS OWN mean:

| row | fixed M*: offset / width of 15 | fixed Mh: offset / width | S3 (z ≤ 1) | S5 | persistence R20/R50/R80 (truth 0.68/0.71/0.59) |
|---|---|---|---|---|---|
| the adopted mean | 14 / 1 | 14 / 0 | — | — | 0.90/0.81/0.71 |
| v1 on the incumbent | 10 / 3 | 10 / 3 | — | exact | — |
| **gauss** (the fitted layer) | **15 / 9** | **15 / 9** | 0.038 | exact | 0.58/0.58/0.56 |
| gauss-scaled (one scale 0.80 on R50) | 15 / 8 | 15 / 11 | 0.029 | exact | 0.62/0.60/0.57 |
| rows (nonparametric) | 15 / 5 | 15 / 8 | 0.039 | exact | 0.64/0.62/0.61 |
| independent (corr = I) | 15 / 5 | 15 / 8 | 0.034 | exact | 0.19/0.18/0.15 |
| persistent (corr = 1) | 15 / 9 | 15 / 9 | 0.036 | exact | 0.95/0.92/0.91 |

Reading the cells, not the tally:

- **R80 — the outer component works**: width 0.26–0.66 (mean) → 0.85–1.17
  (gauss), 5/5 at both conditionings; v1 had 0.48–0.61.
- **R50**: 4/5 (1.10–1.17 at z ≤ 1.5, 1.36 at z = 2); v1 3/5 with offsets
  failing at z ≥ 1.
- **R20 over-dispersed 1.34–1.57×** in every variant (v1: 1.3–2.3×): the
  compact axis's anatomy width is inflated by its flat loss valley.
- **S3 fails** (0.03–0.04 dex at z ≤ 1; gate 0.010): a symmetric log-size
  draw moves the median profile because the profile's response to a larger
  and a smaller deposit is not symmetric (v1 had the same 0.058 inside
  10 kpc).
- **Decision 2 validated**: the fitted correlation reproduces the truth's
  size persistence (0.56–0.58 vs 0.59–0.71); independent draws give 0.19,
  a persistent trait 0.95. S5 exact, S4 +0.002, tier 2e improves in the
  outskirts (M(30–50), M(50–100): 2.4–3.5 → 1.1–1.7 × the floor) and
  worsens M(<10 kpc) at z = 0.4 (0.9 → 3.2), the same compact-axis excess.
- S1 for the record: declining 35 per cent (truth 42.3), decliner median
  −0.12 (truth −0.066) — the v1 walls, unchanged in kind.

## Stage 2, round 2 — the two-scale and profile-centred variants (`outputs/stage2_judge_round2.log`)

Added after round 1 (both held out, same protocol): a profile-CENTRED
variant (per-axis, per-epoch offsets calibrated so the drawn median stays on
the mean at 4.9 / 32.6 kpc) for S3, and a TWO-SCALE variant (the compact
axis's scale calibrated on R20's width, the extended axis's on R80's; R50
is then the test) for R20.

| row | fixed M*: offset / width of 15 | fixed Mh: offset / width | S3 (z ≤ 1 / all) | S4 | persistence R20/R50/R80 |
|---|---|---|---|---|---|
| gauss-centred | 9 / 9 | 9 / 9 | 0.078 / 0.100 | −0.019 | 0.58/0.58/0.56 |
| **gauss-2scale** (c × 0.051, e × 1.017) | **15 / 12** | **15 / 15** | **0.012 / 0.031** | +0.003 | 0.67/0.61/0.57 |
| gauss-2sc-cent | 12 / 11 | 12 / 15 | 0.028 / 0.048 | −0.020 | 0.66/0.61/0.58 |

- **The compact-size draw switches itself off.** Calibrated on R20's width,
  the compact axis's scale runs to the bisection floor (0.051 — the draw is
  0.02–0.03 dex, nothing): the anatomy's 0.4–0.6 dex width on that axis was
  the flat loss valley, not a size diversity the gates want. Even with it
  off, R20's width at fixed M* is 1.20 at z ≤ 1 (from the extended draw and
  the amplitude), 1.08–1.11 at z ≥ 1.5, and 1.05–1.11 at fixed Mh (5/5).
  **The layer that works is amplitude + extended size.**
- **The two-scale layer passes every offset and 27 of 30 widths**: the
  three misses are R20 at z ≤ 1 (1.20–1.21, on the line) and R50 at z = 0.4
  (1.20) at fixed M*. R80: 0.87–1.20, 5/5 at both. S5 exact (1.02 / 1.01 /
  0.99 / 1.00 / 0.98; nearest-epoch 0.650 vs 0.660). Persistence within
  0.10 of the truth's on every size.
- **S3**: 0.012 dex at z ≤ 1 — over the 0.010 gate by the rounding — and
  0.031 over all epochs (z = 2, where the extended width is 0.29 dex, the
  Jensen shift is largest). Once the compact draw is off the shift is the
  extended axis's.
- **Centring is harmful**: pinning the median at two radii pushes it off
  everywhere else (S3 0.078) and shifts the size relation (offsets 9/15).
  The median-profile shift is radius-structured; two offsets cannot centre
  it. If S3 must reach 0.010, the lever is the extended axis's width at
  z ≥ 1.5 or a skewed (log-symmetric in the PROFILE) draw — not offsets.

## Stage 3 — the package (`outputs/stage3_adopt.log`)

`stage3_adopt.py --variant gauss-2scale`: the two-scale layer calibrated
on the full fitting sample, frozen to `outputs/hongshao_v2_layer.npz`
(sigma per epoch and axis, the 10×10 correlation, the two scales, sig_add,
the amplitude correlation, seeds; `draw_cogs(pred, layer, rows, rng)` is
the entry point), and the standard battery with eight drawn populations
overlaid in `figures/qa/qa_*_exp84_v2_layer.*`. Its full-sample tables are
in-sample for the widths; Stage 2 round 2 is the validation record. The
packaged scales are c 0.051, e 0.976; sig_add 0.106 / 0.105 / 0.104 /
0.109 / 0.133 dex; full sample: offsets 15/15 at both conditionings,
widths 13/15 (M*) and 15/15 (Mh), S3 0.011 / 0.030, S5 1.01 / 1.01 / 0.99 /
0.99 / 0.98 with nearest-epoch 0.656 (target 0.660), S4 +0.002, persistence
0.67 / 0.61 / 0.57, declining 34.0 per cent. `qa_planes_exp84_v2_layer`
(truth filled, mean open, draw ×): the draws fill the truth's scatter in
the kpc mass planes at z ≤ 1 (M(30)–M(50–100): 0.22 vs 0.20 at z = 0.4),
stay under it at z ≥ 1.5 (0.23–0.28 vs 0.32–0.46 — the high-z outskirts
remain under-dispersed in mass even where the size widths pass), and
OVER-disperse the Re-relative planes (M(<2Re)–M(2–4Re): 0.16 vs 0.07 at
z = 0.4): the layer's noise is not the truth's self-similar structure
inside Re. A cost to carry with the tier 2e reading above.

## Verdict (for the user's decision)

The re-baselined layer supplies the diversity the mean cannot: on the
adopted exp82 mean, held out, the width sub-gate goes from 1 of 15 to 12 of
15 at fixed stellar mass and from 0 to 15 of 15 at fixed halo mass, every
offset passes, the amplitude statistics are exact, the outer annulus is
untouched, and — the choice this experiment was built around — a per-epoch
draw with the anatomy's cross-epoch correlation reproduces how much a
galaxy's size rank persists in TNG (the independent and persistent
controls miss it by 0.4 either way). The outer component the exp73 verdict
asked for is the one that matters; the compact one calibrates to zero.

Costs to carry: S3 at 0.012 (z ≤ 1) / 0.031 (all) against 0.010; R20 and
R50 at z ≤ 1 over-dispersed by 20 per cent at fixed M*; the decline
statistics unchanged in kind (33 per cent declining vs 42; decliner median
−0.12 vs −0.066) — v1's walls, which no exchangeable draw moves; **tier
2e's M(<10 kpc) at z = 0.4 worsens** (0.9 → 3.3 × the floor, W1 1.3 → 3.4)
and this PERSISTS with the compact draw off (Stage 3's table), so it is the
extended draw's and the amplitude's widening of the central-mass
distribution at low redshift, while the outer distributions improve
(M(50–100): 2.4–3.5 → 1.4–2.3). The layer is right for the outskirts and
over-wide in the centre at z = 0.4 — the same reading as R20/R50's 1.20.

Recommendation: adopt `gauss-2scale` as hongshao's v2 layer, with the
compact axis reported as calibrated to zero (i.e. two components), and log
the S3 residual as an open question with the two candidate levers above.
