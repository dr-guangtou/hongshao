# exp84 — the stochastic layer re-baselined on the adopted mean, with an outer size component (S4)

Status: APPROVED 2026-09-22 (the user), in progress on branch
`exp84-layer-rebaseline` from master `17f2127`. Roadmap:
`doc/plans/2026-09-12-roadmap-evidence-rethink.md` §3 S4 and §5. The mean
under the layer is THE ADOPTED MEAN (exp82: `exp74/rebaseline.adopted_mean()`,
tau_d = 0.15 held + q_e = 0.152 on exp63's twelve; engine
`exp80/size_law.py::predict_law`) on the MEASURED halo history
(`exp74/measured.build_input(recs, "measured")`). The mean is not refitted.

## 1. Why, and what the v1 layer cannot be carried over

The width sub-gate (0 of 15 for every mean model; exp73, C16) is the largest
remaining gate block, and the error budget (exp81 E1/E2) says no halo-only
mean can supply the 0.17 dex of central scatter. The v1 layer (exp60, adopted
2026-08-27) supplied part of it: R50 width 1.14 → 0.73 across redshift, R20
OVER by 2.3×, R80 UNDER by half — the outer component is missing
(`stochastic-layer-size-gate-verdict`). v1 cannot be re-scored on the adopted
mean: it runs on exp57's X3 problem and the official-curve step engine
(`engine.build_curves` asserts the official DiffMAH curve), and its centre
component, the per-galaxy X3 core remap `A_i`, does not exist in the adopted
engine. This is the half-day rebuild `rebaseline.py`'s docstring records as
owed since exp74.

## 2. Decisions (the user, 2026-09-22)

1. **Components.** Three per-galaxy draws: the cross-epoch-correlated
   AMPLITUDE (v1's machinery), the COMPACT deposit size `log_f_c` and the
   EXTENDED deposit size `log_f_e` (the "second size draw" = the outer
   component). X3 is dropped; the adopted mean's delay + q_e carry the mean
   centre. The S1 decline targets (declining fraction 41.8 per cent, the
   decliner median) are REPORTED, not gated — v1 measured them as structurally
   stuck for an exchangeable draw, and they are partly projection geometry.
2. **Persistence.** The size deviations are drawn per galaxy AND PER EPOCH
   from a correlated Gaussian whose cross-epoch correlation is fitted (the
   anatomy measures it, the calibration half fixes it); they act at the
   observed epoch's evaluation. A persistent trait (one pair per galaxy) is
   the `correlation = 1` limit and is reported as a control.
3. Decided by the agent, open to change: the calibration sample is THE
   FITTING SAMPLE (`selection.fitting_sample_mask`, 2356; the mh-complete
   subset is an after-the-fact report), and the gates run at all five epochs
   (z ≥ 1.5 alongside, as the standing size gate does).

## 3. The predictor (`predictor.py`)

`build(smoke)` = `exp74/rebaseline.build(smoke)` (recs, data, the fitting
sample mask, measured curves, measured-mass bins) + `adopted_mean()`.
`Predictor.predict(size_dev=None, nodes=FULL_NODES)` calls
`size_law.predict_law` with ONE engine addition:

    size_dev = dict(c=(n, 5), e=(n, 5))      [dex]

added to log s_c and log s_e at the OBSERVED epoch's evaluation (the compact
kernel is evaluated per epoch only when a deviation is given; the extended
kernel already is, because q_e ≠ 0). `size_dev=None` nests bit for bit
(`selfcheck` asserts it, and asserts that a zero deviation array changes
nothing and a non-zero one changes something). No cube, no interpolation, no
per-draw model call beyond this one.

Ops before any batch: the smoke build MEASURES one `predict_law` call's wall
time and peak resident size at FIT and FULL nodes, with and without a
deviation (the per-epoch compact kernel is 5× the basis work); the stage
budgets below are set from those numbers, not estimated.

## 4. Stages

### Stage 0 — the targets, frozen on the adopted mean (`stage0_targets.py`)

- S5: the adopted mean's amplitude residual log10(model/truth) at 100 kpc on
  the fitting sample — per-epoch half 16–84 width and the 5×5 cross-epoch
  correlation (v1: 0.117–0.142 dex, 0.67–0.80 nearest-epoch).
- S3 reference: the mean's median log CoG per epoch.
- THE PERSISTENCE DIAGNOSTIC: the truth's cross-epoch rank correlation of the
  size residual at fixed stellar mass (dlog R50 about the running median vs
  log M*(<148), epoch pairs) — what decision 2's correlation is tested against.
- For the record: the S1 decline numbers on this sample; the mean's own
  tier 2d (R20/R50/R80 offset + width at fixed M* and Mh) and tier 2e tables
  — the NULL row of every later table.

### Stage 1 — the anatomy (`stage1_anatomy.py`)

exp60 Stage 2's shared-sweep trick, per galaxy AND per epoch: a shared δ on
`log_f_c` over 15 values gives every galaxy's per-epoch loss curve
(`gal_losses` without the epoch mean); the parabolic minimum is δ_c (n, 5).
The same for `log_f_e`; then a 7×7 joint (δ_c, δ_e) grid, because the two
axes trade against each other, giving the joint per-galaxy-epoch minimum.
The 1-D sweep of the other twelve parameters (each at 15 values) is the
control that these two are the individuality axes.

Deliverables, all printed and saved (`outputs/stage1_anatomy.npz`):
the per-epoch marginals of δ_c and δ_e (percentiles), the 5×5 cross-epoch
correlation of each, the c×e cross-correlation per epoch, the fraction at
the grid edge (the anatomy's own ceiling, stated before the draw is built),
the Spearman correlation with halo features (log Mh, f_form, the early-mass
fraction; what would belong to conditioning), and the anatomy's persistence
next to the truth's from Stage 0.

### Stage 2 — compose and judge, held out (`stage2_judge.py`)

Protocol (exp41/exp60): pools and fits from one random half, scored on the
other, both swaps, 8 realizations, seeds fixed.

The draw, per galaxy: a (5, 2) deviation vector from a Gaussian with the
per-epoch σ and the 10×10 correlation (5 epochs × 2 axes) fitted on the
calibration half's anatomy — the portable form; the row-resampling of whole
calibration-half anatomy records is the nonparametric control. Amplitude:
v1's correlated Gaussian, `sig_add` per epoch calibrated in quadrature so
the TOTAL drawn amplitude deviation meets the Stage 0 target.

Variants: (a) the raw anatomy widths; (b) one size scale calibrated on the
R50 WIDTH sub-gate held out (R20 and R80 are then the tests of the axes,
not calibrations); (c) correlation = identity (independent epochs) and
correlation = 1 (the persistent trait) — the persistence controls; (d) the
permuted control (deviations assigned to shuffled galaxies).

Gates, every one on the DRAWS (mean over realizations, scatter across them):

| gate | quantity | pass |
|---|---|---|
| 2d offset | median dlog R20 / R50 / R80 at fixed M* and at fixed Mh, per epoch | abs ≤ 0.05 dex (`qa.SIZE_GATE_OFFSET`), counted per cell |
| 2d width | the model's size scatter at fixed mass over the truth's | within 20 per cent (`qa.SIZE_GATE_WIDTH`), counted SEPARATELY |
| S3 | drawn-population median CoG vs the mean | within 0.010 dex at every radius |
| S5 | amplitude scatter and nearest-epoch correlation | within 10 per cent / 0.05 |
| S4 | the M(50–148) annulus median on the drawn population | within the mean's band |
| persistence | cross-epoch rank correlation of the drawn size residual vs the truth's | reported; a variant that misses it by more than 0.15 is named |
| tier 2e | KS / W1 over the floor for `qa.CDF_KEYS`, on the draws | reported next to the mean's; the mean is the null |
| S1 | declining fraction, medians | reported only |

Rows of every table: the adopted mean alone (null), v1 on its own mean
(exp73's saved `size_gate_layer.npz`, the like-for-like record), each v2
variant. Never quoted: paired per-galaxy errors of draws.

### Stage 3 — package (`stage3_adopt.py`)

`outputs/hongshao_v2_layer.npz` (σ per epoch and axis, the 10×10
correlation, `sig_add`, the scale, seeds, the sample rows) and
`draw_cogs(predictor, rng)`; the standard battery with the drawn populations
overlaid (`figures/qa/*exp84_v2_layer*`); `README.md` with the verdict
table. Adoption is the user's call.

## 5. Tier 2e reads the draws

`hongshao/qa.py` gains `evaluate_draws(draws, data, R, anchor_z, ...)`: runs
`evaluate` (figures off) on each drawn population and returns the tier 2d
cells and the tier 2e KS/W1 ratios averaged over draws with their scatter,
so a layer is scored by the same code path as a mean (C16 closed as a
procedure). `qa.demo` gets the identity and permutation checks for it.

## 6. What this experiment does NOT do

No refit of the mean; no per-galaxy freedom in the delivered model (the
layer is a distribution); no conditioning of the draws on halo features
(Stage 1 measures what would belong there and reports it); no X3 remap; no
decline gate.

## 7. Ops

Branch `exp84-layer-rebaseline`; scripts in
`experiments/exp84_layer_rebaseline/`; every long stage launched through
`experiments/exp82_delay_expansion/launch.py` (own session, caffeinate),
never more than TWO heavy jobs at once, judged by waiting on the npz file,
not on a log line. Smoke (`--smoke`) before every full run.
