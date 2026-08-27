# exp61 — single-epoch baselines and the z ≤ 1.0 scope

Branch `exp60-stochastic-layer` (this experiment runs alongside exp60; both
follow the user's 2026-08-26 direction). One script, `stage0_fits.py`; full
log `outputs/stage0_fits.log`. Seven fits of `gompertz_log-E2-S2` on the
standard sample under the standard mask: five independent single-epoch fits,
one z ≤ 1.0 fit, against the shared incumbent.

## The baseline table

Per-epoch score_A / score_F (lower is better; ~1 means as good as the
halo-mass regression benchmark refitted at that epoch):

| model | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| shared incumbent | 1.115 / 0.934 | 1.056 / 0.908 | 1.028 / 0.896 | 1.042 / 0.845 | 0.941 / 0.775 |
| single-epoch (each its own) | 1.109 / 0.888 | 1.050 / 0.890 | 1.015 / 0.891 | 1.011 / 0.827 | **0.835 / 0.744** |
| z ≤ 1.0 refit | 1.110 / 0.899 | 1.054 / 0.890 | 1.026 / 0.898 | 1.059 / 0.903 * | 1.011 / 0.906 * |

(* out of that fit's scope, evaluated for information only.) Single-epoch
loss gains over the incumbent's theta at that epoch: +4.5 / +2.3 / +1.8 /
+5.2 / **+15.9** percent; the z ≤ 1.0 refit gains +1.9% in scope.
Multi-start spreads 1e-9 to 1e-11, no parameter at a bound.

## The two answers

**1. The z ≤ 1.0 retreat is a REPORTING decision, not a refit — the shared
model is already at its class ceiling there.** Refitting to z ≤ 1.0 moves the
in-scope scores by under 0.01 while costing z=2 seven points of score_A. The
five-epoch fit was NOT compromising the low-redshift epochs to serve the high
ones; hongshao's z ≤ 1.0 performance is what the model class delivers, and
can be claimed as such with the incumbent unchanged.

**2. z=2 is the one epoch where the shared law's FORM binds.** Freed from
serving the other epochs, the same seven parameters reach score_A 0.835 —
better than the per-epoch benchmark — recovering 15.9% of the loss. So
roughly that much of the z=2 gap is not missing information but a functional
form that cannot bend enough: in the drift table every efficiency parameter
walks monotonically with epoch (`a0` −2.57 → −4.30, `a_M` −0.64 → −1.48,
`a_z` +0.57 → +1.68, `a_Mz` +0.42 → +0.86 from z=0.4 to z=2), and a drifting
`a_Mz` is curvature in the mass×time plane that the single cross term cannot
express. The size parameters drift oppositely (`log_f0` −1.09 → −0.72, `b`
−0.19 → −0.97) with the deposit shape `c` stable at 0.66–0.80.

**Caveats, stated**: the drift is INDICATIVE — single-epoch fits can slide
along the parameter degeneracies exp54 Stage 3.9 measured (the four
efficiency parameters absorb each other 12–18×), and no profile likelihood
was run. What is solid is the per-epoch score ceiling, which is a property of
the fits' losses, not of individual parameters.

## What follows from it

- The scope statement for hongshao v1: **validated at z ≤ 1.0 with the
  incumbent as adopted** (plus the exp60 layer if it graduates); z ≥ 1.5
  reported with the known gap.
- The z=2 recoverable component (~16% of per-epoch loss) is a MEAN-MODEL
  form question — a candidate for a mass×time-curvature term (`m·x²`), which
  no E-law variant has tried (E6/E7/E8 bend time only, at fixed mass slope).
  Parked as an open question: the user's stated read is that z ≥ 1.5 likely
  needs additional data (the real merger tree, 3-D information), and the
  stochastic layer leads the current work.

## Stage 1 — the M2 mass x time-curvature fit (2026-08-27): a clean NULL

The one-parameter curvature term the drift table pointed at
(`a_mz2 (logmh − 13.5) ln(1+z)²` per deposit, exp59's exact-nesting
machinery): all four starts — including one launched from the wrong sign and
one from strong curvature — converge to the same interior optimum,
`a_mz2 = −0.056`, loss −0.04%, shape error 13.754%, and the B gates exactly
at the null's values (z=2 bias −0.047 against the null's −0.046). Two things
are established, and the pre-registered sign (`a_mz2 > 0`) is FALSIFIED as a
loss-optimum prediction:

1. **The shared loss will not spend curvature on the z=2 bias** — the same
   0.6%-of-loss blindness exp58 P-B measured, now demonstrated on the exact
   term built to exploit it. The fit is well-identified; the null is real.
2. Together with exp58 P-C (a fit-sample median fix OVERSHOOTS the full QA
   population by +4% at z=2, because the mh-complete and full samples carry
   different biases), the z=2-median question is now fully mapped: reachable
   per epoch (the single-epoch fit), not adoptable by the shared loss (this
   fit), and ambiguous across samples (P-C). It stays parked; any future
   attempt must first decide WHICH z=2 population the model owes its median
   to, which is a scope question for the user, not a fitting question.

## Stage 2 — the figures (2026-08-27)

Two scripts, and neither fits anything. `stage2_figure_data.py` re-evaluates
the seven frozen thetas — cheap, ~0.12 s per evaluation — to recover three
things the tables never saved, and refuses to write unless every one of them
reproduces the fits' own recorded numbers (it does, to 0.00e+00 on the
losses and both scores, and to the last printed digit on the Stage 1 gate
values). `stage2_figures.py` then draws four figures from the npz files
alone. Both write to `outputs/` and `figures/`, which are gitignored and
regenerable:

```
HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 \
  PYTHONPATH=. uv run python -u experiments/exp61_epoch_baselines/stage2_figure_data.py
PYTHONPATH=. uv run python -u experiments/exp61_epoch_baselines/stage2_figures.py
```

### `figures/exp61_baselines` — the baseline table, in four currencies

Panels (a) and (b) are the baseline table drawn: score_A and score_F against
redshift for the shared incumbent, for each single-epoch fit at its own
epoch, and for the z ≤ 1.0 refit (dashed and hollow outside its scope).
score_F carries a second axis in per cent, because score_F × 15.82 is the
model's typical percentage error on the curve of growth — the incumbent's
0.934 at z=0.4 is a 14.8 per cent error, its 0.775 at z=2 a 12.3 per cent one.

Panels (c) and (d) are new here, and they carry the result the scores cannot
show. A score divides by the per-epoch benchmark scatter and so cannot tell a
systematic offset from noise (exp58 P-B); the frozen exp59 B-gate quantities
can. Measured for the single-epoch fits for the first time:

| at its own epoch | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| **B1**, median log10(model/measured) at 103 kpc [dex] — shared | −0.0087 | +0.0065 | +0.0110 | +0.0016 | **−0.0461** |
| the same, single-epoch | −0.0046 | −0.0003 | −0.0015 | −0.0010 | **+0.0016** |
| **B3**, median error of M\*(<10.25 kpc) [%] — shared | −2.2 | +0.5 | +2.3 | −1.5 | **−12.7** |
| the same, single-epoch | −1.4 | −2.0 | −1.3 | −0.4 | **+0.3** |

What that means physically: at z=2 the shared law reproduces only about 90
per cent of the stellar mass inside 103 kpc (0.046 dex light), and only about
87 per cent of the mass inside 10.25 kpc, in the median galaxy. Given the
other four epochs, seven parameters cannot do better. Given z=2 alone, the
SAME seven parameters put both back to zero. So neither miss is a missing
ingredient — both are the cross-epoch compromise, which is exactly what the
score-only reading of the baseline table (+15.9 per cent of loss recoverable
at z=2) already implied, now stated in the currency the user reads off the QA
figures.

### `figures/exp61_transfer` — the compromise is a CLOSE one

Every parameter set evaluated at every epoch. Each cell is the share of that
set's loss at that epoch which a fit devoted to the epoch alone would remove,
`100 × (1 − loss_j(theta_j*) / loss_j(theta_i))`, so the diagonal is 0 by
construction and the top row is the Stage 0 log's own +4.5 / +2.3 / +1.8 /
+5.2 / +15.9 per cent.

The row is the compromise cost. The rest of the matrix is the scale to read
it against, and it is the figure's point: carrying one epoch's own optimum to
another epoch costs far more than the shared law costs anywhere — the z=2 fit
gives up 82.9 per cent of its loss at z=0.4, the z=1.5 fit 64.2 per cent, and
even the neighbouring z=1.0 fit gives up 15.7 per cent there. The shared
seven parameters are not a poor compromise between five incompatible epochs;
they sit close to all five optima at once, and the one place that closeness
runs out is z=2. The z ≤ 1.0 refit's row says the same from the other side:
1.0 / 0.4 / 1.9 per cent in scope (it is already at the ceiling there, which
is result 1 above), 32.1 per cent at z=2 out of scope.

### `figures/exp61_drift` — the drift table, with its caveat attached

One panel per parameter, five coloured points per panel (one per epoch, the
project's epoch colours), the shared value as a dashed line and the z ≤ 1.0
value as a bar spanning the epochs it was fitted to. Each panel's title
carries its z=0.4 → z=2.0 range, and each is labelled with what the parameter
physically does. The four efficiency parameters walk monotonically in one
direction, the two size parameters the other way, and the deposit shape `c`
barely moves (0.66 → 0.80). No fitted value is at a bound — the closest is
`c`, still 0.46 from its nearest limit.

The eighth cell holds the caveat rather than a plot, because this table is
the one that most invites over-reading: single-epoch fits can slide along the
degeneracies exp54 Stage 3.9 measured (the four efficiency parameters absorb
one another by factors of 12 to 18), and no profile likelihood was run. Read
the direction, not the size of the step.

### `figures/exp61_m2_null` — the curvature term, drawn

Panel (a) shows each of the four starts as a hollow marker on the
no-movement diagonal with a line down to where it converged: all four land on
`a_mz2 = −0.056`, inside the bounds, on the falsified side of the
pre-registered `a_mz2 > 0`, with losses agreeing to the sixth decimal.
Panels (b) and (c) put the fitted term's B1 and B3 against the null's: the
two curves never separate by more than 0.0012 dex or 0.8 of a percentage
point at any epoch. One extra parameter, +0.04 per cent of loss, and nothing
the eye can see — which is the 0.6-per-cent-of-loss blindness exp58 P-B
measured, demonstrated on the exact term built to exploit it.
