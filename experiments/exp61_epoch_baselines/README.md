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

**Read that last paragraph with the Stage 3 qualification below.** B1 and B3,
like every fitted loss in this experiment, are defined on the mh-complete +
sane FIT MASK, which at z=2 keeps 839 of the 2397 galaxies. On the whole
clean population the per-epoch fits do NOT put the z=2 bias back to zero —
they overshoot it, to +5.5 per cent inside 10.25 kpc and +12.4 per cent
inside 100 kpc. The statement "the per-epoch freedom removes the z=2 offset"
is true of the sample the fits were driven by and false of the population the
QA figures show.

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

## Stage 3 — the STANDARD QA battery, at every epoch (2026-08-27)

The user's request: the profiles themselves, not only the fitting metrics —
average curves of growth in halo- and stellar-mass bins, the stellar-mass
CDFs, the mass planes, the mass-size relations. `stage3_qa.py` runs
`hongshao.qa.evaluate`, the battery every promoted model in this programme
has been shown on, for three models on the same 2397 galaxies:

| label | what it is |
|---|---|
| `incumbent` | the shared seven-parameter law — hongshao v1's mean, the reference |
| `per-epoch-best` | at each epoch, the fit made at THAT epoch alone: the class ceiling, as profiles |
| `zle1.0-refit` | the z ≤ 1.0 retreat, shown at all five epochs |

Figures: `figures/qa/qa_*_exp61_{incumbent,per-epoch-best,zle1.0-refit}.*` —
eleven families per model (binned CoGs by halo mass and by stellar mass,
aperture and annulus mass tables, the observational planes, the mass-size
planes, the cross-epoch growth planes, the mass CDFs, the density profiles,
the best/worst gallery). Plus one figure the battery does not draw,
`figures/exp61_qa_compare`: all three models' median fractional CoG error
against radius, one panel per epoch. Full log `outputs/stage3_qa.log`.

**Two things to know before reading them.** First, `per-epoch-best` is not a
model anyone could adopt — it is five parameter sets stitched together by
epoch, so galaxy i's z=0.4 profile and its z=2.0 profile come from different
thetas. Every per-epoch tier is a genuine ceiling; tier 2c, the cross-epoch
growth planes, asks whether the same galaxy evolves coherently between two
epochs, and for a stitched composite that question has no owner. Second, the
standing battery caveat (exp57 P-C): the headline tiers cannot see a
redistribution of a few per cent of the mass inside 10 kpc.

### The result the battery adds: the class ceiling does not transfer

`stage3_qa.py` reports every bias twice, on the whole clean population and on
the mh-complete + sane fit mask that every loss and every B gate in this
experiment is defined on. The mask keeps 2397 / 1780 / 1435 / 1144 / **839**
galaxies at z = 0.4 / 0.7 / 1.0 / 1.5 / 2.0 — at z=2 it is a third of the
sample, and the third with the most massive haloes.

Median relative bias of the stellar mass inside 100 kpc [%]:

| model | sample | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|---|
| incumbent | all 2397 | −1.9 | +2.6 | +3.6 | +0.8 | −6.4 |
| incumbent | fit mask | −1.9 | +1.6 | +2.7 | +0.6 | −10.0 |
| per-epoch-best | all 2397 | −1.0 | +0.6 | +1.4 | +4.7 | **+12.4** |
| per-epoch-best | fit mask | −1.0 | −0.1 | −0.4 | −0.2 | **+0.6** |
| z ≤ 1.0 refit | all 2397 | −2.2 | +1.3 | +1.4 | −3.5 | −13.1 |
| z ≤ 1.0 refit | fit mask | −2.2 | +0.5 | +0.9 | −2.3 | −14.1 |

Inside 10.25 kpc the same split reads −12.5 (all) / −12.8 (mask) for the
incumbent and **+5.5 (all) / +0.3 (mask)** for the per-epoch ceiling.

So: **the per-epoch fits do not repair the z=2 profile, they re-aim it.** On
the sample they were driven by, they land on zero. On the wider population
they cross over to an excess of comparable size in the other direction — 12
percentage points of movement at 148 kpc, against 4 for the shared law over
the same sample change. The halo-mass-binned figure
(`qa_bins_exp61_per-epoch-best`) shows where it comes from: at z=2 the
ceiling runs +21 per cent at 2 kpc rising to +25 per cent at 148 kpc in the
LOWEST halo-mass tercile, against −5 to +4 per cent over the same radii in
the highest — and the lowest tercile is what the mh-complete mask removes at
that epoch.

This is exp58 P-C, which measured that a fit-sample median correction
overshoots the full QA population by +4 per cent at z=2, now visible as
profiles and larger than that estimate. It does not change either of exp61's
two answers — the z ≤ 1.0 ceiling result stands (all three models agree
closely there, and the sample split is under a point), and z=2 remains where
the shared law's form binds. It changes what may be CLAIMED from the z=2
recoverable component: 15.9 per cent of the loss is real and reachable on the
fit sample, and it is not a population-level fix. Any future z=2 attempt must
first answer open question C13 — which z=2 population the model owes its
median to — and this table is the sharpest statement of why that question is
not optional.

### What the battery says about the rest

- **z ≤ 1.0 is a genuinely flat region.** In `exp61_qa_compare` all three
  models lie on top of one another at z = 0.4, 0.7 and 1.0: inside ±5 per
  cent beyond about 8 kpc, and 7 to 13 per cent low in the inner few kpc.
  That inner deficit belongs to every epoch scope alike, so it is a property
  of the deposition-only class, not of the fit's redshift range — the same
  conclusion exp58's decliner work reached from a different direction.
- **The ceiling buys nothing in the battery's headline tiers.** M*(<100 kpc)
  scatter is unchanged to three decimals — 0.1163 / 0.1428 / 0.1464 / 0.1541
  / 0.1849 dex for the incumbent against 0.1158 / 0.1425 / 0.1464 / 0.1548 /
  0.1898 for the ceiling, so it is a hair better at z ≤ 0.7 and a hair worse
  at z ≥ 1.5. Tier 3 profile max|rel| beyond 5 kpc is 0.2765 / 0.2805 /
  0.2868 / 0.2965 / 0.3233 against 0.2775 / 0.2808 / 0.2907 / 0.3015 /
  0.3258 — the ceiling is marginally WORSE at every epoch. The 15.9 per cent
  it recovers at z=2 is a median offset, not scatter and not profile shape,
  exactly as the exp57 P-C caveat says the battery would show.
- **The retreat is expensive out of scope, and buys nothing in it.** The
  z ≤ 1.0 refit tracks the incumbent to within a point or two at its three
  epochs and is 13 to 25 per cent low at z = 1.5 and 2.0 — the profile
  version of Stage 0's answer 1.
