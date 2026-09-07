# exp74 — C19: the model's past must not know the halo's future

Branch `exp74-c19-history-leak` from `master` `6bc2ecc`, 2026-09-02 evening.
Plan: `doc/plans/2026-09-02-exp74-c19-history-leak.md`. Question: C19 in
`doc/open_questions.md`.

## The problem in plain language

The model grows each galaxy by depositing stars along its halo's growth curve.
That curve is not the measured history: it is the official DiffMAH fit, four
numbers fitted to the halo's whole peak-mass history from 2 Gyr to z = 0 and
anchored at z = 0. So the curve the engine integrates "before z = 2" is a
function of the halo's mass at z = 0, and before 2.15 Gyr it is not even
interpolated — it is the fit's extrapolation from later, mostly future, data.
exp71 measured the consequence: at fixed halo mass at z = 2, a halo's future
growth is predicted with R² 0.50–0.69 from the DiffMAH curve read before z = 2
and with 0.00 from the measured history. The model's stellar mass therefore
depends on future growth at +0.125 dex per dex where TNG300's depends at
+0.003, and that leaked dependence is the channel through which the z = 0.4
progenitor selection tilts the model's z = 2 halo-mass relation (C18).

**exp74 changes one thing: the halo history the engine is given.** Same model
(exp63's two-channel mean), same sample rule, same objective, same starts.

## Stage 0a — the pre-epoch history (`history.py`)

For each epoch, a DiffMAH curve per galaxy fitted ONLY to the catalog's
peak-mass history at snapshots up to that epoch's anchor (running maximum of
`M200c`, t ≥ 1.0 Gyr; the official cut of 2 Gyr would leave one point before
z = 2), anchored at the epoch with the anchor mass fixed at the measured value
and the three shape parameters fitted. Galaxies with too few points get a
lower-tier fit, never dropped (dropping on history coverage would select on
the thing under test). `HaloCurve` gained an anchor field (`logt0`, default
the official z = 0 anchor; the engine selftest passes unchanged).

| epoch | usable snapshots | tiers 3/2/1/0 | median rms [dex] |
|---|---|---|---|
| z=0.4 | 9 (1.18–9.39 Gyr) | 2397/0/0/0 | 0.079 |
| z=0.7 | 7 | 2397/0/0/0 | 0.077 |
| z=1.0 | 6 | 2394/3/0/0 | 0.072 |
| z=1.5 | 5 | 2366/29/2/0 | 0.060 |
| z=2.0 | 4 (1.18–3.28 Gyr) | 2211/156/28/2 | 0.038 |

**G1 (reproduction, z = 0.4)**: the pre-epoch and official curves differ by
0.062 dex rms (median) at the nine shared snapshots, less than the official
fit's own misfit of the peak data there (0.105 dex; the pre-epoch fit's is
0.079). The official fit anchors at z = 0 and has a 2 Gyr cut, so the two
earliest points are its extrapolation. **OK.**

**G2 (no future — THE gate)**: cross-validated R² of future growth at fixed
M200c(z_k) from the curve read at the three pre-anchor snapshots (exp71's
measurement, mass stripped from both sides):

| epoch | official | pre-epoch |
|---|---|---|
| z=0.7 | 0.501 | 0.000 |
| z=1.0 | 0.688 | −0.003 |
| z=1.5 | 0.656 | −0.003 |
| z=2.0 | 0.518 | −0.005 |

The pre-epoch curve knows nothing about the future, like the measured history.
**OK.**

One more number: the official DiffMAH curve **under-states the halo's mass at
the anchor** by 0.02 / 0.05 / 0.06 dex (median) at z = 1.0 / 1.5 / 2.0, with a
16–84 range reaching +0.18 dex — the fit to the whole history smooths over the
fast growers' early mass. The pre-epoch curve passes through the measured
value by construction.

## Stage 0b — the input change at frozen θ (`stage0_frozen.py`)

exp63's joint θ held fixed; every epoch predicted on the official curves and
on the pre-epoch curves (`outputs/stage0_frozen.log`, figures
`figures/qa/*exp74_frozen_*`).

**G3 (reach).** Fraction of the model's deposited mass laid down before the
first usable snapshot (1.18 Gyr) and before the official fit's own 2 Gyr cut:
4 / 15 per cent at z = 0.4, rising to **14 / 43 per cent at z = 2**. Nearly
half of the model's z = 2 galaxy is built on a stretch of the curve the
official fit never saw data for.

**The leak is gone at frozen θ.** The residual's dependence on future growth
at fixed catalog mass at 103 kpc, dex per dex (the truth's own in the last
row):

| | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|
| residual, official curves | +0.173 | +0.145 | +0.126 | +0.120 |
| residual, pre-epoch curves | −0.014 | −0.045 | −0.027 | −0.003 |
| the truth's own dependence | −0.005 | +0.051 | +0.027 | +0.003 |

The model's own stellar mass goes from +0.123–0.196 per dex of future growth
to −0.000–0.006: it no longer knows. (The truth carries a small positive
dependence at z = 1.0–1.5, +0.03–0.05, that no model without the future can
follow; the official model overshot it four-fold.)

**The z = 2 halo-mass tilt.** Fitting sample | mh-complete, residual at
103 kpc against the catalog mass: official **−0.121 | +0.060** (exp71's C18
numbers reproduced) → pre-epoch **−0.034 | +0.093**. Three-quarters of the
fitting-sample tilt was the leak; what remains on the mh-complete subset is a
real mass dependence of the model at z = 2 that got slightly larger.

**The price at frozen θ: the model gets heavier.** The pre-epoch curves carry
more early mass (the anchor-mass difference above, and a steeper early
history), so the same θ deposits more stars: +4–6 per cent at z = 0.4 and
+11 per cent at z = 2 on the fitting sample, **+18 per cent at z = 2 on the
mh-complete subset**. In the halo-mass-binned figure the most massive third
is 15–18 per cent too heavy at z = 1–1.5; the amplitude-pinned (shape)
residuals stay within a few per cent outside 5 kpc, so this is an amplitude
change the efficiency law will re-absorb. Half-mass radii move the other way
at z ≤ 1 (−3 to −7 per cent) and up at z ≥ 1.5 (+6, +10 per cent): the curve
also sets r200 at the deposit times.

**Decision rule met**: the leak fell by more than half (to zero) and the
profiles moved by far more than exp63's bin-median rms. Stage 1 runs.

## Stage 1 — the refit on the pre-epoch curves (`stage1_refit.py`, `stage1_eval.py`)

exp63's joint problem subclassed with one curve list per epoch; the
objective's references are the nested incumbent on the OFFICIAL curves, so
every term is normalised exactly as exp63's was; same bounds; four starts
(nested, exp63's optimum, default, jitter0), run as four processes (five
`predict2` calls per evaluation instead of one). Five starts; the two far ones
(default, jitter0) stepped into the unbuildable-model penalty on their first
line search and railed — a far-start failure under this objective, so a NEAR
start (exp63's optimum moved by 5 per cent of each bound's range) was added.
The machine slept for four hours mid-fit; `caffeinate` for next time.

| start | loss | note |
|---|---|---|
| nested | 18.113 | stopped at the cap |
| **exp63** | **17.294** | converged, 1418 evaluations |
| exp63_near | 17.305 | at the cap, 0.06 per cent above |
| default, jitter0 | 125 | railed into the penalty |

**The refit is exp63 with one parameter moved: `a_z` 0.348 → 0.286**, the
redshift exponent of the deposition efficiency; every other parameter is
within 0.05 of exp63's. *The model's knowledge of the halo's future had been
stored as a steeper rise of the efficiency with redshift.*

### The 2 × 2 (`stage1_eval.py`, `outputs/stage1_eval.log`)

exp63's four-term loss, per epoch, references = the nested incumbent on the
official curves:

| θ \ curves | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 | total |
|---|---|---|---|---|---|---|
| exp63 on official | 3.468 | 3.005 | 2.958 | 2.945 | 2.978 | **15.353** |
| exp63 on pre-epoch (frozen) | 4.518 | 4.173 | 4.622 | 6.072 | 3.481 | 22.866 |
| refit on official | 4.783 | 3.368 | 3.042 | 4.497 | 4.269 | 19.959 |
| **refit on pre-epoch** | 3.834 | 3.361 | 3.195 | 3.493 | 3.400 | **17.284** |

**The honest input costs 12.6 per cent of exp63's loss**, most of it at
z = 0.4 (+11 per cent) and z ≥ 1.5 (+19, +14 per cent). That is the size of
the advantage the official curves gave the model by carrying the future: a
number every fit in the record before this one enjoyed.

### What the 12.6 per cent bought

| | exp63, official | refit, pre-epoch | the truth |
|---|---|---|---|
| residual dy/dG at 103 kpc, z=2 [dex/dex] | +0.120 | **−0.004** | +0.003 |
| same, z=0.7 / 1.0 / 1.5 | +0.173 / +0.145 / +0.126 | −0.011 / −0.046 / −0.028 | −0.005 / +0.051 / +0.027 |
| z=2 halo-mass tilt, fitting sample | −0.122 | **−0.046** | — |
| z=2 halo-mass tilt, mh-complete | +0.060 | +0.083 | — |
| M(<103 kpc) at z=2, mh-complete | −9.7 % | **−2.5 %** | — |
| M(<10 kpc) at z=2, mh-complete | −4.7 % | +0.2 % | — |
| M(<103 kpc) at z=2, fitting sample | −4.6 % | −2.1 % | — |
| R50 width ratio at fixed M*, z=2 | 0.27 | **0.40** | 1 |

- **The leak is gone after the refit too.** The model no longer knows the
  future at any epoch; the small negative values at z = 1.0–1.5 are the
  truth's own positive dependence (+0.03–0.05, assembly bias TNG300 really
  has) that a model without the future cannot follow.
- **Sixty per cent of the z = 2 fitting-sample tilt was the leak** (−0.122 →
  −0.046). The mh-complete tilt is real and slightly larger (+0.083): at z = 2
  the model still gives the least massive of the complete progenitors too
  little relative to the most massive.
- **The z = 2 massive progenitors are now right.** exp63's biggest documented
  z = 2 error — the mh-complete subset 9.7 per cent too light at 103 kpc, the
  population C13 asked about — is 2.5 per cent, and its centre is 0.2 per
  cent. The fitting-sample total at z = 2 improves from −4.6 to −2.1.
- **The model's galaxies became more diverse in size.** The R50 width ratio
  at fixed stellar mass rises at every epoch (0.63 → 0.69 at z = 0.4, 0.27 →
  0.40 at z = 2), because the pre-epoch curve passes through each halo's
  MEASURED mass at the epoch instead of DiffMAH's smoothed value. A third of
  C16's missing z = 2 diversity was the smoothing.

### What it cost

- **The mh-complete subset at z = 1.0–1.5 is now 5–9 per cent too heavy**
  (was +1.5–4). The honest curves carry more early mass for the massive
  progenitors, and one shared efficiency law cannot be right for them at
  z = 1.5 and at z = 2 both; the loss chose z = 2.
- **High-z sizes are bigger**: R50 +6.9 / +10.8 per cent at z = 1.5 / 2 (exp63
  +3.3 / +4.9); the size gate's offset passes 13 of 15 against 14 (R20 at
  z = 1.5 joins z = 2 in failing). z = 0.7 sizes are 5 per cent small.
- The centre is untouched: −12.8 per cent at 2 kpc at z = 2 against −11.1 —
  the central defect is a separate problem, as C18 said.

### Addendum — the same 2 × 2 with bins by the MEASURED mass (`rescore_bins.py`)

exp63's binned term uses halo-mass terciles by the official DiffMAH mass, the
variable the honest model no longer uses. Re-scored with terciles by the
measured M200c(z_k) (references rebuilt in the same bins):

| θ \ curves | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 | total |
|---|---|---|---|---|---|---|
| exp63 on official | 3.441 | 3.023 | 2.953 | 2.934 | 3.105 | **15.457** |
| refit on pre-epoch | 3.923 | 3.554 | 3.562 | 4.640 | **2.840** | **18.519** |

The gap is not a binning artefact: it is 12.6 per cent in exp63's bins and
19.8 per cent in the measured-mass bins (the refit was not optimised in
these). In the measured-mass bins the honest model is the better one at
z = 2 (2.84 vs 3.11) and pays at z = 1.0–1.5, where its massive progenitors
are too heavy. The cost of the honest input is 13–20 per cent of the loss,
concentrated at z = 0.4 (the z = 0 anchor) and z = 1.0–1.5.

## Variant B — the measured history as the input (`measured.py`, 2026-09-04, the user's direction)

The user's rule, stated 2026-09-04: for hongshao's application the halo's
full history from the simulation — DiffMAH parameters, peak and final halo
mass included — is legitimate input; what the model must never see is the
galaxy's own stellar mass. What C19 found is therefore not a cheat but a
mis-description: the official DiffMAH form lets the final mass rewrite the
early history, and the model's stellar mass then depends on the halo's future
four to forty times more than TNG300's does. Variant B keeps the whole history
as input and lets the deposition read it as it was: the running-peak M200c at
the sixteen catalog snapshots, a monotone cubic (PCHIP) between them in
(log t, log M) — each segment shaped by its own two points and the slopes at
them, a later snapshot steadying the curve just before it and no further — and
a power law continuing the first segment before 1.18 Gyr. One curve per
galaxy serves every epoch; the integral is truncated at the epoch as always,
so the future is present in the input and never consulted. The engine reads
a curve's own table functions when it carries them (`log_mah_table`,
`dm_dlnt_table`; `HaloCurve` unchanged, selftest passes).

**Gates.** The curve passes through the measured peak masses exactly (the
official curve's rms there is 0.105 dex). The no-future test at the
pre-anchor snapshots AND at the midpoints between them, where the PCHIP slopes
use neighbours: R² −0.003 / −0.001 / −0.005 / +0.007 at z = 0.7…2 (official
0.50 / 0.72 / 0.72 / 0.61 at the midpoints). Every galaxy has ≥ 6 usable
points; the early power-law slope is 3.1 (16–84: 1.2–5.7). At the engine's
nodes before each anchor the measured curve differs from the official one by
0.25–0.33 dex (median |Δ log M|) and from the pre-epoch fit by 0.23–0.26 —
most of it in the extrapolated stretch before 1.18 Gyr, which carries 4 (z =
0.4) to 14 (z = 2) per cent of the deposited mass.

**At frozen θ** (`stage0_frozen.py --curves measured`) the measured curves
reproduce the pre-epoch result to within a point: mass +4.4 / +3.5 / +6.6 /
+10.2 / +11.5 per cent at 103 kpc across the epochs (pre-epoch +4.3 … +11.0);
the residual's future dependence −0.014 / −0.045 / −0.025 / −0.005 (pre-epoch
−0.014 / −0.045 / −0.027 / −0.003); the z = 2 tilt −0.032 | +0.095 (pre-epoch
−0.034 | +0.093). **The two honest inputs agree with each other; the
functional form was not what mattered, the pinning to the future was.** The
refit with the measured curves is Stage 1's third column.

### Variant B, the refit (`stage1_refit.py --curves measured`, `stage1_eval.py`)

Four starts, one process each (one `predict2` call per evaluation, as exp63's
problem): exp63's optimum 17.014, the pre-epoch refit's optimum 17.026, the
near start 17.055, the nested incumbent 17.121 — **a single basin**. The
parameters that moved from exp63 by more than 0.05: `b_c` −0.80 → −0.88, `b_e`
−1.19 → −1.03, `c_e` 0.86 → 0.80 (the size-time exponents, not `a_z` this
time — the same absorbed future, taken up by a different pair of dials).

| θ \ curves | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 | total |
|---|---|---|---|---|---|---|
| exp63 on official | 3.468 | 3.005 | 2.958 | 2.945 | 2.978 | **15.353** |
| refit on pre-epoch | 3.834 | 3.361 | 3.195 | 3.493 | 3.400 | 17.284 |
| **refit on measured** | 3.799 | 3.296 | 3.159 | 3.483 | 3.274 | **17.010** |
| exp63 on measured (frozen) | 4.705 | 4.432 | 4.977 | 6.605 | 3.588 | 24.308 |

**The measured history costs 10.8 per cent of exp63's loss, the pre-epoch
DiffMAH 12.6.** Two points of the pre-epoch cost were its functional form;
the rest is the price of not reading the future.

| | exp63, official | refit, pre-epoch | refit, measured | truth |
|---|---|---|---|---|
| residual dy/dG at z=2 | +0.120 | −0.004 | −0.006 | +0.003 |
| z=2 tilt, fitting \| mh-complete | −0.122 \| +0.060 | −0.046 \| +0.083 | −0.032 \| +0.094 | |
| M(<103) z=2, mh-complete | −9.7 % | −2.5 % | **−1.1 %** | |
| M(<10) z=2, mh-complete | −4.7 % | +0.2 % | −0.3 % | |
| M(<103) z=1.5, mh-complete | −0.4 % | +7.5 % | +8.6 % | |
| R50, z=1.5 / 2 | +3.3 / +4.9 % | +6.9 / +10.8 % | +9.0 / +13.1 % | |
| size gate OFFSET passes | 14/15 | 13/15 | 12/15 | |
| R50 width ratio, z=0.4 / z=2 | 0.63 / 0.27 | 0.69 / 0.40 | 0.66 / 0.35 | 1 |

**The two honest inputs give the same model** to within a point or two
everywhere: the leak is zero for both, the z = 2 massive progenitors are
right for both (the measured history best, −1.1 per cent), both pay at
z = 1.0–1.5 on the mh-complete subset (+5–9 per cent) and in high-z sizes,
where the measured history pays a little more (R50 +13 per cent at z = 2, and
its R50 offset now fails the gate there at +0.054 dex). The centre is
identical in all three.

## The verdict

**C19 is confirmed at the level of the model, and its origin is now exact.**
The official DiffMAH curve — one four-number fit to the whole history,
anchored at z = 0 — lets the halo's final mass rewrite its early history. The
model read that and made its stellar mass depend on the halo's future growth
at +0.12–0.17 dex per dex where TNG300's depends at −0.01 to +0.05. Under the
user's rule (2026-09-04: the halo's full history, final mass included, is
legitimate input; the galaxy's own stellar mass never is) this is not a cheat
but a wrong conditional distribution: at fixed current halo mass the model
sorted galaxies by a variable the simulation does not sort on, four to forty
times too strongly. It showed up as 60–75 per cent of the z = 2 halo-mass tilt
on the fitting sample, as the massive z = 2 progenitors 10 per cent too light,
and as an 11–13 per cent advantage in the loss.

**Two honest inputs, built differently, give the same model.** A DiffMAH
form fitted only to the past (variant A) and the measured history read
directly (variant B) agree with each other to a point or two on every
quantity: dependence on the future zero, z = 2 massive progenitors within
1–2.5 per cent, z = 1.0–1.5 massive progenitors 5–9 per cent heavy, high-z
sizes 7–13 per cent big, loss 11–13 per cent worse than exp63's. What
mattered was not the functional form but the pinning of the past to the
future.

**Recommendation: adopt the measured history (variant B) as hongshao's
standard input.** It uses everything the simulation provides — the full MAH,
the final mass available to any part of the model — reads it as it was, and
needs no fit to build. It is the cheaper of the two honest inputs (one
`predict2` call per evaluation; the pre-epoch DiffMAH needs five), the better
one on the loss (17.01 vs 17.28), and the best of all three on the z = 2
massive progenitors. Its known costs — the z = 1.0–1.5 massive progenitors
and the high-z half-mass radii — are where the next modelling step should
look, with the future-dependence test kept as a standing QA gate: at fixed
current halo mass, the model's stellar mass must depend on future growth no
more than TNG300's does. The incumbent and the v1 stochastic layer were built
on the official curves and inherit the leak; they should be re-baselined on
the measured history once the input is adopted.

**What is not changed by any input**: the centre (−11 to −13 per cent at 2 kpc
at z = 2, −9 at z = 1.5), the width collapse (every mean model), and the
z = 1.0–1.5 real positive dependence on the future (+0.03–0.05) that a causal
deposition cannot follow.
