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
`predict2` calls per evaluation instead of one). Results below when the fit
lands.
