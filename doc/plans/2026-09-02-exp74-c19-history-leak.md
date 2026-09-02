# exp74 — C19: the model's past must not know the halo's future

**Branch** `exp74-c19-history-leak` from `master` `6bc2ecc`. **Question** C19
(`doc/open_questions.md`). **Started** 2026-09-02 evening, the user's direction:
wrap up exp73, then C19 as a new experiment, continued without intervention as
far as it can be taken.

## The problem, in one paragraph

The model reads each halo through the official DiffMAH curve: four numbers
fitted to the halo's WHOLE peak-mass history (from 2 Gyr to z = 0, anchored at
z = 0, `tng_data.DMAH_FIT_TMIN_GYR = 2.0`). The engine evaluates that curve at
nodes from z ~ 15 up to each epoch's anchor and deposits stars along it. So the
curve the model integrates "before z = 2" is a function of the halo's mass at
z = 0 — and before 2.15 Gyr it is not even interpolated, it is the fit's
extrapolation from later, mostly future, data. exp71 measured the consequence:
future growth at fixed M200c(z_k) is predicted with R² 0.50–0.69 from the
DiffMAH curve read before z_k, 0.79 from `logmp` alone, and 0.00 from the
measured pre-z_k history. The model's stellar mass then depends on future
growth at +0.125 dex per dex where TNG300's has +0.003, and that leaked
dependence is the channel through which the z = 0.4 progenitor selection
tilts the z = 2 halo-mass relation (C18).

## What exp74 does

**The only change is the halo history the engine is given.** Same model
(exp63's two-channel mean, `model2.predict2`), same sample rule, same objective
(exp63's four-term binned fixed-kpc loss — exp73 showed the objective is a
dial, so it is not touched here), same starts.

### Stage 0 — the pre-epoch history, and the leak at frozen θ (no fit)

`history.py` builds, for each epoch k, a `HaloCurve` per galaxy fitted ONLY
to the catalog's peak-mass history at snapshots ≤ the epoch's anchor
(`hs["M200c"]`, running maximum, `GroupFlag == 1`, t ≥ 1.0 Gyr — the official
2 Gyr cut would leave one point before z = 2), **anchored at that epoch**
(`HaloCurve.logt0 = log10 t_k`, a new field with the official default) with
`logmp` FIXED at the measured anchor mass and `logtc`, `early`, `late` fitted
(bounded least squares, as `hongshao.diffmah.fit_mah`). Galaxies with too few
points get a lower-tier fit (fewer free shape parameters), never dropped:
dropping on history coverage would select on the thing under test (exp71's
`MIN_COVER` reasoning). The fit tier is recorded per galaxy-epoch and counted.

Gates, every one run before a model is evaluated:

- **G1 (reproduction)**: at z = 0.4 (8 points, 1.18–9.39 Gyr) the pre-epoch
  curve reproduces the official curve at the shared snapshots to within the
  official fit's own rms on the same points. If not, the anchoring or the
  peak-mass treatment differs and must be understood first.
- **G2 (no future, THE gate)**: exp71's cross-validated R² of future growth
  at fixed M200c(z_k) from the curve read at the pre-anchor snapshots
  (`channel_origin.cv_r2` with the mass stripped from both sides): the
  pre-epoch curve must score ≈ 0.00, like the measured history; the official
  scores 0.50–0.69. A pre-epoch curve that still predicts the future leaks
  through the fit and is disqualified.
- **G3 (reach)**: the fraction of the official model's deposited stellar mass
  laid down before the first usable snapshot (1.18 Gyr) and before the
  official fit's own 2 Gyr cut, per epoch — how much of the model's input is
  extrapolation rather than history. Reported, not gated.

`stage0_frozen.py` then predicts every epoch with exp63's joint θ FROZEN, once
on the official curves and once on the pre-epoch curves, and measures what the
input change alone does:

- the median profile change per halo-mass tercile and epoch (bins figure);
- `dy/dG`, the residual's partial dependence on future growth at fixed
  catalog mass at 103 kpc (`selection.partial_growth`): official +0.125 → ?
  against the truth's +0.003;
- the z = 2 halo-mass tilt, fitting sample and mh-complete (exp71's
  `leakage.py` numbers were −0.122 / +0.061);
- the size gate offsets (the curve sets r200, which sets the deposit size).

**Decision rule for Stage 1.** If `dy/dG` falls by more than half toward the
truth at frozen θ and the profiles move by more than exp63's own bin-median
rms (1–4 per cent), the refit is justified and its comparison is
like-for-like. If the frozen-θ prediction barely moves, the leak is in the
fitted parameters' compensation, and the refit is still the test — but the
expectation is recorded first.

### Stage 1 — the refit on pre-epoch curves (~1.5–2 h, background)

exp63's `stage2_fit.py --joint --objective binned --sample sane` machinery
with the curve list swapped per epoch (the joint problem's `predict2` call
takes one curve list for all epochs; exp74 gives it five, one per epoch —
`JointProblem2` is subclassed, not edited). Four starts (nested, exp63,
default, jitter0), the nested start asserted at the null. The judge is the
standard battery plus exp71's diagnostics, official-curve fit against
pre-epoch-curve fit, and the size gate. Every number on the fitting sample and
the mh-complete subset.

### What would change the programme

If the pre-epoch fit removes the +0.125 dependence and the z = 2 tilt on the
fitting sample moves toward the mh-complete value, every high-z number in the
record was partly the leak, and C18's "report it, do not fit it" stands with a
cleaner model behind it. If the fit compensates and the residual leak returns,
the leak is a property of reading any smooth history, and the honest input is
the measured history — a second representation (`history.py` variant B: the
running-peak history interpolated in log t, power-law extrapolated before the
first usable snapshot), which Stage 0 also builds and gates so that Stage 1
can be run on it without a new design.

## Costs

Stage 0: curve fits 2397 × 5 bounded least squares (~minutes); predictions at
frozen θ ~1 min. Stage 1: ~25 min per start at 0.45 s per evaluation.
