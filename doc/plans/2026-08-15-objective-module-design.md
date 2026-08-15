# Design — the objective module (`hongshao/objective.py`)

**Date:** 2026-08-15 · **Branch:** `objective-module` · **Type:** refactor, not
an experiment.

## Why

There is currently **no loss/objective code in `hongshao/` at all**. The
production objective is hardcoded in six lines inside
`experiments/exp38_deposit_rethink/stage2_multiepoch.py::gal_loss`, and a
flexible version — five selectable axes, twenty configurations measured — was
built for exp48 in `experiments/exp48_objective_profile/objective.py`. A third
copy of the scoring half lives in `exp48/shootout.py::score_cogs`, written
deliberately as a duplicate because editing the first would have corrupted a
multi-hour job mid-run.

exp48 established that the objective is the single most effective lever on the
model's known defect (it under-fills compact galaxies' centres at every epoch):
comparing surface-density profiles with a log residual moved the mean plane
energy relative to the truth's own split-half floor from 2.72 to 1.95, held-out
2.71 to 1.96. The user's decision was to **adopt nothing**, because that
objective steepens the redshift trend the wrong way. But the finding stands, and
the next time anyone wants to vary the objective they should not have to edit a
loss function by hand.

This refactor promotes the flexible objective into the library, folds the
duplicate into it, and **keeps current behaviour as the default**. The adopted
model and the adopted theta do not change.

## Scope, measured

Adding the library modules is purely additive. The only existing file edited is
`exp38/stage2_multiepoch.py`.

- `s2.gal_loss` is called from **six other experiments at about twenty sites**:
  `exp40/tests.py`, `exp41/stage0_anatomy.py`, `exp41/stage1_deviation.py`,
  `exp42/compare.py`, `exp43/ladder.py`, `exp44/decorrelation.py`, and
  `exp38/stage3_inner.py`. Several produced adopted numbers (exp40 chose the
  adopted z <= 1.5 fit scope; exp41 produced the adopted stochastic layer), so
  its behaviour must not change. Its signature is preserved and an equivalence
  check gates the port.
- Twelve experiments define their **own local** loss variants for different
  model families (`gal_loss_core`, `gal_loss_2ch`, `gal_loss_inner`,
  `gal_loss_form`, ... in exp32/33/35/36/38-stage1/38-stage3/39/44/47). These
  are not copies of the production objective and are untouched.
- Nothing imports `hongshao/fitting.py` (it does not exist yet). The two private
  simplex helpers — `exp47/core_shape.py::_simplex` and
  `exp48/objective.py::_simplex12` — stay where they are.

## Decisions

| decision | choice | why |
|---|---|---|
| boundary | **pure scoring on arrays** | a loss must not know about kernels, thetas, galaxy dicts, or multiprocessing; it can then score any model, including ones that do not exist yet |
| API form | **frozen dataclass** | mirrors `forward.py::Deform`, where the bare `Deform()` is the frozen baseline; validates axis names once instead of on every one of the tens of thousands of calls Nelder-Mead makes; gives the canonical cache-key name a home |
| failure | **NaN, counted** | the `4.0` sentinel is a fit-steering signal, not a measurement; a reported population mean silently containing 4.0s cannot be interpreted |
| simplex helper | **separate file** | how a loss is minimized is not the loss's business |
| epoch axis | **added, default unchanged** | see below |
| equivalence check | **one-off script in the branch**, not kept | it gates the port once; its output goes in the commit message |

## The module

```python
@dataclass(frozen=True)
class Objective:
    quantity: str = "cog"        # cumulative mass | surface density
    residual: str = "frac"       # fractional | absolute | log
    radial:   str = "rms"        # rms | mean-abs | worst-radius
    weight:   str = "uniform"    # uniform | per-dex | inner
    galaxy:   str = "mean"       # mean | median | worst-20%
    epoch:    str = "mean"       # mean | explicit weights
```

Bare `Objective()` reproduces the current production objective exactly.
`Objective(quantity="density", residual="log")` is exp48's winner.

Methods: `per_galaxy(model_cogs, data_cogs, radii)` -> (n_galaxies,), NaN where
a galaxy is unusable; `__call__(...)` -> the population number;
`name` -> canonical short string for cache keys; `to_dict` / `from_dict` for the
existing `.npz` and JSON configs.

Inputs are plain arrays of shape (n_galaxies, n_epochs, n_radii) for both model
and truth, plus the radius grid in kpc. Imports only numpy and
`density_from_cog` from `hongshao/profile_emulator.py`.

## The scoring math

1. **Quantity.** `cog` compares the cumulative profiles M(<R) on the measured
   24-radius grid. `density` differences both sides to surface density Sigma(R),
   landing on 23 midpoint radii (geometric means of adjacent grid radii). The
   radius grid used downstream therefore depends on the quantity, so the radial
   weights cannot be built once and shared.
2. **Residual**, per radius. `frac` = (model - truth)/truth. `log` =
   log10 model - log10 truth. `abs` = the linear difference divided by that
   galaxy's own normalization — its measured M(<148 kpc) for the cumulative
   profile, its peak surface density for the density profile.
3. **Radial aggregation.** `rms` = sqrt(sum w r^2); `mean-abs` = sum w |r|;
   `worst-radius` = max|r|, which ignores the weights (documented, not silent).
   Weights sum to one so losses stay comparable across weightings: `uniform`
   = 1/n; `per-dex` proportional to the spacing in log radius (equal
   contribution per decade, not per grid point); `inner` proportional to 1/R.
4. **Epoch aggregation.** Arithmetic mean, or an explicit weight vector.
5. **Galaxy aggregation.** `mean`, `median`, or `worst-20%` (the mean of the
   worst fifth), over the finite entries, with the failure count reported.

### Why density is differenced rather than evaluated analytically

The deposit primitive **is** a surface density — Moffat,
`Sigma(R) ~ (1 + (R/rc)^2)^-gamma` — and `basis_mof` evaluates its enclosed-mass
form `1 - (1 + (R/rc)^2)^(1-gamma)` analytically. So the model could produce
Sigma directly. It must not, for two reasons.

- **The truth leaves no choice.** TNG gives a *measured curve of growth*
  (`logmstar_cog`, from the isophote analysis). There is no measured
  surface-density profile in the dataset; the only way to get one is to
  difference the measured cumulative curve.
- **The same operator must act on both sides.** Differencing over a finite bin
  gives the *area-averaged* density across that annulus, not Sigma at the
  midpoint. The two differ wherever the profile has curvature across the bin,
  and the difference is largest where the profile is steepest — the inner few
  kpc of compact galaxies, i.e. exactly the population whose central masses are
  the defect under study. Evaluating the model analytically at the midpoint
  while differencing the truth would inject a bias shaped like the signal.

Secondary: exp48's measured result was obtained with both sides differenced.
Changing the model side would be a different objective and would no longer
reproduce it.

Honest caveat, already carried by `density_from_cog`'s docstring: differencing
amplifies noise, and a non-monotonic measured curve gives a non-positive
annulus. Safe on TNG's noiseless profiles; not safe on real observations.

## The epoch axis (new, default unchanged)

Both existing versions take a plain arithmetic mean over the fitted epochs,
hardcoded — a sixth aggregation nobody has ever varied. It is added here with
`"mean"` as the default (so no behaviour changes) plus the option to pass an
explicit weight vector, one number per epoch.

**Why it is worth having.** The handover flags one defect as still open: on the
test of how fast total stellar mass declines with redshift, both the adopted
model and exp48's winner decline too steeply where the truth is nearly flat
between the first two epoch pairs — the truth goes 0.372 -> 0.361 while the
models go 0.403 -> 0.315. Epoch weighting is the obvious untested lever on
exactly that, and until now there was no way to try it without editing the loss
by hand. Note also that the differential gate checks only ONE epoch pair
(z=0.7 -> 0.4), which is what let this shared defect hide.

No `median` / `worst-epoch` options are provided. An untested option in a menu
reads as a recommendation.

## Connection to existing code

`stage2_multiepoch.py::gal_loss` keeps its signature exactly —
`gal_loss(p, g, ks, variant)`, one galaxy, returns a float. Its body becomes:
build the model profiles as now, hand the single galaxy to a module-level
`Objective()`, and apply the `4.0` sentinel if the result is NaN. The sentinel
moves out of the scoring and into this wrapper, where a fit-steering signal
belongs. Net effect on every caller: none.

**The NaN trap this creates, and the guard.** `exp48/objective.py::reduce_galaxies`
*drops* non-finite per-galaxy losses before averaging. Combined with NaN-on-
failure, a fit that started failing on some galaxies would see its loss silently
*improve*. The fitting side must therefore substitute the sentinel before
reducing and never rely on the drop. The equivalence script exercises this
explicitly on galaxies where the kernel fails.

## `hongshao/fitting.py`

One function, `initial_simplex(p0, floor=0.02, rel=0.05)`, returning the
(n+1, n) array to pass as `options=dict(initial_simplex=...)`. `floor` may be one
number or one per parameter.

**The trap it fixes.** scipy's Nelder-Mead (checked against 1.13.1) builds its
starting simplex by perturbing each parameter *relatively*, by 5%
(`nonzdelt = 0.05`) — with a special case for a parameter that is exactly 0.0,
which gets a fixed `zdelt = 0.00025`. Nelder-Mead only explores the space its
starting simplex spans, and near a flat direction it contracts rather than
grows. So a parameter handed a span of 0.00025 in a loss surface where it acts
at order unity comes back essentially unchanged — indistinguishable from a
genuine measurement that the parameter does nothing. This produced a false null
in exp47: three shape slopes started at exactly 0 (deliberately, so the new
component nested the old model as a special case) and came back at ~0.01, a
property of the start point, not of the data.

The bug is **broader than "exactly zero"**: because the step is purely relative,
any parameter starting small in absolute terms but acting at order unity gets a
near-meaningless span. exp48's six conditioning slopes sit *near* zero and hit
the same wall through the ordinary 5% path. Hence a relative step with an
absolute floor, `max(rel * |p0|, floor)`, covering both cases; and hence a
per-parameter floor, because exp47 needed 0.3 / 0.6 / 0.15 for parameters with
different natural scales.

Nothing calls it. Every existing fit keeps scipy's default simplex, so no
adopted number moves.

## Checks

Self-check blocks (`python -m hongshao.objective`, `python -m hongshao.fitting`),
matching `metrics.py` and `qa.py`:

- model equal to truth scores exactly zero;
- each axis moves the number in the direction it should — worst-radius >= rms >=
  mean-abs for a given residual vector; worst-20% >= mean; inner weighting
  scores higher than uniform when the error is central;
- every weight vector sums to one;
- a failed galaxy returns NaN and is counted, not silently dropped;
- `from_dict(to_dict(o))` round-trips; an invalid axis name raises;
- `fitting`: a two-parameter toy where one parameter starts at exactly 0.0 stays
  frozen under scipy's default simplex and reaches the right answer with the
  explicit one — the trap, reproduced as a check.

One-off equivalence script in the branch (not kept): real galaxies, all five
epochs, current `gal_loss` vs the new path, agreement to ~1e-12; galaxies where
the kernel fails, confirming the sentinel substitution; and a timing comparison,
since the whole-population array call is a different shape of computation from
the per-galaxy loop and the faster one is a measurement, not a guess.

## Documentation

- Module docstring: the axes, the same-operator reasoning, the `abs`
  normalization.
- `doc/todo.md`: close the refactor entry; record epoch weighting as newly
  available and untested, tied to the too-steep-with-redshift defect.
- `doc/lessons.md`: write up the scipy simplex trap properly — it has cost two
  false results and lives only in two docstrings.

No figure: a refactor with no new measurement.

## Commits

Branch `objective-module` off master, roughly four commits: the objective module
with its self-check; `fitting.py`; the exp38 port plus the equivalence script's
recorded output; the documentation updates. No merge without permission.
