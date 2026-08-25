# exp54 Stage 3.9 — genuine profile likelihoods (open question B1)

*Written 2026-08-25, before the run, so that what was decided in advance can be
told apart from what the numbers suggested afterwards.*

## The question

Every statement this experiment has made about whether a parameter is
determined is a **conditional slice**: one parameter is displaced and the other
six are held **fixed**. That answers "how much worse does *this* fit get if I
nudge one number". Nobody means that. What they mean is "how much worse does
the **best achievable** fit get if I insist on this value", and answering it
requires re-optimising the other six at every value tried.

The two differ whenever parameters are correlated, and they differ in one
direction only: a displacement in one parameter can be partly undone by the
others moving to compensate, so the profiled curve is never steeper than the
conditional slice and is usually much flatter. **The conditional version
therefore systematically overstates how well determined a parameter is.**
`fit.py`'s own docstring says exactly this and says not to quote the slice as
evidence on its own. It has never been done. This stage does it.

## The object under test

`gompertz_log-E2-S2`, the incumbent, fitted on the clean sample
(`selection.sane_history_mask` applied — the broken cross-match at row 181 is
out). Seven parameters, shared by all 2397 galaxies, no stellar input and no
per-object tuning:

| parameter | what it sets |
|---|---|
| `a0` | how much stellar mass a halo of `10^13.5 M_sun` keeps, at redshift 0 |
| `a_M` | how that changes with halo mass |
| `a_z` | how it changes with redshift |
| `a_Mz` | how the mass dependence itself changes with redshift |
| `log_f0` | the size of a deposit, as a fraction of its halo's radius, at redshift 0 |
| `b` | how that size fraction changes with redshift |
| `c` | the shape of a single deposit's profile (Gompertz) |

Incumbent values `a0=-2.5525 a_M=-0.5625 a_z=+0.5055 a_Mz=+0.3178
log_f0=-0.8433 b=-0.8957 c=+0.7965`, loss **1.874253**, profile-shape error
**13.787%** — the typical error the model makes on the normalised curve of
growth `M*(<R)/M*(<100 kpc)`.

## Design, fixed in advance

1. **The grid is geometric, not linear**, at `±1, 2, 4, 8, 16 × d0`, where `d0`
   is measured (by bisection, on the production evaluator) as the displacement
   at which the **conditional** rise reaches 0.10 in loss units — about 0.9
   percentage points of profile-shape error. `d0` is measured separately in
   each direction because the loss is not symmetric: at the incumbent, `c` must
   move `+5.02` to cost 2.0 in loss and only `−0.40` to cost the same.
   The conditional slice is very nearly exactly quadratic here (the
   displacement scales as the square root of the rise to three digits over two
   decades of rise), so the grid spans conditional rises from 0.10 to
   `0.10 × 256 = 25.6` — from "slightly worse" to "catastrophically worse".
   Whatever the profile does across that span is the answer.

2. **The conditional slice is measured at the identical grid points**, one loss
   evaluation each, so the headline comparison is point-by-point and needs no
   quadratic model of either curve.

3. **Two starts per grid point**: the neighbouring point's solution
   (a continuation sweep outward from the centre in both directions, so the
   curve is smooth rather than a scatter of independent optimiser outcomes) and
   the incumbent's own six values. Better of the two kept. Nelder-Mead, the
   same optimiser and the same bound handling as every other fit in exp54.

4. **A gate at the centre.** The middle grid point pins the parameter at its
   own incumbent value and re-optimises the rest, so it must return the
   incumbent's loss. Higher means the optimiser is failing; materially lower
   means the incumbent was never converged. Either way every rise on that
   curve would be measured against the wrong zero.

5. **Nothing is a confidence interval.** The loss is
   `score_A² + score_F²`, a weighted sum of two scores each normalised to ~1 at
   the halo-only regression benchmark. It is not a log-likelihood and no `Δ`
   threshold on it means anything distributionally. Every width is quoted as
   *how far the parameter can move before the best achievable fit costs one
   percentage point of profile-shape error* (13.79% → 14.79%), which is
   `Δloss = 0.114`.

## Rejected in advance, and why it is worth recording

The first design sized the grid from the **Gauss-Newton prediction**
`1/(H⁻¹)_jj` of the profiled curvature. **It does not work here**, and the
failure is informative. Across finite-difference steps spanning a factor of
ten, the five largest eigenvalues of the loss Hessian are stable to better than
0.1% while the two smallest move by a factor of 25 and one goes **negative**.
This is not a convergence problem — re-optimising all seven parameters from the
incumbent reproduces its loss to the last digit — it is cancellation: pulling a
curvature of order 0.1 out of a matrix whose diagonal is of order 100.

A seven-parameter model with two directions too flat to measure by differencing
is exactly the situation in which conditional slices mislead, so the Hessian is
still computed and stored, as a diagnostic rather than as a foundation.

## What "success" means

A statement of which of the seven are genuinely determined, with a width in the
currency above, replacing a claim that is currently conditional. **A parameter
that turns out to be poorly determined is a finding, not a failure**: the
model's headline results do not rest on any single parameter's value, and
`fit.py`'s own rule is that a parameter that fails may stay in the model but
its value is never quoted.

## Watch list, from the handover

- **`log_f0`'s upper bound is 0.0** — a deposit half-mass radius equal to its
  halo's radius at redshift 0 — and a Stage 3.7 fit has already reached it. If
  a profile runs into that bound, the curve is a property of the bound and must
  be reported as one (open question C7).
- **Nelder-Mead in six dimensions is not reliable from a single start.** Hence
  the continuation sweep.
- **`--smoke` must never write a production filename.** It writes
  `stage39_grid_SMOKE.npz` and `stage39_profiles_SMOKE/`.
