# exp49 — the railed parameter `g`: optimizer artifact, or the model asking for something it cannot have?

**Status: the rail is REAL — not an optimizer artifact. But `g` is NOT yet a
measured quantity: it is one coordinate of a strong degeneracy, and the
widened-domain optimum has not been properly profiled.** Continues under
`doc/plans/2026-08-16-kernel-identifiability.md`.

> ### Superseded claims — read before the sections below
>
> The chronology is preserved, but these interpretations, made during the
> investigation, are withdrawn:
>
> - **"the rail is an optimizer artifact"** — true only on the 100-galaxy dev
>   subsample. It does NOT generalize; the full sample rails. Run this
>   diagnostic at full size.
> - **"the ridge is flat, so widening the box tells you nothing"** — wrong.
>   exp40's stress fit (box 4 -> 6) found `g = 4.3670`, `log_rc = 2.9611`,
>   loss 0.1535654, better than the bounded 0.1536124.
> - **"there is a genuine minimum at `g ≈ 4.07`"** — read off a scan with the
>   other ten parameters HELD FIXED, and its value (0.153644) is worse than the
>   fully optimized bounded result (0.153612). Not a profiled result.
> - **"per-galaxy fits independently prefer `g ≈ 4.04`, so this is not a
>   joint-fit artifact"** — the exp41 scans vary ONE parameter at a time and are
>   dominated by a single shared weak direction: the fitted corrections are
>   rank-correlated at `g` vs `log_rc` −0.998, `mu` +0.995, `gamma` +0.980, `q`
>   −0.966, `sig` −0.960. That is one degeneracy seen six times, not six
>   measurements.
> - **"small `rc` only removes the core radius"** — wrong. For
>   `Sigma(R) = M(gamma-1)/(pi rc^2)[1+(R/rc)^2]^-gamma` the central density
>   scales as `rc^-2`, so a small scale radius genuinely IS a concentrated
>   component and can legitimately encode a massive galaxy's high central
>   stellar density.
> - **the six-family rail table below mixed TWO objectives** and cannot be
>   ranked in one column. The rail *pattern* survives within the log-density
>   objective alone, but the comparison as first presented was invalid.

## Question

The adopted 1ch-mof theta has `g = 4.00003` against an upper bound of 4.0 —
"the rail". `g` is the exponent in the deposit size law

    rc(t_i) = 10**log_rc * (t_i / t_obs) ** g

where `t_i` is the cosmic time a parcel of mass is deposited and `t_obs` the
time the galaxy is observed. It sets how strongly a parcel's deposited size
depends on WHEN it arrived: large `g` means early material is laid down very
compactly and late material very extended.

A parameter sitting on its bound is not a measurement, so before anything is
read from it: is `g = 4.0` what the DATA wants, or an artifact of how the fit
was run?

Three candidate explanations, and the design that separates them — L-BFGS-B
(`hongshao.fitting.minimize_loss`), fit scope z <= 1.5, full sample (n=2397),
five starting points crossed with two formulations of the box:

| explanation | prediction |
|---|---|
| the DATA wants `g = 4.0` | every start converges there, including one begun at `g = 2.0` |
| the STARTING POINT holds it | starts from below stay below |
| the PENALTY kink holds it | real bounds release it, the soft penalty does not |

The third mattered because the production kernel imposes its box as a
**one-sided quadratic penalty**, whose derivative is exactly zero AT the bound —
so a gradient method can report a converged gradient norm while stuck at a kink
it has no information to leave.

## Result — the rail is real

| formulation | start | loss | `g` | `log_rc` | at a bound | converged |
|---|---|---|---|---|---|---|
| bounded | adopted | 0.153612 | **4.00000** | 2.75274 | `g` | yes |
| bounded | `g_low` (from 2.0) | 0.153612 | **4.00000** | 2.75263 | `g` | yes |
| bounded | `base_x1.2` | 0.153612 | **4.00000** | 2.75256 | `g` | yes |
| bounded | `base_x0.8` | 0.153612 | **4.00000** | 2.75260 | `g` | yes |
| bounded | random | 0.153612 | **4.00000** | 2.75251 | `g` | yes |
| penalty | adopted | 0.153612 | 4.00001 | 2.75262 | `g` | yes |
| penalty | `g_low` (from 2.0) | 0.187374 | 2.48748 | 2.06806 | – | **NO** |
| penalty | `base_x1.2` | 0.153612 | 4.00001 | 2.75312 | `g` | yes |
| penalty | `base_x0.8` | 0.153612 | 4.00001 | 2.75269 | `g` | yes |
| penalty | random | 0.153612 | 4.00000 | 2.75244 | `g` | yes |

Nine of ten converge to an identical loss of 0.153612 with `g` at its upper
bound and `log_rc` interior at 2.7525–2.7531. Starts spanned `g` from 2.0 to
4.0 and `log_rc` from 2.19 to 3.00. **All three explanations are closed: not
the penalty kink (both formulations agree), not the starting point (`g` climbs
from 2.0 to 4.0 under real bounds), so it is the data.**

Two sub-results worth keeping.

- **The one failure is an optimizer failure, not a second minimum.**
  `penalty::g_low` reports `success=False` after 3107 evaluations / 39.5 min at
  a loss 22% worse. It is the only cell combining the soft penalty with a start
  far below the solution, where the route to the optimum runs along the box edge
  and the quadratic barrier lives. **Measured argument for preferring real
  bounds to a box penalty.**
- **The soft penalty is leaky, and that is where `g = 4.00003` came from.** With
  real bounds `g` stops cleanly at 4.00000; with the penalty it settles a hair
  past, at 4.00001 — the same signature the adopted theta carries.

### A small-sample result that does NOT generalize

On the 100-galaxy dev subsample every converging optimizer put `g` at ~3.54,
interior, with nothing at a bound — which reads as "the rail is an optimizer
artifact" and is **wrong for the full sample**. Recorded because it is a live
trap: this diagnostic must be run at full size.

## Physical reading (in progress)

`g` is the model's central-concentration knob. Mass-weighted fraction of the
deposit landing inside 5 kpc — the aperture exp47 used to define the
compact-galaxy defect — at the fitted `gamma`, z=0.4 epoch, first 300 galaxies:

| `g` | f(<5 kpc) | f(<10 kpc) | f(<30 kpc) |
|---|---|---|---|
| 3.0 | 0.113 | 0.201 | 0.400 |
| 3.5 | 0.200 | 0.306 | 0.506 |
| **4.0 (the bound)** | **0.296** | **0.408** | **0.594** |
| 4.5 | 0.389 | 0.497 | 0.664 |
| 5.0 | 0.473 | 0.574 | 0.720 |
| 6.0 | 0.609 | 0.690 | 0.799 |

Each unit of `g` roughly doubles the mass reaching the central 5 kpc. So **the
railed parameter is the lever on the defect exp47/exp48 spent two experiments
characterising** — the model under-filling compact galaxies' centres by 30–44%.

Caveat: this varies `g` alone with the other parameters fixed, so it isolates
`g`'s mechanical role but NOT what the fit would do, since `g` moves along a
strong degeneracy with `log_rc` (and with `mu` and `gamma`). The profiled
version — re-optimizing the other 11 parameters at each `g` — is stage 2 of the
plan and is the one that says whether raising `g` actually buys central mass.

The size law at `g = 4` puts the earliest material at a very small scale. For a
galaxy observed at t = 9.4 Gyr, the deposit scale radius by arrival time:

| `g` | t=0.94 Gyr (z≈6) | t=2.8 Gyr (z≈2.3) | t=6.6 Gyr (z≈0.84) |
|---|---|---|---|
| 3.0 | 0.57 kpc | 15.3 kpc | 194 kpc |
| **4.0** | **0.057 kpc** | **4.6 kpc** | **136 kpc** |
| 6.0 | 0.001 kpc | 0.41 kpc | 67 kpc |

**This is not by itself an argument that the model is unphysical** (an earlier
draft of this README claimed it was). Because the Moffat amplitude carries
`rc^-2`, a small scale radius is a genuinely concentrated component, and massive
galaxies really do have high central stellar densities. The right tests are how
much stellar mass enters that component, its effect on the RESOLVED curve of
growth above 1 kpc, and whether the result survives projection and resolution
checks — none of which have been run.

**The pressure at the margin looks weak in a fixed-parameter slice**: the loss
gradient at the bound is −0.0016 per unit `g` against a loss of 0.1536, while
`g = 3.9` costs 6%. But a slice is not a profile — the widened refit moves `g`
to 4.37, so this slice understates how far the optimum actually lies. Do not
read the margin from it.

## The degeneracy (measured, and the reason `g` is not yet a number)

The loss valley in `(log_rc, g)` is a straight line,
`g = 1.4822 * log_rc - 0.0843`, with scatter of only 0.0044 in `g`. This is the
intercept–slope degeneracy of a straight-line fit whose pivot lies outside the
data: taking logs, `log rc(t_i) = log_rc + g * log(t_i/t_obs)`, so `log_rc` is
the intercept **at `t_i = t_obs`** — the last deposit, where almost no mass
arrives. The data constrains `rc` where the mass is, so intercept and slope
trade off freely.

Converting the ridge slope to a pivot epoch gives `t_p/t_obs = 0.2115`, against
a measured mass-weighted deposition time of 0.2469 geometric (median 0.2301) /
0.2611 arithmetic (median 0.2412). Close, but neither physical proxy is a
substitute for the Hessian-derived pivot.

**Every family in exp48 rails, but not all on the same parameter** — and within
the log-density objective (the only one with all six), moffat, cuspy and
gompertz_log rail `g` at 4.0 while loglogistic, richards and sersic_r50 rail
`log_rc` at 3.0 and come in at `g` = 3.79–3.95. That is the same ridge meeting a
box CORNER, with which wall you hit set by where the family's scale convention
sits along it. Note `rc` is not the same physical quantity across families
(exactly R50 for Sersic and log-logistic, a shape-dependent transition scale for
Moffat), so the shared `log_rc <= 3` bound is not a shared physical constraint.

Precedent, reached and dropped two experiments ago: exp38's README rejected the
sersic candidate with *"n-scale degeneracy; needs an R50 reparameterization"* —
the same diagnosis.

## Stage 1 — local geometry (`jacobian.py`): the degeneracy is much bigger than the `(log_rc, g)` pair

Finite-difference Jacobian of the full fractional-CoG residual array (2397
galaxies x 4 epochs x 24 radii = 230112 residuals) with respect to the 12
population parameters, column-normalized before the SVD. Three steps (1e-6,
1e-5, 1e-4); the weakest-1, -2 and -3 subspaces agree across all three to
**0.000 degrees** in principal angle.

**This is LOCAL GEOMETRY AT A CONSTRAINED REFERENCE, not parameter
uncertainty** — `g` is on an active bound here.

**The headline: all six BASE parameters are near-collinear in their effect on
the residuals.**

|  | log_rc | g | q | mu | sig | gamma |
|---|---|---|---|---|---|---|
| **log_rc** | 1.000 | −0.995 | 0.942 | −0.981 | 0.921 | −0.960 |
| **g** | −0.995 | 1.000 | −0.958 | 0.987 | −0.893 | 0.940 |
| **q** | 0.942 | −0.958 | 1.000 | −0.956 | 0.834 | −0.868 |
| **mu** | −0.981 | 0.987 | −0.956 | 1.000 | −0.908 | 0.940 |
| **sig** | 0.921 | −0.893 | 0.834 | −0.908 | 1.000 | −0.938 |
| **gamma** | −0.960 | 0.940 | −0.868 | 0.940 | −0.938 | 1.000 |

Every base-parameter pair correlates at |rho| = 0.834–0.995 (median 0.940). The
conditioning slopes are a **separate and far better-conditioned block**
(|rho| median 0.393 among themselves) and are **nearly orthogonal to the base
parameters** (|rho| median 0.083, max 0.325).

**Effective rank: 7 of 12** singular values exceed 10% of the largest. The
spectrum declines smoothly with no clean gap — the classic sloppy-model
signature — from 2.412 to 0.026 (condition number 93.3).

Weakest directions, with their squared overlap with the `(log_rc, g)` plane:

| | singular value | in (log_rc, g) | largest entries |
|---|---|---|---|
| sv11 | 0.02586 | **0.942** | g +0.712, log_rc +0.660, mu −0.197 |
| sv10 | 0.07791 | 0.297 | mu −0.812, log_rc −0.497, g +0.223 |
| sv9 | 0.16664 | 0.193 | gamma −0.681, q +0.440, g +0.361 |

**What this means for the planned fix.** The single weakest direction IS
essentially the `(log_rc, g)` pair (94%), so the time pivot targets a real
object. But it is only the weakest of **five or six** poorly-constrained
directions among the base parameters, and the next two are dominated by `mu`
and by `gamma`/`q`. **A time pivot alone will not make the base parameters
individually interpretable** — it flattens one direction out of several. An
earlier reading of this section's dev-sample version ("94% in the plane, good
news for the pivot") was too optimistic: that figure describes the weakest
direction only, not the degeneracy as a whole.

Raw column norms (absolute sensitivity) put `g` second-lowest of the twelve at
326.7 and `q` lowest at 245.6, against `sig` at 856.7 — so `g` is both weakly
influential and strongly entangled.

The valley slope from the full 12-parameter geometry is **d(g)/d(log_rc) =
+1.079**, against +1.4822 from the 2-D grid with the other ten parameters
frozen. The two measure different things and need not agree.

Figures: `figures/exp49_jacobian_full.png` (correlation + singular vectors),
`figures/exp49_spectrum_full.png` (sensitivity spectrum).

## Open — see `doc/plans/2026-08-16-kernel-identifiability.md`

1. Residual-Jacobian SVD at the bounded reference (column-normalized).
2. Widened multi-start fit and a **properly profiled** `g` curve — re-optimizing
   the other 11 parameters at each fixed `g` — under BOTH objectives. Must
   resolve 4.07 vs 4.37, and widen `log_rc` too or the fit just moves to the
   other corner.
3. Interior Hessian at the widened solution (NOT at `g = 4`, which is an active
   boundary), and how much of the weak subspace the `(log_rc, g)` plane actually
   spans — the conditional correlations with `mu` and `gamma` are nearly as
   strong, so the degeneracy may exceed two dimensions.
4. Two independent coordinate fixes: the time pivot, and a primitive-R50
   parameterization (`R50 = rc*sqrt(2**(1/(gamma-1)) - 1)` for Moffat, verified
   exact).
5. Only then: the fit-scope factorial with a shuffled-MAH null, and a fair
   family comparison in common physical coordinates.

## Files

- `railtest.py` — the ten-cell test. Resumable; saves after every cell.
- `outputs/railtest.npz` — thetas, losses, starts, convergence flags.

Run: `HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python
experiments/exp49_g_rail/railtest.py [--dev]` (~3 h full, ~10 min dev).
