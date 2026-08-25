# exp54 Stage 3.8 — a second deposit channel with its own radial scale

**Status: PLAN. Not started.** Written 2026-08-25, after Stage 3.7.

## The one sentence this stage exists to test

Stage 3.7 established that **with one radial scale per deposit, the central mass
at redshift 2 and the outer mass at redshift 2 cannot both be right**: a free
shared size law fitted to reduce the central epoch swing reaches 0.061 dex from
0.145, and pays for it with 0.104 dex of outskirt mass at the same epoch,
taking the model from 2% to 23% too light beyond 52 kpc.

**A second channel with its own scale should break that trade.** If it cannot,
the deposition picture itself is the limit, not its parameterisation. Either
answer is worth having, and the experiment is the same either way.

## What is being added, and what it is NOT

Each deposit's unit profile becomes a mixture of two families, weighted by
**when the mass was laid down**:

    F_j(<R) = w_j * F_compact(<R ; rho * R50_j) + (1 - w_j) * F_ext(<R ; R50_j)

- `F_ext` is the incumbent family (`gompertz_log`) at the incumbent size law.
- `F_compact` is an **exponential** (`families.expo`, Sersic n = 1 fixed, no
  free shape parameter), at a fraction `rho < 1` of the same radius. **`rho` is
  the second radial scale, and is the whole point.**
- `w_j` is the compact fraction at the deposit's own redshift.
- Both are truncated at the same `R_trunc` and are individually unit-mass, so
  the mixture is unit-mass and mass conservation is untouched.

**Two things this is not**, both recorded so they are not conflated later:

1. **Not two components per deposit in the sense of a fixed decomposition of
   every galaxy.** The user was explicit: "by two component, it doesn't mean
   every deposit should have two component". What varies is the MIXING WEIGHT
   with deposition time — early deposits compact, late deposits extended.
2. **Not the rejected `c_z`.** exp54 rejected a redshift-dependent deposit
   SHAPE parameter: one family's shape drifting continuously with redshift.
   This switches between two FAMILIES at two different SIZES, which `c_z` could
   not express at any parameter value. (`c_z` was also rejected on the sample
   that still contained row 181 — `doc/open_questions.md` A4.)

## Why an exponential, and how much weight that evidence carries

A parallel experiment (exp55, another agent) finds that an **exponential core
plus an extended Moffat** reproduces measured curves of growth as well as a
three-component decomposition, and that those profiles connect to halo growth
**when constrained at a single epoch**.

That constrains one epoch; this model must serve five, so **the conclusion does
not transfer — only the hint about the inner component's shape.** The
exponential is therefore adopted as the first thing to try, not as a result. If
it fails, `gauss` (n = 0.5) and `sersic` with free `n` are the fallbacks, and
the failure of the exponential specifically would itself be worth reporting
back to exp55.

## The nested ladder, cheapest first

Every rung reduces to the incumbent exactly when its new coefficients are zero,
and `model.py` must assert that bit-for-bit as it does for E6/E7/E8 and S4/S5/S6.

| rung | the compact fraction `w` | extra parameters |
|---|---|---|
| `P1` | a constant `w0` | 2: `w0`, `log_rho` |
| `P2` | logistic in `ln(1+z)`, transition fixed at z = 2 | 3: `w0`, `s`, `log_rho` |
| `P3` | free logistic: amplitude, transition, width | 4: `w0`, `x_t`, `s`, `log_rho` |

`P1` is the reviewer's prescribed second step ("then try a fixed compact
fraction") and is the control for `P2`/`P3`: if a *constant* compact fraction
does as well as a time-varying one, the time dependence has not earned itself.

## The decisive test, which is not the loss

Stage 3.7's lesson, learned the hard way and for the third time: **a bound must
be optimised on the quantity it is reported on.** So this stage runs each rung
twice.

1. **Fitted under the production loss**, as any candidate model would be. This
   answers "would we adopt it".
2. **Fitted to minimise the central span directly**, loss held within 5% of the
   incumbent's — the identical protocol to Stage 3.7's span bound. This answers
   the structural question, and it is the one that matters:

> Can the mixture reach a central span near 0.06 dex **while the error beyond
> 52 kpc stays near the incumbent's −0.001 to −0.008 dex?**

- **If yes**: the second scale breaks the trade. That is the result, and the
  next question is whether a fit under a *repaired* objective would find it on
  its own.
- **If no** — if it empties the outskirts exactly as the one-scale law did —
  then the trade is not about radial scales at all, and the deposition picture
  is the limit. Stage 3.4's finding that **53% of these galaxies have a
  declining central mass** would then be the whole story.

## Also required

- **Run it on the non-declining subset separately.** Stage 3.4 split these
  galaxies on whether their measured mass inside 4.9 kpc falls with time; a
  deposition-only model cannot follow a decline, so the compact channel can only
  address the 47% that do not. Reporting the population number alone would
  understate a real effect or overstate a fake one.
- **The identifiability report.** `w0` and `log_rho` are expected to trade
  against each other — a small very-compact component and a larger mildly
  compact one make similar central mass. If they are undetermined, say so and
  quote neither.
- **Watch the cost.** Two `cog_truncated` calls per evaluation instead of one,
  so roughly twice the fitting time. Measure it before launching the ladder.

## Open questions this stage should record

- Does the compact component want to be truncated at the same three halo radii
  as the extended one? Almost certainly not, and `C = 3` has never been scanned
  (`doc/open_questions.md` B2).
- If `P1` (a constant compact fraction) matches `P3`, is the "early compact"
  picture wrong, or merely unnecessary at this resolution?
- exp55 pairs its exponential with a **Moffat**; the incumbent here is
  `gompertz_log`. If the mixture helps, is the extended family's identity worth
  re-testing under it — given exp48 found the objective reverses the family
  ranking?
