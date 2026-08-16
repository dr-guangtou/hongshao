# Plan — kernel identifiability, physical coordinates, and a fair profile-family comparison

**Date:** 2026-08-16 · **Status:** agreed program, not yet executed.
**Provenance:** drafted from an independent review of the exp49 rail test,
merged with seven modifications and four qualifications agreed in discussion.

**Starting point:** the adopted Moffat transport kernel (1ch-mof, 12
parameters) fitted to 2397 galaxies over z = 0.4–1.5 with the production
fractional-CoG objective.

## Goal

Determine which combinations of the 12 population parameters the data actually
constrains, locate the widened-domain optimum of `g`, separate population
sharing from multi-epoch constraints, and repeat the deposit-family comparison
in common physical coordinates.

**All work is experimental until the full scientific evaluation supports
adoption. Do not change the adopted parameters or the fiducial library
interface during this program.** (`hongshao.objective` and `hongshao.fitting`
are additive and are the intended tools; using them is not a change to the
adopted model.)

## Why this program exists

exp49 established that `g` — the exponent in the deposit size law
`rc(t_i) = 10**log_rc * (t_i/t_obs)**g` — sits exactly on its upper bound of
4.0 in the adopted fit, and that this is **not** an optimizer artifact: nine of
ten converging cells, spanning starts from `g = 2.0` to 4.0 and both box
formulations, land at `g = 4.00000` with an identical loss of 0.153612.

But three findings show the value itself is not yet a measurement.

1. **`g` and `log_rc` are strongly degenerate.** The loss valley is a straight
   line, `g = 1.4822 * log_rc - 0.0843`, with scatter 0.0044 in `g`. This is
   the intercept–slope degeneracy of a straight-line fit whose pivot
   (`t_i = t_obs`, the last deposit) lies outside where the data is.
2. **The whole 12-parameter fit shares one weak direction.** In exp41's
   per-galaxy conditional scans the fitted corrections are almost perfectly
   rank-correlated: `g` vs `log_rc` −0.998, vs `mu` +0.995, vs `gamma` +0.980,
   vs `q` −0.966, vs `sig` −0.960. The tight per-galaxy `g` distribution
   (median 4.042, 16–84% 3.840–4.290) is that one direction seen six times, not
   six independent measurements.
3. **Widening the box moves `g` a long way.** exp40's stress fit (box 4 → 6,
   Nelder-Mead) found `g = 4.3670`, `log_rc = 2.9611`, loss 0.1535654 — better
   than the bounded 0.1536124. A fixed-other-parameter ridge scan suggested
   4.07; the actual refit says 4.37. Only a properly profiled fit can resolve
   this.

### Superseded claims (correct the record first)

These were stated during the investigation and are wrong or unsupported:

- *"the rail is an optimizer artifact"* — true only on the 100-galaxy dev
  subsample (where every converging method gives `g ≈ 3.54`, interior). **It
  does not generalize; the full sample rails.** This diagnostic must be run at
  full size.
- *"the ridge is flat, so widening the box tells you nothing"* — the ridge has
  structure, and widening moves `g` to ≈ 4.37 with a real loss improvement.
- *"there is a genuine minimum at `g ≈ 4.07`"* — read off a scan with the other
  ten parameters **held fixed**, and its value (0.153644) is worse than the
  fully optimized bounded result (0.153612). Not a profiled result; withdrawn.
- *"per-galaxy fits independently prefer `g ≈ 4.04`, so it is not a joint-fit
  artifact"* — the conditional scans vary one parameter at a time and are
  dominated by the shared weak direction (point 2 above). Withdrawn.
- *"small `rc` only removes the core radius, it does not concentrate mass"* —
  wrong. For `Sigma(R) = M(gamma-1)/(pi rc^2) [1+(R/rc)^2]^-gamma` the central
  density scales as `rc^-2`, so a small scale radius **is** a genuinely
  concentrated component and can legitimately encode the high central stellar
  density of a massive galaxy.
- **the six-family rail table mixed two objectives** — moffat/cuspy/sersic_r50
  values came from the production objective, gompertz/loglogistic/richards from
  the log-density objective. Those cannot be ranked in one column. The rail
  *pattern* does survive within the log-density objective alone (moffat, cuspy,
  gompertz_log rail `g`; loglogistic, richards, sersic_r50 rail `log_rc`), but
  the table as presented was invalid.
- *"no parameter-impact visualizations exist"* — wrong; see
  `experiments/exp48_objective_profile/figures/exp48_profiles.png` (all
  primitives at the same R50 and total mass, so shape alone) and
  `experiments/exp38_deposit_rethink/figures/stage1_deposit_tracks_dev.png`
  (fitted scale, `g` and shape across epochs and families). Neither is the full
  atlas of stage 6.

## Stage 1 — correct the record, and diagnose the present model

- **Amend the exp49 README before adding results.** Preserve the chronology;
  mark the superseded claims above as such.
- Compute the finite-difference **Jacobian of the full fractional-CoG residual
  array** at the current bounded solution.
  - Report raw column norms (absolute sensitivity).
  - **Column-normalize before the SVD** — otherwise the singular vectors report
    unit choices rather than degeneracies.
  - Save singular values, right-singular vectors, finite-difference steps and
    parameter names.
- **Do NOT interpret a Hessian at `g = 4` as unconstrained curvature.** `g` is
  on an active boundary; the Hessian belongs at the widened interior solution
  (stage 3).
- Produce a parameter-correlation heatmap and a singular-vector heatmap,
  labelled **local geometry at the bounded reference, not parameter
  uncertainty**.

## Stage 2 — the widened-domain optimum and the profiled `g` curve

Fit all 12 parameters with L-BFGS-B (`hongshao.fitting.minimize_loss`), **true
bounds, no duplicate box penalty** — measured: the penalty formulation failed
from a far start where real bounds succeeded.

- Domain: `0 <= g <= 6`, `1 <= log_rc <= 3.5`; keep existing physical bounds on
  the other four base parameters; leave the six conditioning slopes unbounded.
- Five starts: (1) the exp49 bounded solution; (2) the exp40 widened solution at
  `g = 4.367`; (3) `g = 2` with `log_rc` shifted along the measured ridge;
  (4) `g = 5` with the pivot radius preserved; (5) one deterministic displaced
  start, seed recorded.
- If the best solution lies within 2% of a widened boundary, expand and repeat.
  Confirm no internal scale clipping is active.
- Require convergence and agreement of independent starts to within 1e-6 in the
  mean objective.

**Bounds audit (all parameters, not just `g`).** Report a normalized
distance-to-bound for every population parameter and every conditioned
per-galaxy effective parameter. `g` is unlikely to be alone: exp41 shows `sig`
at a box edge for 4.0% of galaxies and `q` for 1.7%, and exp38's README already
flags *"efficiency peak mu railed at 2/5 epochs (watch)"*. Stress active
**empirical** bounds individually and jointly. **Do not blindly widen
mathematical constraints** — `gamma > 1` is required for finite Moffat mass and
`sig > 0` is structural; approaching those indicates the wrong parameterization
or primitive, not a bound that needs loosening.

**Profile the objective over `g`:** coarse grid 3.0 → 5.5 in steps of 0.25;
refine within ±0.30 of the minimum in steps of 0.05; **re-optimize the other 11
parameters at every fixed `g`**; warm-start from both grid directions plus an
independent displaced start at selected points; extend the grid if the
objective has not risen on both sides before an endpoint.

Record per profiled point: optimized parameters and convergence status; mean
production objective relative to the global minimum; primitive radii at fitted
deposition-time quantiles; median model-to-TNG residual in M*(<5 kpc) and
M*(<10 kpc); central aperture residuals for compact and extended galaxies
separately; all four differential-deposition epoch pairs beyond 50 and 100 kpc.

Figures: profiled objective vs `g`; `log_rc`, physical pivot radius and other
parameters vs `g`; central aperture residuals and differential deposition vs
`g`; measured and modelled CoGs plus residuals for representative galaxies at
the minimum and at statistically similar points along the profile.

**Both objectives.** Run a complete widened fit and profiled `g` curve
separately for the production fractional-CoG objective **and** the log-density
objective. exp48 measured the objective as the strongest lever on this model,
and `g` is the central-concentration knob that log-density reweights, so the
optimum may well move. **Do not compare their objective values numerically.**
Compare optimum `g`, profile-curve width, physical deposition radii, central
residuals and differential deposition. **Use the production-objective pivot as
the common downstream coordinate convention.**

**Do not convert objective differences into likelihood confidence intervals.**
This is a fitting loss, not a likelihood. Interpret constraint strength through
fractional objective changes, resolved prediction changes, and resampling
stability.

## Stage 3 — interior geometry, then physical coordinates

At the converged widened-domain solution:

- Compute the 12-parameter Hessian of the actual scalar objective at **three
  standardized central-difference steps: 1e-6, 1e-5, 1e-4.** (Measured:
  gradients on this loss are stable from 1e-4 down to 1e-7, while 1e-2 misreads
  `g` by a factor of five. `minimize_loss` refuses steps coarser than 1e-4.)
- Report raw diagonal curvatures and the unit-invariant Hessian correlation
  matrix. Compare the weak subspace with the residual-Jacobian SVD.
- Compare **across steps** by gradient directions, Hessian correlations and
  weak subspaces. **For nearly equal eigenvalues compare eigenSPACES via
  principal angles**, not individual eigenvectors, which rotate freely.
- Robust sandwich covariance from per-galaxy gradients with the Hessian, for
  finite-sample variation. Validate against a **pilot** set of full
  galaxy-bootstrap refits before choosing the final bootstrap count; record
  runtime and convergence first.

**How many dimensions is the degeneracy?** Measure the fraction of each weak
eigenvector lying in the (`log_rc`, `g`) plane, and the **principal angles
between that plane and the full weak subspace**. The time pivot remains a
useful physical coordinate improvement even if the overlap is modest — but in
that case it must be described as *one* coordinate improvement, **not** a
solution to the whole degeneracy. The conditional correlations (`g` with `mu`
+0.995 and `gamma` +0.980, nearly as strong as with `log_rc` −0.998) make this
a live possibility.

Then introduce **two independent coordinate changes**, tested separately and
then together.

### 3.1 Time pivot

- Derive the fixed pivot that locally removes the Hessian cross-term between
  the size intercept and `g`.
- Compare with the equal-galaxy, equal-epoch **geometric** mass-weighted
  deposition time. Measured for reference: geometric mean t/t_obs = 0.2469
  (median 0.2301), arithmetic 0.2611 (median 0.2412), against the
  ridge-implied pivot 0.2115 — so neither physical proxy is a substitute for
  the Hessian-derived value.
- **Freeze one population pivot.** Do not recompute per galaxy or during
  optimization.

### 3.2 Primitive half-mass radius

- Replace each family's native scale with its exact primitive R50. For Moffat,
  `R50 = rc * sqrt(2**(1/(gamma-1)) - 1)` — **verified exact**
  (M(<R50)/M_tot = 0.5000000000 at the fitted gamma).
- Implement exact conversions for every later family.

**Equivalence checks — and the one requirement that must be relaxed.** Require
**exact prediction and objective equivalence at matched parameter points**
(max relative difference in CoGs and in the population objective below 1e-10,
across halo-conditioned parameters and every fitted epoch).

Exact *feasible-domain* equivalence is **not achievable** and should not be
required: both the time pivot and the R50 conversion turn the old rectangular
box into a curved domain, and L-BFGS-B accepts only rectangles. R50/rc runs
from 32.0 at gamma = 1.1 to 0.64 at gamma = 3.0, so the coupling is severe.
Instead: **define a new rectangular domain in physical coordinates and verify
the relevant old and widened optima lie comfortably inside it.**

Recompute the Hessian after transformation. Predictions and the profiled `g`
curve must be unchanged; improved numerical conditioning is the only expected
effect.

## Stage 4 — separate population sharing from multi-epoch constraints

Run the adopted Moffat model in the new physical coordinates in four cells:

| fit | z = 0.4 only | four epochs, z <= 1.5 |
|---|---|---|
| population | fit all 12 parameters | fit all 12 parameters |
| individual galaxy | fit six effective base parameters | fit six effective base parameters |

Individual fits: do not fit halo-conditioning slopes separately; validate
multiple starts on a stratified subset (by halo mass, compactness and fit
quality) before scaling to all 2397; retain optimizer status, bound hits,
objective, parameter vector and derived physical radii; profile the dominant
weak directions for representative galaxies rather than treating point
estimates as PDFs.

Compare the four cells on: joint and marginal parameter distributions;
parameter-correlation matrices; physical deposition radii at supported time
quantiles; central and outer aperture residuals; direct per-object CoG and
residual plots; and the **change in degeneracy structure between one and four
epochs**.

**Shuffled-MAH null (required before any assembly claim).** Shuffle the
complete MAH **and its derived fz2 together**, within narrow final-halo-mass
bins, keeping the galaxy profile, M(<500 kpc) and c_200c fixed. Use **several
deterministic shuffles**, seeds recorded. A flexible per-galaxy fit can absorb
a shuffled MAH, so a small loss penalty would NOT show assembly is irrelevant —
**the meaningful comparison is paired-versus-shuffled behaviour across the
one-epoch and four-epoch scopes.**

## Stage 5 — repeat the profile-family comparison fairly

Fit Moffat, cuspy Moffat, Sersic, Gompertz, log-logistic and Richards
primitives with: the same fixed time pivot; primitive R50 for every family; one
common physical R50 domain; the widened `g` domain; the same redshift scope,
starts, optimizer and convergence requirements; and the production
fractional-CoG objective as the primary comparison.

Choose the common R50 domain by mapping the union of existing family domains
into physical R50, then widening until every primary optimum is more than 2%
from a boundary. Apply that final domain unchanged to all families.

Use **identical 10-fold held-out splits** to account for differing numbers of
shape parameters. Run the log-density objective only as a **second complete
comparison with all six families**; never combine objectives in one ranking.

Full evaluation per family: held-out CoGs and surface-density profiles; central
aperture masses for all / compact / extended galaxies; outer-envelope
residuals; all differential-deposition epoch pairs; observational mass–mass and
size planes; parameter and physical-radius bound diagnostics; representative
truth-versus-model profiles and residuals.

**Treat exp48's "profile form is a weak lever" as provisional until this
controlled comparison is complete.**

## Stage 6 — the parameter-impact atlas

Two complementary response views:

1. **One parameter at a time**, explaining the mechanical role of
   `log_R50_piv`, `g`, `q`, `mu`, `sig`, `gamma`.
2. **The model varied along the strongest and weakest** Jacobian/Hessian or
   profiled directions — the combinations the data actually allows.

Choose perturbation sizes from measured profile curvature or bootstrap spread,
**not equal numerical steps**.

Show at least a median, a compact, an extended and a high-stellar-mass galaxy.
At every epoch display: the primitive deposition profile; the transported CoG;
the transported surface-density profile; changes in M*(<5 kpc), M*(<10 kpc),
M*(50–100 kpc), M*(<148 kpc); and residuals against the corresponding TNG
profile.

Save PNG + PDF and surface every figure for visual review.

## Execution hygiene

- **Coupled full-population fits: serial by default**, resumable, saving
  **atomically after every start, fixed-`g` cell, objective and bootstrap
  replicate.** The pooled path has hung twice (0% CPU, parent blocked in
  `pool.map`, workers reporting BrokenPipeError), costing hours each time.
- **Independent per-galaxy fits may parallelize** — exp41 and exp44 did so
  reliably. Benchmark a small stratified batch serially and with a bounded
  worker pool before choosing.
- **Stall detection must monitor the actual Python worker plus a heartbeat or
  evaluation counter — never wrapper CPU.** Measured 2026-08-16: during a
  healthy 3-hour run, `uv` and `caffeinate` both reported 0.0% CPU while the
  Python worker ran at 100%. A naive wrapper-CPU alarm would have killed it.
- Long runs via `nohup` + `caffeinate` + `disown`; always
  `export HONGSHAO_DATA_DIR=...`.
- Rank candidates with the judge, never by loss — they can be anti-correlated.

**Early pilot (scheduling).** Run a small, full-six-parameter per-galaxy pilot
early to measure cost and expose additional degeneracies: a single galaxy's
loss is ~0.32 ms, so a six-parameter fit may be ~0.1 s and all 2397 may take
minutes per start rather than hours. **Even if it is very fast, retain the
exact physical reparameterization before scaling to all 2397** — the point of
the execution order is not to run expensive fits in coordinates already known
to be badly correlated.

## Records and stopping rules

- Keep the widened fit, profile curve and local-geometry work in **exp49**.
- Create subsequent numbered experiment directories only when their stage
  begins: physical coordinates, fit-scope factorial, fair family comparison,
  parameter-impact atlas.
- Every output stores the git SHA, sample definition, objective, parameter
  bounds, starts, optimizer status and finite-difference settings.
- Update `doc/todo.md` at each stage boundary; record corrected interpretations
  in `doc/lessons.md`.

**Stop and diagnose before continuing if:**

- independent starts disagree by more than 1e-6 in objective;
- an optimum reaches a widened boundary;
- the transformed model fails numerical equivalence;
- finite-difference weak subspaces are unstable across steps;
- failed galaxies are dropped instead of penalized during fitting
  (`hongshao.objective.reduce_galaxies` DROPS non-finite entries — measured,
  five failures of 2397 moved the population loss from 0.149704 DOWN to
  0.149470; the fit wrapper must substitute its sentinel before reducing).

## Execution order

1. Correct exp49 and run the bounded-solution residual Jacobian.
2. Widened multi-start fit and profiled `g` curve (both objectives).
3. Interior Hessian, weak-subspace dimensionality, local sampling covariance.
4. Introduce and verify the fixed time pivot and the incumbent Moffat R50
   coordinates.
5. Four-cell population/epoch experiment in the new coordinates, with the
   shuffled-MAH null.
6. Generalize physical coordinates to all families; fair family comparison.
7. Two-layer parameter-impact atlas.

This order keeps the expensive 2397-galaxy fits out of coordinates already
known to be badly correlated.

## Deferred — the probabilistic model

**Do not run a Bayesian sampler on the current fitting objective.** It is a
loss, not a likelihood; exponentiating it gives "posterior" widths set by an
arbitrary scale factor. A later hierarchical model must first define:
halo-conditioned population distributions for per-galaxy latent kernel
parameters; an explicit residual covariance for CoGs or surface-density
profiles; priors in the physical pivot and R50 coordinates; and held-out
posterior-predictive and calibration tests.

**Marginal parameter PDFs alone will not count as a degeneracy analysis.** The
required products are joint distributions, weak directions, and posterior
distributions of derived physical radii.

Stages 1–3 obtain most of the degeneracy insight — Jacobian SVD, Hessian
eigenstructure, profiled losses, bootstrap — at a small fraction of the cost.
