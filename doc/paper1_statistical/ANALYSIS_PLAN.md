# Confirmatory analysis plan

**Operational update, September 28:** implementation will live in the
separate `ushmr1` repository, not a new HongShao experiment. Follow
[USHMR1_LAUNCH_PLAN.md](USHMR1_LAUNCH_PLAN.md) for stages, figure packages,
restart/stop rules, and current authorization. The scientific comparisons
here remain the detailed reference. No implementation was run in this closeout.

Draft updated September 27, 2026 with the user's scope decisions; see
[agreed scope and representation strategy](DECISIONS_20260927.md).
This specifies a bounded study, not permission to start fitting. No new
experiment number is reserved. Freeze implementation details and check active
ownership before implementation in the separate `ushmr1` repository.

## 1. Define exactly what we will test

Let `H` be the measured halo mass at the observation epoch, `A` a description
of its measured assembly history, and `C` its concentration. Let

\[
 S=\log_{10}M_\star(<R_{\rm out}),\qquad
 q(R)=\log_{10}M_\star(<R)-S,\qquad
 r=\log_{10}R_{50}.
\]

`R_out` is the native outer aperture near 148 kpc (use its exact stored value).
`R50` encloses half of the **same finite-aperture** stellar mass. `Y` denotes
the complete discretized profile, including its amplitude. All masses are
h-free, all radial conventions explicit. The anchor and radial support will
be chosen from measurement/observational considerations before viewing new
scores, not selected for the strongest association.

### The nested comparisons

| Comparison | Reference | Added information | Scientific question |
|---|---|---|---|
| F1: full halo-only profile | `P(Y | H)` | `A` | Does assembly improve on a mass-dependent profile relation? |
| F2: concentration-controlled | `P(Y | H,C)` | `A` | Is MAH useful beyond concentration? |
| S1: amplitude-controlled shape | `P(q | H,S)` | `A` | Is there information beyond stellar mass? |
| S2: size-controlled shape | `P(q | H,S,r)` | `A` | Is there information beyond mass and size? |
| S3: strongest attribution check | `P(q | H,S,r,C)` | `A` | Does finer shape trace history beyond concentration as well? |
| R1: reverse summary test | `P(a | H,S)` and `P(a | H,S,r)` | observed profile shape | Can radial information distinguish a specified assembly summary? |

R1 is optional supporting evidence, not a required comparison. Forward
prediction is the scientific objective; accurate individual-MAH recovery is
not a goal for Paper 1 or the project.

`a` is a small, predeclared subset of interpretable summaries of `A`, not a
claim to reconstruct the full time series. S1–S3 and R1 are **information
audits** using measured stellar quantities, not production halo-only
predictors. Conditioning on size also removes potentially real
assembly-mediated structure; report F1/S1 as well as S2. None of these
conditioning operations establishes a causal effect.

A genuine conditional dependence is not directional, but the achievable
accuracy and prior sensitivity of an inverse predictor are not supplied by
a forward model's RMS. The reverse test makes the paper's “stored
information” language concrete without constructing an inference framework.

## 2. Phase A — certify inputs and sample before fitting

**Deliverable:** frozen input manifest, data dictionary, sample membership,
fold membership, and attrition table.

1. Use private snapshots of settled artifacts, retaining original galaxy IDs
   and SHA-256 hashes. Verify coordinate/ID joins, finite values, units, and
   radial conventions. Never execute another worktree's drivers or share
   writable caches. Use `profile_data.load_profiles(version=2)` for new
   isophote/error work through the private data configuration.
2. Trace the original parent selection. Reconcile the legacy `use` sample,
   all-epoch fitting sample, and the parent catalog used by the related
   published profile studies. Report exclusions versus halo mass, stellar
   mass, size, and measured assembly descriptors.
3. Use the approved Paper 1 exception: a quality-certified z=0.4 sample
   without exclusions from unrelated multi-epoch stellar-history criteria.
   Freeze exact measurement/input quality cuts before fitting; compare the
   established strict selection as a sensitivity. Measurement failures are
   not equivalent to unusual physical histories. Other experiments are unchanged.
4. Certify current mass and history definitions per source. Prefer a measured
   history and exact endpoint of the same mass definition. If the available
   history is bound/peak mass while `H` is `M200c`, include that history's
   endpoint in **both** nested models and add history shape only in the
   larger model. Report this extra present-mass control explicitly.
5. Compare own and official DiffMAH histories only after reconciling anchors,
   units, and mass definitions. A z=0-anchored fit is not a neutral replacement
   for a measured z=0.4 history. Do not impose official-catalog matching as an
   unexamined survivor selection.
6. Predeclare physical radial support and resolution masks. The agreed
   stellar-mass anchor is the native outer CoG aperture near 148 kpc;
   retain 100-kpc observables as useful comparisons, not the primary anchor.
   Restrict comparisons to shared physical support; do not treat 103.45 kpc
   as 100 kpc. Retain the measured central mass in annular transformations.
7. Distinguish isophotal density from CoG-derived annular density. The user
   confirms Xu's satellite-particle treatment for this drop; document the
   exact pipeline/array lineage. Do not explain the density difference by
   removed satellites or assume that smooth ICL is removed by clipping.

**Pause condition:** unresolved IDs, mass definitions, exact sample criteria,
or target definition. A good regression score cannot repair these ambiguities.

## 3. Phase B — one matched forward comparison

**Deliverable:** saved out-of-fold predictions for F1/F2, absolute error and
probability-score tables, and first direct QA.

### Representation and models

- Primary halo description: measured MAH on a declared common time grid,
  normalized by its own measured endpoint, plus that endpoint in the control
  model where necessary. Four MAH PCs are a reasonable *predeclared historical
  reference*, not a claim that four are sufficient. Include DiffMAH as a
  matched main input comparison, not merely an incidental sensitivity. Fit
  the same source history/endpoint convention; use sampled fitted curves in
  the same PCA basis as a diagnostic if parameter-based results differ.
- The leading MAH PCs maximize history variance, not stellar predictive
  information. A component-count plateau alone cannot prove that no useful
  low-variance history direction remains. A regularized full-history
  regression is a bounded diagnostic if the measured-PC result is ambiguous;
  it is not an invitation to search many supervised representations.
- Primary stellar description: finite mass anchor plus three normalized-CoG
  PCs as the historical reference. Also predict a short vector of coarse
  nonoverlapping stellar masses directly (initial candidate: `<10`, `10–30`,
  `30–50`, `50–100`, and `100–R_ap kpc`, where the last edge is the exact
  native outer aperture). These are not statistically independent merely
  because their apertures do not overlap.
- Learn every scaler, PCA basis, and feature residualization on training
  galaxies only. Report held-out representation errors separately from
  halo-prediction errors. Reconstruct profiles to compare folds; do not pool
  arbitrarily signed PC coefficients as though they share one basis.
- Add the existing aperture-normalized cubic-logit CoG as one bounded
  analytic alternative after its representation/stability checks. Compare
  final profile predictions, not raw parameter R-squared across unlike
  coordinates. See the September 27 note for stages and extrapolation limits.
- Give the mass-only reference a smooth nonlinear dependence on halo mass.
  Start with a low-complexity spline or quadratic reference, not an arbitrarily
  weak straight line. Added-MAH models must contain that same reference.
- For the strict shape audits, permit nonlinear dependence on stellar mass
  and size, including their interactions. Use a small, declared set of
  quadratic/spline alternatives with regularization chosen in training-only
  inner folds. No broad model search.
- Include a simple additive halo-feature model first and one nonlinear
  sensitivity. Compare fitted flexibility fairly: a larger model should not
  win merely by representing curvature that the reference was forbidden to fit.
- Hold target representation and scatter treatment fixed across each nested
  pair. Allow halo-dependent residual width in the reference too; otherwise
  covariance flexibility can be mistaken for MAH information.

### Scores

For galaxy `i`, form a radial loss with weights fixed before fitting:

\[
 L_i=\sum_j w_j(y_{ij}-\widehat y_{ij})^2,
 \quad \sum_jw_j=1,\qquad
 G=1-\frac{\sum_iL_i^{\rm added}}{\sum_iL_i^{\rm reference}}.
\]

Use equal galaxy weights and a declared radial measure (recommended:
equal weight per interval in log radius). Report RMS in dex **and** `G`.
Also report coarse-annulus errors separately; a cumulative score can hide
outer-mass failures. A positive `G` means reduction in reference residual
squared error, not the fraction of all physical information recovered.

CRPS on a fixed target vector evaluates the predictive distributions;
paired score differences use the same galaxies. Joint log scores require
the same nonsingular coordinates, the same treatment of discarded PCA
variance, and a documented covariance regularization. Do not compare a
singular truncated-PC profile density with a full-dimensional likelihood.
Do not call a regression gain “mutual information in bits.”

### Folds and tuning

- Five outer folds grouped by galaxy/halo identity; all epochs and projections
  of an object stay together. Preserve consistent folds across every model.
- Tune only on training data. Keep the primary component counts fixed; a
  bounded component-count sensitivity must not silently replace the primary.
- Repeated fold assignments diagnose split sensitivity; repetitions are not
  independent universes. Report pooled paired galaxy-level uncertainty, not
  a standard error divided by the number of overlapping CV repetitions.
- If positions are available, add a spatially blocked sensitivity so local
  environment/volume correlations are not hidden by random folds. If not,
  state that limitation rather than claim cosmic-variance uncertainty.
- No new random split of this extensively explored catalog is described as a
  pristine external validation set. Predeclare the new analysis and stop
  tuning against its held-out plots.

## 4. Phase C — separate amplitude, size, and remaining shape

**Deliverable:** the paper's central information-decomposition figure and
a table of absolute residual variances under each control set.

Run S1–S3 on fixed-kpc profiles first. Then repeat the Exp62 scale-free
construction as a sensitivity, using common objects when comparing
`0.7–3 R50` with `0.85–3 R50`. Explicitly separate changing the radial range
from adding/removing galaxies. Omit the fixed half-mass point from scored
information rather than letting a deterministic anchor improve the metric.

Compare PCA results with direct radial or coarse-annular regression. Check
one simple mass+size profile reference and one inner/outer mass-ratio
summary. If these explain the gain, that is a scientific simplification,
not a reason to keep adding PCs until a more elaborate claim survives.

For intuition, match/reweight early and late assemblers to comparable control
distributions and show measured profile residuals. Define assembly quantiles
and matching using training data. Plot overlap and sample counts; do not
extrapolate a response to combinations of halo properties absent in the data.
Treat these panels as associations, not controlled physical interventions.

**Decision:** retain a fine-shape claim only if it survives the declared
flexible controls and is not driven solely by unresolved radii, radial
coverage, or a handful of objects. If not, keep the mass/size result and
explicitly report the shape null or weak effect.

## 5. Phase D — nulls, uncertainty, and influence

**Deliverable:** a compact robustness table and radial uncertainty bands.

### Conditional nulls

Shuffling histories across broad halo-mass bins is a useful gross check but
not an exact null after controlling stellar mass, size, and concentration.
For each comparison, randomize the **whole assembly-feature block** while
preserving its relation to that comparison's controls as closely as possible.
Use local matching/conditional resampling with training-defined controls;
inspect balance, neighbor distances, and support before accepting a null.
Preserve the internal correlations among MAH epochs or coordinates.

Conditional resampling is approximate in continuous high-dimensional control
space. Validate it on synthetic null data with realistic feature correlations,
and compare with a second residual-based check. If balance fails, report
the sensitivity instead of publishing an exact-looking p-value. Refit the
same full analysis under each null, including any allowed model selection.

Start with a small mechanics-only set of randomizations in the sub-minute
test. Proposed full target: 999 randomizations for the primary comparison
if measured cost is acceptable; otherwise predeclare coarser attainable
tail resolution before inspecting significance. Use a single global radial
statistic for the primary claim and simultaneous/max-statistic bands for
exploratory radial locations. Do not select the best radius, epoch, or MAH
summary from many tests and quote its unadjusted significance.

### Uncertainty and outliers

- Bootstrap paired held-out losses by independent galaxy (or spatial block
  where available). Add a bounded refit-bootstrap/split sensitivity to
  distinguish sampling uncertainty from training instability.
- Report all valid objects in the primary score. Alongside it, use a
  predeclared robust loss and an influence analysis that identifies which
  objects dominate the **difference between models**, not just the worst
  individual residuals.
- Compare flagged-history versus unflagged galaxies, plausible measurement
  failures versus unusual but valid profiles, and halo-mass strata. With the
  sample exception now approved, refit both selections with the same protocol.
- Do not remove objects simply because they weaken the thesis. If omission
  changes the sign or practical size of the gain, show it and investigate
  that limited influential subset. Reviewing every intermediate CoG is not
  a prerequisite; measuring their aggregate influence is.

Use confidence intervals and effect sizes, not the inherited Exp62 “5% in
three epochs” threshold. That rule was for a different experiment. A positive
statistically resolved but tiny effect may be scientifically interesting;
whether it is useful observationally requires a separate error comparison.

## 6. Phase E — a small reverse-information test

Optional supporting diagnostic only. It is not required for the thesis or
paper completion, and is not a full history-inference project.

Predeclare two summaries before inspecting their new predictability:

1. the time the measured main branch reaches half its endpoint mass;
2. growth over a fixed recent lookback interval, for example 2 Gyr, subject
   to confirming snapshot support before fitting.

Use the same mass definition and epoch throughout. Report formation time as
cosmic or lookback time explicitly, in Gyr. Compare `H`, `H+S`, `H+S+r`, and
`H+S+r+shape`; include concentration only as a separate simulation-information
reference, not an observationally known quantity. Test simple aperture ratios
against PCA shape. Do not substitute the most predictable epoch after seeing
the result.

Score held-out RMS, residual-variance reduction, rank ordering, and predictive
interval coverage. Show residual distributions and matched measured MAHs for
profile-selected groups. Randomize the added profile information conditional
on controls, analogous to Phase D. No new likelihood or cosmological sampler
is needed: this is a supervised diagnostic within the simulation.

If the incremental gain is weak, the paper can still establish conditional
dependence but must not advertise accurate individual assembly recovery.
Noisy halo-mass proxies and galaxy-only inference are separate, harder tests.

## 7. Phase F — population realism and minimal observational sensitivity

For the statistical emulator, assess jointly sampled mass and shape rather
than supplying a galaxy's measured stellar amplitude. A useful factorization
is

\[
 p(S,q\mid H,A)=p(S\mid H,A)\,p(q\mid S,H,A).
\]

In forward use, **draw S from the first factor** and then draw shape; the
actual galaxy's measured S is never an input. If size is an intermediate
coordinate, draw it conditionally too. Alternatively fit a joint distribution
of the profile coordinates. These are statistical models, not production
physical prescriptions.

State complexity in separate categories: per-galaxy output coordinates,
global regression/scatter coefficients, learned PCA basis, and any parameters
a later observational fit would vary. For a linear model with `p` inputs and
`d` outputs, even the mean alone has `d(p+1)` coefficients; a full constant
residual covariance adds `d(d+1)/2`. Heteroscedastic terms and learned bases
add further fitted structure. Do not call this merely a “three-parameter
model” because it predicts three PCs.

Required QA includes:

- measured/mean/drawn CoGs in halo- and stellar-mass bins;
- density, coarse annulus, and outskirt residuals and their distributions;
- inner–outer stellar-mass and size–mass planes, including conditional widths;
- interval coverage split by halo mass, size, and assembly;
- preserved assembly-conditioned trends in draws, not just global histograms;
- fixed-rule representative best, typical, and worst profiles;
- negative annulus / nonmonotone-draw frequency with no silent clipping.

If Gaussian CoG-PC draws violate positivity appreciably, test one established
positive-annular-mass representation, retaining central mass and exact outward
summation. Re-evaluate every nested reference under the same representation.
Do not start a new analytic-profile grammar to fix a statistical coordinate.

For an intrinsic proof-of-principle paper, mandatory observationally relevant
sensitivities are finite radial cuts, inner resolution, and the available
projection/aperture cross-checks. Three projections of one halo are not three
independent halos. Keep direct 2-D masses as cross-checks, not replacement
modeling targets.

Adding realistic PSF/sky/M/L noise or noisy halo mass is highly valuable but
requires agreed error assumptions or existing mock products. Perturb coupled
quantities together: total mass, size, and normalized shape share measurement
errors. An arbitrary independent Gaussian perturbation is a stress test, not
a survey forecast. This work becomes mandatory if observational recovery is
the paper's headline; otherwise it can remain a clearly labeled extension.

Hydrodynamical halo masses, concentrations, and histories may themselves
respond to baryonic physics. Prediction within TNG-Hydro therefore does not
by itself establish portability to an N-body halo catalog. A matched
dark-matter-only check would address that different question; its inputs
are not assumed to be ready in the current drop.

## 8. Priority, execution, and stopping

| Priority | Work | Why it is needed |
|---|---|---|
| Required before headline numbers | Phase A; matched F1/F2; S1/S2; conditional null and influence checks | Establish that the claimed information is not mass mismatch, baseline weakness, or selection |
| Required for a full-profile emulator result | Held-out compression, S3 attribution, correlated-draw standard QA | Establish that the representation and population prediction are sound |
| Optional supporting evidence | Restricted reverse-summary test | Illustrate conditional association without making individual recovery a goal |
| Valuable extension | Observational perturbations, DMO/hydro-matched inputs, independent simulation/resolution sample | Test practical access and transfer; these need data/assumption decisions |
| Possible appendix/application | Independent epoch fits; population generation in a supplied N-body catalog | Extend the forward demonstration without claiming joint evolution or stellar validation in N-body data |
| Defer | Joint five-epoch model, full merger-tree query, physical deposition engine, broad ML/model search | They are not required to settle the first paper's central question |

Implementation sequence after approval:

1. Recheck branch/worktree ownership; create the separate `ushmr1` repository
   and a feature branch there, following the September 28 launch plan.
2. Freeze science choices and protected data roles. Write small checks for
   joins, fold isolation, mass nesting, normalization, null recovery, and
   cumulative/annular round trips before the driver.
3. Run a measured sub-minute end-to-end mechanics test, including one fit,
   prediction, null check, and QA output. If it exceeds a minute, reduce only
   the operational sample and record timing; do not change the scientific
   sample to meet a speed target.
4. Benchmark full-path cost on the private environment with `uv`; use those
   measurements to schedule work. There is no justified runtime estimate yet.
5. Run the fixed matched comparison and control decomposition. Inspect QA
   before expanding to reverse or transfer tests; report negative results.
6. Freeze the result set, regenerate all paper figures/tables from saved
   predictions, and write a claim-by-claim review. Run focused checks only;
   unrelated missing Exp07 input still blocks repository-wide collection.

**Stop the search** when the prescribed comparisons and sensitivities are
complete, even if the result is modest. Additional model flexibility is
justified only by a diagnosed failure of a declared reference, and requires
a recorded amendment—not by wanting a larger assembly signal.

Do not push, modify shared master, close Exp79, or touch Claude's active work
as part of this study. The September 28 planning merge is user-authorized,
but still requires coordination with Claude before changing its checkout.
