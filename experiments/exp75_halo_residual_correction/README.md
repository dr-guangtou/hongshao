# Exp75 — can halo inputs predict the deposition model's residuals?

## Final disposition (2026-09-09)

Closed by user decision, with no additional science runs required. Retain the
original and measured-MAH results as diagnostic benchmarks, not a production
residual-correction framework. Halo properties predict some of the physical
model's remaining errors, but a flexible TNG300-calibrated correction does
not expose those missing relationships through a compact, interpretable set
of parameters for observations to constrain. The direct model remains an
empirical comparator, not a proven statistical upper limit. Historical scores,
QA, and sample membership are unchanged. The next direction is a compact
physical prescription shared across epochs, evaluated with CoGs and annular
masses together; no unfinished follow-up is required to close this experiment.

Status: original discovery completed, 2026-09-05; measured-history continuation
completed, 2026-09-08. Neither result is a production qualification.
The September 8 continuation is predeclared in `OVERNIGHT.md`; its outputs
and figures are separate from every original artifact. The original protocol
below is retained as a historical record, including its original scope limits.

## September 8 continuation: input and numerical checks

The physical baseline predicts the stellar profile by integrating deposition
along a halo history. The hybrid adds four smooth profile corrections
predicted from halo properties. The direct model predicts five profile
coordinates from halos without the deposition equation. All three predict
stellar amplitude; none receives an evaluated galaxy's stellar mass or size.

This continuation asks whether the correction remains useful after the
physical baseline reads measured MAHs. It uses the committed Exp74 input
implementation at `360d2d6`, merged into this isolated branch; no active
Claude code or full-sample stellar-fit parameters are imported. The saved
halo-only histories are copied into a hash-verified private snapshot.

- The original 842 discovery IDs, CoGs, folds, and fitting/calibration/
  evaluation roles are unchanged. Exp67 selection and validation have no role.
- Every galaxy has a measured final halo mass and a usable measured history.
  Earlier target epochs have 840, 839, 840, and 838 valid catalog halo masses.
  Missing snapshots are not missing histories: keep the galaxies, preserve
  Exp74 interpolation, and impute missing regression inputs from training only.
- The official-input path reproduces every saved original held-out baseline
  prediction exactly. Thus bringing in the new interface did not change the
  historical reference.
- The supplied pre-epoch DiffMAH arm is invalid as supplied, not a negative
  scientific result. At z=2, 21 of 842 curves have negative halo growth inside
  the integration domain; eight yield nonpositive stellar profiles with frozen
  original parameters. Stop this arm without clipping growth, removing galaxies,
  or redesigning the halo fit. See `outputs/continuation/history_audit.json`
  and `figures/continuation/pre_epoch_input_audit.png` (PDF alongside).
- Measured interpolation reproduces its knots to 1.78e-15 dex. Its smallest
  sampled logarithmic derivative is -4.44e-16, numerical roundoff around zero;
  the derivative tolerance is 1e-12 and the curves themselves are unchanged.
- The 30-galaxy operational path took 4.59 seconds, below the 60-second limit.
  It includes capped fitting, both feature sets, predictions, serialization,
  scores and a PNG/PDF individual-CoG figure; its scores are not science.
- All five 150-galaxy pilot folds passed. Their integration differences after
  doubling the quadrature resolution are 0.000385--0.000473 dex, below the
  0.001 dex tolerance. Both initializations converge within the allowed 1%
  training-loss difference. Fold wall times range from 31.89 to 50.94 seconds.
- Some trial parameters make the inherited truncated-profile normalization
  undefined. Reject and record such optimizer trials with infinite loss;
  final predictions must remain finite. This does not change bounds or the
  original equal-weight log-CoG fitting objective. It is not a repair of the
  inherited profile table over its entire parameter domain.

All five full-discovery baseline fits also passed: two converged starts within
1% training loss, and fitted/full integration differences of 0.000426--0.000469
dex, below the 0.001 dex requirement. Recorded fold wall times were
193.06--266.87 seconds, one process and one thread at a time.

## September 8 result and decision

The halo-predicted correction remains useful with measured MAHs, but it does
not yet provide a production-quality profile model. All comparisons below
use the same 842 discovery galaxies, held out by galaxy across five epochs.
The score is the mean of each galaxy-epoch's radial log10 CoG RMS, in dex;
smaller is better. It is not a probabilistic calibration score.

| Prediction | Mean radial CoG error (dex) | Meaning |
| --- | ---: | --- |
| Measured-history physical baseline | 0.11762 | Matched reference without correction |
| Baseline plus global intercept correction | 0.11608 | Global recalibration alone helps little |
| Baseline plus original halo-descriptor correction | 0.10957 | Correction still helps after the history swap |
| Baseline plus measured-mass/concentration correction | 0.10771 | Best measured-history hybrid |
| Baseline plus final-mass-only correction | 0.11516 | Final mass alone does not explain the gain |
| Baseline plus shuffled-history correction | 0.11599 | Assembly information contributes to the gain |
| Direct measured-mass/concentration prediction | 0.10815 | Similar pooled score, worse at z=2 |

The measured-feature hybrid improves on its matched physical baseline by
8.42%, with a paired galaxy-bootstrap 95% interval of 7.05--9.88%. It passes
all five original discovery criteria, including improvement over the intercept
and the epoch and halo-bin safeguards. These criteria do not test all the
observables HongShao must eventually predict.

This is **not** an 8.42% advance over the original Exp75 hybrid. That model's
error was 0.10843 dex: the new hybrid is only 0.66% better, with a paired
95% interval from 0.98% worse to 2.26% better. The measured physical baseline
itself is 2.97% worse than the old baseline's 0.11422 dex under this particular
equal-log-CoG loss. Exp74 used another objective and population, and official
DiffMAH describes another halo-mass definition. These results support input
flexibility and matched tests, not a universal accuracy ordering of inputs.

The full standard QA battery was generated and visually inspected for all
three models (36 figures). See `QA.md` for captions and the figure index.

- At z=2, median log R50 prediction/data improves from +0.13363 dex for the
  baseline to -0.00960 dex for the hybrid; median log R90 improves from
  +0.11931 to +0.00580 dex. The correction substantially fixes the typical
  size offset, although individual profiles can still be poor.
- At z=2, the relation between log stellar mass in 50--100 kpc and inside
  30 kpc remains too steep: its fitted slope is 2.118 for the hybrid,
  compared with 2.462 for the baseline and 1.565 in the data. Exact apertures
  are linearly interpolated in radius and cumulative mass.
- For the outermost 100--148.22 kpc envelope at z=2, median log mass
  prediction/data falls from +0.34960 dex to +0.27343 dex. This is an
  improvement, but still corresponds to 88% too much envelope mass for the
  median positive-mass comparison (829 of 842 galaxies). No value at the
  grid endpoint is mislabeled as a 150 kpc measurement.
- The direct model has a better outer-mass plane but its z=2 radial CoG
  error is 0.14417 dex, worse than the hybrid's 0.12718 dex and the
  baseline's 0.13800 dex. A better-looking plane is not a complete solution.
- All three are point predictions, not sampled galaxy populations. Their
  narrower relations cannot alone disqualify their conditional means; a
  future probabilistic layer must separately reproduce widths and epoch
  coherence. No such layer is fitted or qualified here.

Fitting the four corrections directly to every true stellar CoG gives a
0.00693 dex mean radial error, compared with 0.10771 dex when predicting
them from halos. This diagnostic uses stellar labels and is never a halo
prediction. It shows that the four-coordinate family can represent much
more of the cumulative shape than the halo map predicts. However, its
outermost-envelope median error at z=2 is still +0.40004 dex (151% too much
mass). A tiny cumulative error does not guarantee an accurate difference
between two large cumulative masses. Before adding more halo predictors or
profile coordinates, test whether the loss is asking for the right thing.

The future-growth check compares the model response with the measured
response, not with a blanket zero target. At z=2, the hybrid's residual
response of log stellar mass inside 100 kpc to log later M200c growth is
-0.04374 dex/dex, with a pointwise 95% interval [-0.08851, -0.00159], after
controlling quadratically for current M200c (838 valid galaxies). The baseline
residual is -0.02548 dex/dex with an interval including zero. A similarly
marginal under-response occurs at z=0.7. These intervals condition on trained
models, do not correct for multiple epochs, and are not proof of forbidden
inputs. They are an additional reason to require conditional QA in follow-up.

Decision: close Exp75 as a useful discovery result, not production adoption.
Agree with Exp74's application contract: full MAHs, peak/final halo masses
and DiffMAH parameters are legitimate; target stellar masses are not inputs.
The supplied pre-epoch arm remains an input-validity failure, not a scientific
null. The next isolated question is whether adding an annular-mass term to
the correction loss improves outer masses without sacrificing CoGs and sizes;
keep the baseline, halo features, radial family and discovery roles fixed.

## Continuation reproducibility and records

- `continuation.py`: private snapshots, frozen-input swap, operational gate,
  pilot and discovery fold fits; checkpoint files are never overwritten.
- `history_audit.py`: supplied pre-epoch negative-growth audit and galaxy IDs.
  This is the authoritative validity report; the first `frozen_swap.json`
  retains NaN scores for invalid profiles as a historical failed check.
- `continuation_report.py`: assembled held-out comparisons, direct-fit
  representation diagnostic, exact annuli, conditional checks and full QA.
- `continuation_figures.py`: matched aperture planes, fixed and most-changed
  individuals, representation and future-growth figures.
- `outputs/continuation/summary.json`: numerical results and paired intervals.
  Fold JSON/NPZ records contain immutable inputs, roles, choices, convergence,
  runtime, fitting-time git SHA and code hashes. `manifest.json` in that
  directory inventories the final artifacts separately from the old manifest.
- Focused checks: `uv run --no-sync python -m unittest discover -s
  experiments/exp75_halo_residual_correction -p 'test_*.py'`; do not collect
  repository-wide pytest, blocked by the unrelated missing Exp07 input.
- Report commands: `uv run --no-sync python
  experiments/exp75_halo_residual_correction/continuation_report.py --help`.
  Use a fresh isolated output location to reproduce completed products;
  existing science checkpoints intentionally refuse replacement. The recorded
  overnight deadline also intentionally stops a later fitting invocation.

The original 62 output/figure files are checked against their original hashes
before archival. Selection and validation artifacts remain untouched, no
library default changes, and Claude's worktrees remain outside this work.

## Original September 4--5 protocol and result

The protocol below was committed before driver implementation or fitting
(2026-09-04, `aebcc473639fe6ca8f976f5d73744e30b0c43854`).
Base: master `6bc2ecc326d73cff11d6da8feb941a8a3daa865d`.
Branch: `exp75-halo-residual-correction`. No Exp74 code or fitted artifacts.

## Question and scope

Predict a small correction to the two-channel deposition model from halo
properties alone. Separate missing information from information not exploited
by the deposition equation. This is a phenomenological prediction experiment,
not a claim that the correction describes a physical deposition mechanism.
No symbolic search, stochastic-layer development, merger-tree query, library
promotion, merge or push is part of this experiment.

Full halo histories and final/peak halo masses are legitimate inputs. A target
galaxy's measured stellar mass or size at any epoch is never an inference
input. Amplitude is predicted, never normalized to the target. Training labels
and population-quality selection are distinct from inference inputs.

## Isolation and sample roles

All code runs from this worktree, with a private environment and output/cache
directories. Copy only named, existing input products; verify source hashes
before and after extraction and store hashes with the snapshot. Never import
code from the active worktree, launch its drivers, or modify its files.

Use ONLY Exp67's existing discovery galaxy IDs, read from the preserved master
discovery manifest/archive. Exp67 selection and validation have NO ROLE in
Exp75 and are excluded before sample-quality tests or scientific calculations.
Use the intersection with the established halo population. Apply the existing
`selection.fitting_sample_mask` to this discovery population and record counts.
No halo-mass completeness cut in fitting. Missing concentration is imputed
from training data, not grounds for silently removing a galaxy. This is a
discovery-restricted experiment, not validation on a new unseen population.

Target: `profile_data.load_profiles(...)["cog_provided"]` on the original
24-point semi-major-axis grid, five epochs, h-free solar masses. Verify its
identity with the matching population CoGs. Halo inputs: official DiffMAH's
four parameters and log concentration at z=0.4. The sole feature extension is
the five-epoch concentration history (instead of z=0.4 concentration alone).
The measured-MAH versus DiffMAH comparison remains Exp74's question.

Assign five folds by seed 75, stratified by final halo mass; keep every epoch
of a galaxy together. For rotation f: fold f is evaluation, fold (f+1)%5 is
correction calibration, and the remaining three folds fit the deposition
baseline. Thus correction targets are residuals on galaxies the baseline did
not fit. Inner two-way splits of the calibration fold choose regularization
and features without reading evaluation labels. The direct reference trains
on the same baseline-plus-calibration union (80%); it has more direct access
to labels than the correction's calibration map, a conservative comparison
for the hybrid, explicitly reported. All methods share evaluation IDs.

## Frozen references and fitting objectives

1. Baseline: Exp63's twelve-parameter, compact-in-physical-kpc two-channel
   family, original bounds and original DiffMAH input. Refit from two
   predeclared, data-independent starts on baseline-training galaxies only.
   Do NOT initialize with a full-sample fitted artifact. Objective: mean square
   log10 CoG error, equal galaxy/epoch/radius weight. This is an explicit
   matched-loss rebaseline of the FAMILY, not a reproduction of Exp63's
   four-term optimum. The loss is a distance, not a likelihood; correlated
   cumulative radii are equally weighted, not claimed independent errors.
2. Baseline plus an intercept-only profile correction: separates global
   calibration from galaxy-specific halo information.
3. Baseline plus halo-conditioned correction: the candidate below.
4. Direct statistical reference: Exp51's central-censored five-coordinate
   decoder, linear or full degree-two ridge map, same available halo inputs.
   Report decoder-only representation error separately.
5. Correction controls: final-mass-only; history/concentration feature vectors
   shuffled jointly within training-defined final-mass quintiles, separately
   within calibration and evaluation sets, with the correction refitted.
   This tests the correction's incremental assembly use, not all assembly
   information already present in the deposition baseline.

Baseline optimizer: bounded least squares, scaled by parameter ranges;
scientific cap 300 residual evaluations per start, tolerances 1e-6. Starts:
`[-3,-0.5,0.3,0,12.5,0.7,0.4,-0.5,-0.8,-0.3,1,0.8]` and
`[-3.5,-0.8,0.5,0.1,13,1,0.7,-0.8,-0.6,-0.5,0.7,1]`, clipped to bounds.
Record convergence, calls, boundary positions and loss difference. Require two
successful starts within 1% in training loss before calling a scientific
baseline comparison settled; a failure is an optimizer blocker, not a null
scientific result. Verify fitted versus full integration within 0.001 dex.

## Correction: amplitude plus three radial directions

Write each baseline CoG as nonnegative shell masses, including the central
aperture. Multiply shells by exp of a smooth radial tilt, then renormalize
to the BASELINE total times a separately predicted amplitude factor. This
normalization never involves the galaxy's true total.

Use three fixed radial directions, linear/quadratic/cubic Legendre terms in
log shell radius mapped to [-1,1]. Four correction numbers per galaxy-epoch:
log10 total-mass change and the three tilt coefficients. Zero correction must
reproduce the baseline exactly; all corrected CoGs remain nondecreasing.
This changes central fraction and inner/outer structure without introducing
24 independent radial adjustments. Fit the four target coefficients on
CALIBRATION galaxies only by log-CoG least squares; those fitted individual
coefficients are never supplied for an evaluation galaxy.

Map coefficients from halo features using ridge regression, separately per
epoch. Candidate map: linear or full degree two; ridge penalties
{0.01, 0.1, 1, 10, 100} applied to the mean-square standardized-target objective.
Imputation/scaling and feature expansion are fit on training rows only.
Choose by inner-held-out log-CoG error; within 1% prefer linear, fewer halo
features, then stronger regularization. No post-result feature expansion.

This per-epoch correction makes no claim of interpolation or a consistent
deposition history. Temporal behavior is audited, but Option 2's coherent
generative model is deferred. Do not score mean populations as stochastic draws.

## Execution gates and scientific judgment

Operational sample: 30 mass-stratified admitted discovery galaxies, all five
epochs. Synthetic recovery, one baseline fit capped at THREE evaluations,
calibration/evaluation split, correction fit, direct reference, poison and
nesting checks, serialization and a PNG/PDF direct-fit figure must together
finish in strictly less than 60 measured seconds, including imports/data read.
This checks execution only; capped fits carry NO accuracy conclusion. Record
wall time and do not turn a runtime failure into a scientific null.

Pilot: 150 mass-stratified admitted discovery galaxies, all five folds, two
baseline starts, full numerical/optimization gates. No significance or
production claim. Proceed to all admitted discovery galaxies only if the
baselines converge and numerical gates pass. No protected data are unlocked.

Primary discovery statistic: pooled held-out mean per-galaxy log-CoG RMS,
against the matched baseline and against intercept-only correction. Also
report each epoch, central/aperture/annular masses, p90 per-object error,
mass-binned radial biases, shell-density residuals and coordinate extremes.
Use 1000 galaxy bootstrap resamples (all epochs together) for paired 95%
intervals. A candidate earns further work only if the pooled RMS improves by
at least 5% versus baseline, the paired improvement interval excludes zero,
it improves over intercept-only, no epoch RMS worsens by >2%, and no maximum
absolute halo-mass-tercile median radial bias worsens by >0.01 dex. These
thresholds are decision tolerances, not expected gains. Corrected mean sizes
need not reproduce population widths; that is Option 2's job.

Before calling it interesting, inspect average CoGs and residuals in halo-mass
bins, inner/outer stellar-mass planes, and best/typical/worst individual CoGs.
Then run the full standard QA (apertures, annuli, outskirts, density, sizes,
mass/size planes, CDFs, cross-epoch mass and size changes), distinguishing means
from draws. Every figure gets a self-contained caption and PNG/PDF companions.
All results retain indices, fold roles, model settings, source hashes, git SHA
and measured runtime. No production adoption from this discovery experiment.

## Discovery result and decision

We fit a second statistical model to the deposition model's leftover profile
errors, using halo inputs alone to predict amplitude and three radial changes.
On 842 discovery galaxies evaluated once each, this reduces mean per-galaxy
log-CoG RMS from 0.11422 dex to 0.10843 dex: a 5.074% improvement over the
matched-loss deposition baseline. The paired 1000-galaxy-bootstrap 95% interval
is 4.141–6.006%. It excludes no improvement, but does not establish that the
population gain exceeds 5%. This is uncertainty conditional on these fitted
fold models, not a bootstrap of model training or a fresh validation sample.

All five predeclared discovery criteria pass. This supports predictable
halo-dependent residual structure beyond a global recalibration. It does NOT
make the current correction a production model: full QA exposes outer-mass
biases and inadequate population diversity. Preserve the frozen result;
predeclare any next experiment rather than retuning this discovery comparison.

### Matched held-out comparisons

Each entry is the mean radial log10-mass RMS per galaxy-epoch, pooled equally
over the same galaxies and five epochs; lower is better. The deposition family
was newly fitted under the same log-CoG objective, so these numbers must NOT
be quoted as improvements over the original published Exp63 optimum.

| Prediction | Held-out CoG RMS (dex) | Interpretation versus the baseline |
| --- | ---: | --- |
| Deposition baseline | 0.11422 | Reference |
| Baseline plus global intercept correction | 0.11430 | No improvement |
| Baseline plus final-mass-only correction | 0.11355 | Small improvement |
| Baseline plus mass-bin-shuffled assembly correction | 0.11416 | Essentially unchanged |
| Baseline plus halo-conditioned correction | 0.10843 | 5.074% better |
| Direct five-coordinate statistical predictor | 0.11007 | Better than baseline; slightly worse pooled error than the hybrid |

The direct predictor remains better at z=0.4 and 0.7 and has smaller maximum
mass-binned radial bias (0.03556 dex, versus hybrid 0.05792 dex and baseline
0.08654 dex). It is not dominated by the hybrid. It trains on 80% of galaxies,
whereas the correction map trains on 20% and its baseline on a separate 60%.
Do not infer a universal architectural ranking from this asymmetric comparison.

| Redshift | Baseline RMS (dex) | Corrected RMS (dex) | Improvement |
| --- | ---: | ---: | ---: |
| 0.4 | 0.10790 | 0.10171 | 5.74% |
| 0.7 | 0.10681 | 0.10214 | 4.37% |
| 1.0 | 0.10782 | 0.10383 | 3.70% |
| 1.5 | 0.11458 | 0.10834 | 5.45% |
| 2.0 | 0.13400 | 0.12612 | 5.88% |

Every fold's inner calibration chooses a linear map with the five-epoch
concentration history and ridge penalty 1. This is consistent selection, not
an independent measurement of concentration history's marginal contribution.
The direct comparator always chooses quadratic features with the z=0.4
concentration and penalty 0.01. Its decoder alone, supplied with measured
stellar coordinates for this diagnostic only, has 0.00455 dex mean radial RMS
against measured CoGs. That is a representation check, NOT a halo prediction.

### Full QA changes the production judgment

- At z=2 the median log half-mass-radius error improves from +0.07141 dex
  (baseline relative to measured radii) to -0.00145 dex (hybrid). However,
  the median log R90 error worsens from -0.05885 dex to +0.06921 dex.
  Better half-mass radii do not ensure better outer profiles.
- At z=2 the 50–100 kpc annular-mass RMS improves from 0.55341 dex to
  0.40618 dex, but the median log mass bias between 100 kpc and the grid edge
  worsens from -0.09470 dex to +0.47991 dex. The latter uses the 829 galaxies
  with positive measured envelope mass; nearly flat CoG tails make fractional
  and log errors especially sensitive. No new density-measurement mask was
  invented. This is a CoG-derived envelope warning, not a claim about the
  independently measured isophotal density.
- The standard R20/R50/R80 offset check passes 13/15 hybrid epoch-size cases,
  versus 14/15 for the baseline and 15/15 for the direct predictor. Hybrid
  R20 fails at z=1.5 and 2.0, but 74.94% and 85.39% of the corresponding
  measured R20 values lie below the 2 kpc grid and use the inherited log-log
  extrapolation. This is a resolution-limited warning, not a resolved core test.
- None of the three point predictors passes any of the 15 halo-conditioned
  size-width cases. This diagnoses the absence of a generative scatter layer;
  it is not a rejection of a conditional point prediction for being narrow.
- For total stellar masses at z=2 versus z=0.4, rank correlation is 0.80576
  for the hybrid versus 0.59176 in the data (baseline 0.84383). Cross-epoch
  diversity remains missing. No interpolation or continuous-history claim
  is made for five independently corrected epochs.

The next useful step is a separately predeclared, small test of whether the
correction can keep its cumulative-mass gain while controlling outer annular
mass and size biases. Keep the direct statistical predictor as a reference;
do not spend protected validation data on the present outer-profile defect.
Only after this mean-model choice should a radius-resolved, temporally coherent
scatter layer be calibrated. Measured-MAH substitution remains Exp74's work.

### Execution, safeguards and provenance

The discovery membership has 1200 galaxies, of which 850 overlap the established
halo population; the existing quality selection removes 8, leaving 842.
No selection/validation galaxy receives a role. Missing concentration is
training-imputed. All science uses a private hash-verified input snapshot;
source files were opened read-only and checked unchanged before/after extraction.

The complete 30-galaxy operational gate took 5.741 seconds against a strict
60-second limit. All five 150-galaxy pilot rotations passed. All five full
discovery rotations passed, taking 261.260 seconds summed over the serial
fold commands. Both baseline starts converged in every rotation; the largest
training-loss disagreement was 0.6712%, below the 1% tolerance. The maximum
integration-resolution difference was 0.0000380 dex, below 0.001 dex.
The deposition transition-width parameter reaches its lower bound in several
fits: convergence does not establish physical identifiability or a global optimum.
Full standard QA of baseline, direct and hybrid took 53.968 seconds, separately
from the discovery fitting commands and the three overview figures.

Ten focused tests pass, including synthetic coefficient recovery, exact
zero-correction nesting, monotonicity, fold separation, training-only
imputation, scientific-gate tests, and an end-to-end evaluation-label poison
test: multiplying held-out stellar CoGs by 100 changes none of the six
predictions. The predicted correction coordinates reconstructed from the saved
model CoGs reproduce them within 4.34e-13 dex. Coordinate extrema are saved
in `outputs/discovery_standard_qa.json`; no evaluation stellar coordinate
enters a halo prediction. No repository-wide pytest collection was attempted.

The scientific fit code is committed in `acd67aca4ab1b461420d6bcfc82287a01d9ed4f6`;
later changes to `run.py` only format it and correct an overview caption.
The original fold-zero discovery figure says "provisional pilot fits" because
of that inherited caption error; fold JSON and NPZ stage/role records are
authoritative. The final combined figures have correct discovery labels.
The overview mass plane labels use the actual nearest grid radii (10.25,
52.30 and 103.45 kpc), not nominal 10/50/100 kpc; standard QA separately
interpolates to its named apertures. Unmeasured 100–150 kpc annuli are null,
not zero, because the original grid ends at 148.220 kpc.

Run scripts from this worktree with its private `uv` environment and one BLAS
thread. Sequence: `prepare.py` once with explicit read-only source/archive
roots; `run.py gate`; `run.py pilot --rotation N` for N=0..4;
`run.py discovery --rotation N`; `report.py discovery`; `standard_qa.py`.
Existing input/fold/standard-QA outputs refuse overwrite. The report may be
regenerated from frozen predictions. `report.write_manifest()` inventories
final code, records and figures; fold JSON preserves fitting-time hashes and
settings. All outputs/figures remain gitignored and local; no merge or push.

### Figure guide

All figures compare identical held-out discovery galaxies and have PNG/PDF
companions. The three `figures/discovery_*` overview figures compare baseline,
hybrid and direct predictions: halo-binned means/residuals, stellar-mass planes,
and best/typical/worst individual profiles. They establish the modest mean
improvement and expose unrecovered individual departures.

The twelve `figures/standard_qa/qa_*_exp75_hybrid` figures show:

- `mass_kpc_aper`: aperture masses; residual individual scatter remains.
- `mass_kpc_diff`: annuli/envelopes; outer-envelope biases remain large.
- `mass_Re_aper`: apertures in each profile's own half-mass-radius units;
  amplitude scatter remains despite small median biases.
- `mass_Re_diff`: size-relative annuli/envelopes; shape errors are not removed.
- `planes`: inner/outer mass relations; slopes and widths remain mismatched.
- `bins`: halo-mass-binned CoGs; the model-input-conditioned diagnostic.
- `bins_ms`: stellar-mass-binned CoGs; regression to the mean affects the view.
- `dens`: derivatives of measured/model CoGs, NOT independent isophotal
  densities; cumulative-mass agreement hides outer-slope differences.
- `cases`: individual best/worst maximum fractional errors; severe failures
  remain. `idx` here is the local 842-row position, unlike original IDs in
  the overview figure.
- `growth`: same galaxies across epochs; model histories are too correlated.
- `size`: mass–size planes; accurate R50 does not ensure accurate R90.
- `cdf`: mass-distribution CDF residuals; annular distributions still differ.

Inherited dotted "amplitude-pinned" overlays in the binned QA only remove
amplitude for a shape diagnostic. They are never predictions or inputs to the
reported primary scores. Width/CDF/plane comparisons of point predictions are
descriptive, not tests of a sampled population. No scatter draws were produced.
