# Exp75 — can halo inputs predict the deposition model's residuals?

Status: PREDECLARED, before driver implementation or fitting (2026-09-04).
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
