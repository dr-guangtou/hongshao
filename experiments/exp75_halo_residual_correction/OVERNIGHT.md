# Exp75 continuation: measured halo histories

Predeclared 2026-09-08 before implementation or new fitting. User approved
the overnight plan. Deadline: 2026-09-07 23:00 UTC / September 8 07:00 Shanghai.
Science stops at 22:00 UTC; closeout finishes by 22:55 UTC. Platform limits
are an earlier stop. Do not leave owned science processes past the cutoff.

## Question and isolation

Does the halo-conditioned correction still help when the deposition model
reads measured MAHs? Separate input-history errors, profile representation,
and prediction of the correction from halos. Original outputs are immutable.
Use new `outputs/continuation` and `figures/continuation` directories.
Work only on `exp75-halo-residual-correction`; protect Claude's
`adopt-measured-input` and `exp76-growth-rate-split`, and recheck ownership
before creating branches or integrating results. Use pinned committed Exp74
code, private input snapshots, environment and caches, serial one-thread jobs.
Never import active-worktree code or modify its artifacts. No library promotion.

## Frozen comparisons

Keep the original 842 discovery IDs, five galaxy folds (seed 75), three folds
for baseline fitting, one for correction calibration, one for evaluation.
All epochs of each galaxy stay together. Selection and validation have no role.
Keep original measured CoGs and sample membership; no new quality selection.
Full halo history is legitimate; evaluated stellar masses never enter a
prediction, initialization, normalization or hyperparameter selection.

Inputs: official DiffMAH (saved reference), measured running-peak M200c, and
pre-epoch DiffMAH fitted to the same M200c. Reuse Exp74's local interpolation,
running peak, valid-snapshot rule and early power-law extension, and its
halo-only pre-epoch fits, explicitly aligned by galaxy ID. The official curve
was fitted to SubhaloMass; do not attribute all differences to anchoring.
Verify units, knot reproduction, nonnegative growth, coverage and extrapolation.
Reject misalignment or missing histories rather than silently changing sample.

Input audit, before fitting: 2/3/2/4 of the 842 galaxies have no valid M200c
at z=.7/1/1.5/2, respectively; all have final masses. A missing snapshot is
not a missing history. Preserve every galaxy and Exp74 interpolation across
missing knots. Leave missing measured regression features as NaN for the
existing training-only imputation; condition QA on available measured masses
and report counts. Require every galaxy to have a usable measured history.

Numerical audit before scientific refitting: the supplied pre-epoch DiffMAH
fits can have negative derivatives between measurements and yield nonpositive
CoGs in the frozen swap. Save affected IDs and direct history plots; stop that
input arm as invalid as supplied. Do not clip accretion or change its sample.
Measured interpolation has only floating-point endpoint derivatives as small
as -4.44e-16 in dlogM/dlogt; allow 1e-12 numerical tolerance without altering it.
The measured pilot also reached an undefined inherited profile-normalization
trial. Reject such optimizer trials with infinite loss and record their
parameters; require finite final profiles. Bounds and scientific loss remain
unchanged. The first interrupted attempt had no completed fit checkpoint.

For each new input, first evaluate saved original fold parameters without
refitting. Then refit Exp75's twelve-parameter deposition model under its
original mean-square log-CoG loss, bounds, two data-independent starts,
300 evaluations/start and 1e-6 tolerances. Require both starts converged with
training losses within 1%, and integration agreement <=0.001 dex. One retry
at 600 evaluations/start is allowed. Integration resolution can be doubled
up to three times, separately measured; numerical failure is not a science null.

Refit correction with unchanged four-coordinate family and original halo
features, with intercept, mass-only and within-mass-bin-shuffled controls.
Then separately replace the four DiffMAH descriptors with five epoch log halo
masses; append concentration at z=0.4 or all five epochs, selected within
calibration. Apply the identical feature candidates to hybrid and direct.
Retain linear/quadratic maps, penalties .01/.1/1/10/100, two-way inner splits
and the original 1% simplicity preference. Direct trains on the 80% union;
hybrid correction on 20%, baseline on 60%. Report this asymmetry.

Representation diagnostic: directly fit the four correction numbers to each
discovery galaxy-epoch under the original log-CoG loss, compare its cumulative
AND annular errors with the halo-predicted correction. Direct-model measured
coordinates are a separate reconstruction check. Neither is a halo prediction.
Direct fits may be used for diagnostics on evaluation galaxies, never training.

## Gates, outputs and decision

New full operational path: 30 mass-stratified discovery galaxies, capped
baseline fits, snapshot read, prediction, correction, score, serialization and
PNG/PDF figure under 60 measured seconds. Then 150-galaxy five-fold pilot
passes numerical checks before full discovery. Focused checks only (unrelated
Exp07 repository-wide collection blocker remains outside scope).

Tests: official-input parity, row/fold separation, training-only feature
processing, evaluation-stellar-label poisoning, exact zero nesting,
monotonicity, synthetic recovery, interpolation/integration, checkpoints and
owned-process deadline handling. Save failures and interrupted status.

Retain original discovery criteria: >=5% pooled mean galaxy-epoch radial
log-CoG RMS improvement against its matched baseline, positive paired 1000
galaxy-bootstrap 95% interval, better than intercept correction, no epoch >2%
worse, maximum absolute halo-tercile median radial bias no >0.01 dex worse.
These do not constitute production qualification. Flag outer masses, sizes
and conditional relations separately. Bootstrap keeps all epochs together;
intervals condition on trained fold models and do not resample training.

Use measured epoch halo masses for new conditional QA; retain old bins for
historical comparisons. Measure residual stellar mass versus future halo
growth at fixed epoch mass with bootstrap uncertainty; target agreement with
data, not a universal ban on future information. Report all aperture/annular/
envelope masses, CoGs, CoG-derived densities, sizes, CDFs, mass planes and
cross-epoch behavior. Generate the full standard figure set for baseline,
hybrid AND direct, with matched comparisons and explicit model definitions.
Show representative and most improved/worsened galaxies using fixed IDs.
Use exact named-aperture interpolation. Flag out-of-grid masses and
extrapolated sizes; never present a point predictor as a sampled population.

Record IDs, roles, hashes, settings, convergence and measured wall time per
checkpoint. Show every new figure with full path and PDF companion. Captions
state what, reference and interpretation. Final result separates radial-family
failure, halo-mapping failure, history-input compensation and inconclusive fits.

## Closeout and conditional follow-up

Close Exp75 first: focused checks, scientific README/lessons/roadmap, commit,
hash-verified non-overwriting artifact copy to master, no-ff merge through a
private integration worktree, normal push only, verify remote, then journal
and one-sentence append in Obsidian vault `wensai`. Preserve other agents'
changes and retry integration on a new remote head; unresolved conflicts stop
the merge, never force it. Retain worktrees/branches for recovery.

Only if Exp75 closes before 04:30 Shanghai and a measured pilot fits the
remaining window: create one new uniquely numbered outer-profile-correction
experiment after checking local/remote branches and directories. Do not
duplicate Claude's growth-rate work. Freeze the measured baseline and radial
family; compare original loss with log-CoG MSE + lambda * annular MSE for
lambda=.25,1 (reference 0). Annular edges: 0,2,10,30,50,100,grid endpoint.
Annular residual transform: asinh(M_ann/(.001*true_total))/ln(10); the true
total scales training losses only. Choose lambda and maps inside calibration.
No symbolic grammar, new data queries, scatter layer or production adoption.

Follow-up interest requires >=10% improvement in held-out outer-annular RMS
(mass residual divided by true total, equally weighting the last two annuli),
positive paired improvement interval, <=2% worse pooled CoG RMS, and <=.01
dex worsening in maximum halo-binned radial bias and absolute median R50/R80/
R90 bias. Report positive-mass log errors and zero counts too. A significant
increase in residual future-growth dependence blocks advancement. Same full
QA and operational gate. If radial representation fails, close diagnostically.
Exploratory reuse of discovery galaxies is not independent validation.
