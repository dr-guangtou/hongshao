# Exp77 — does explicitly fitting annular masses improve the halo correction?

Status: discovery completed, QA inspected; final attribution controls and
closeout in progress. The declared candidate fails the size and conditional
growth safeguards; no production adoption or replacement of that verdict.

Predeclared 2026-09-08 before driver implementation or fitting. Base master
`89bf4544e347d6fdc6a8f9d55734a21e98c345a1`; isolated branch
`exp77_annular_correction_loss`. Exp75 is merged, pushed and journaled.
Claude's measured-input adoption and Exp76 growth-rate work remain untouched.

## Question and scope

Keep Exp75's measured-history deposition baseline and four correction
coordinates fixed. Change only what the correction fit is asked to minimize:
add annular stellar masses to cumulative stellar masses. Small cumulative
errors can conceal large errors in their outer differences. This is a test
of the correction loss, not a new physical model or a production adoption.
No new halo inputs, symbolic families, stochastic model, data query, or
changes to Claude's growth-dependent deposition split.

## Inputs and sample roles

Copy only named Exp75 artifacts from its closed worktree, verifying hashes
against its final manifest and before/after copying. Read the original and
continuation input snapshots and five settled measured-history fold archives
and records. Use the original 842 discovery IDs and original five folds;
60% baseline fitting, 20% correction calibration, 20% evaluation, all epochs
of a galaxy together. Baseline predictions are frozen, never refitted here.
Selection and validation have no role. The 30- and 150-galaxy operational
and pilot subsets preserve original fold labels and use only those rows'
calibration labels. The inherited baseline has seen other baseline-training
galaxies; this is not a new small-sample baseline fit.

The predictor sees halo properties only. Use each fold's measured-feature
configuration selected within Exp75 calibration: five measured log M200c
values plus final concentration or its five-epoch history. Freeze that feature
choice per fold. Missing halo features are imputed on the map-training rows.
True stellar mass may scale training or evaluation losses, never predictions.

## Frozen coordinates, references and loss

Correction = a predicted log-amplitude and three Legendre tilts of positive
baseline shell masses, normalized to the predicted baseline total. Exact
zero recovers the baseline. Reuse Exp75's decoder without changing its radial
basis, positivity rule or four-parameter count. The reference is the saved
Exp75 measured-history hybrid, reconstructed by fitting its original
calibration coefficient targets and its saved feature/degree/penalty choice;
require agreement within 1e-8 dex before evaluating a new loss.

For weight w in {0, 0.25, 1}, fit four coefficients to each calibration
galaxy-epoch by minimizing mean(log10 predicted/data CoG)^2 plus w times
mean transformed-annular-error squared. Annular edges are exactly
0, 2, 10, 30, 50, 100, 148.22 kpc (actual last grid value). Interpolate in
linear cumulative mass and radius, as in standard QA. The transform is
asinh(M_ann/(0.001 * true_total))/ln(10); negative/zero observed annuli are
represented rather than discarded. This is a fitting distance, not a
likelihood. Equal galaxy/epoch weights apply. Four-coefficient least squares
starts at zero, max 100 evaluations, tolerances 1e-9; one 300-evaluation retry
is allowed and recorded. Final curves must be finite, positive and monotone.

Weight zero uses the original Exp75 coefficient fitter and saved map choice
as an exact nested reference. For positive weights, retain linear/quadratic
ridge maps and penalties 0.01, 0.1, 1, 10, 100. Within the calibration fold,
use the same two-way inner split (row parity). Compare candidates on a common
criterion: RMS error of the last two annular masses divided by the true total,
requiring inner-held-out mean radial CoG RMS no more than 2% above the
weight-zero reference. Among candidates within 1% of the best eligible
outer error, prefer lower weight, lower degree, then larger penalty.
The weight-zero saved-map configuration is always an eligible fallback.
Selection sees no outer evaluation labels. Refit the chosen map using all
calibration rows and predict only from held-out halo inputs.

Representation diagnostic: additionally fit weights 0, 0.25 and 1 directly
to each true profile using its corresponding held-out baseline. These fits
use stellar labels and are NEVER halo predictions or training examples.
First test the pilot: if neither positive weight improves outer RMS by 10%
without worsening mean CoG RMS by more than 2%, record that limitation;
still distinguish a small absolute CoG cost from the relative safeguard.
The full diagnostic may close the experiment without a halo-map advancement
if the family cannot satisfy the declared outer/CoG compromise. No change
of thresholds after viewing a result.

## Checks, decision and figures

Before full execution: focused synthetic checks for exact zero nesting,
weight-zero parity, positive monotone predictions, annulus conservation,
known-coefficient recovery, training-only preprocessing and evaluation-label
poisoning. Complete a 30-galaxy path (read, fit, select, predict, score, save
JSON/NPZ and direct CoG PNG/PDF) in less than 60 measured seconds, then a
150-galaxy pilot before the fixed 842-galaxy run. One process, one thread,
private environment and caches. Preserve checkpoints without replacement;
record input/code hashes, SHA, IDs, roles, convergence and wall times.

Advance only if the selected halo prediction improves pooled outer-annular
RMS by at least 10% versus Exp75 hybrid, with a positive paired 1000-galaxy
bootstrap 95% interval; pooled mean radial CoG RMS may worsen by at most 2%.
Maximum absolute final-halo-tercile median radial bias and absolute median
R50/R80/R90 log bias at every epoch may worsen by at most 0.01 dex.
Report log annular residuals on positive masses and zero/nonpositive counts.
For residual future-growth response, compare with the measured response
after quadratic current-mass control; a paired 95% lower bound above zero
for an increase in absolute residual slope at any epoch blocks advancement.
Keep all epochs together in bootstrap draws; intervals condition on fitted
models and are pointwise, not multiple-test adjusted.

Generate and inspect matched halo-mass-binned average CoGs, exact stellar-mass
planes, density/annulus/size summaries and fixed/most-improved/most-worsened
individuals. Any scientifically interesting halo candidate receives the full
standard QA battery, with Exp75's already generated baseline/hybrid/direct
figures as the frozen references. Every new figure is shown with its full
PNG path, PDF companion and a plain-language caption. A null still receives
direct profile and mass-plane figures, not just summary numbers.

Discovery reuse is exploratory, not independent validation. Record whether
the limit lies in the radial correction, its fitting loss or its halo map.
No library promotion. Science stops September 8 at 06:00 Shanghai; preserve
owned checkpoints and stop owned fitting processes then. Finish commit,
private no-ff integration, normal push, verified non-overwriting artifact
archive and Obsidian journal/snapshot before 06:55. Retain owned worktrees.

## Post-result attribution check, declared before its driver

All five calibration folds selected annular weight 0.25 and linear maps;
one also changed the ridge penalty. To avoid assigning a map-selection gain
entirely to the annular targets, evaluate two diagnostic controls on the same
fixed folds: (a) weight 0.25 with each fold's original Exp75 feature, degree
and penalty held fixed; (b) weight-zero coefficient targets with degree and
penalty selected on the same inner outer-error criterion and 2% CoG safeguard.
For (b), prefer lower degree then larger penalty within 1% of the best eligible
outer score. These are post-result attribution checks, not new candidates for
advancement, and cannot override any failed predeclared criterion. No new
weight, feature or radial basis is introduced. Save predictions, paired
outer-error intervals and a direct profile/annular comparison figure.
