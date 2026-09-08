# Exp77 — does explicitly fitting annular masses improve the halo correction?

## Final disposition (2026-09-09)

Closed by user decision, with no production correction-map integration and
no further fitting required. Annular calibration produced a promising
improvement in outer masses, despite the recorded strict relative safeguards.
The architectural decision is separate: 140/220 fitted halo-map coefficients
and 20 epoch-specific correction outputs encode TNG300 residual relationships
without a compact physical prescription that observations can constrain.
Retain the annular-objective lesson and all diagnostic results; move to a
compact physical model shared across epochs. Historical gates are unchanged.
The earlier recommendations below are retained as chronology, not active tasks.

The September 9 PDF describes the mechanism accurately, but its discussion of
possible frozen-component integration predates this final decision and is
not a production recommendation. This final disposition takes precedence.

## Model explanation (September 9)

The six-page [model and profile-correction note](../../output/pdf/exp77_model_and_profile_correction.pdf)
explains the implemented mathematics, calibration and prediction pseudocode,
parameter accounting, and prospective forward-model use. Rebuild it with
`uv run --no-project --with reportlab --with pymupdf python experiments/exp77_annular_correction_loss/build_model_note.py`.
The generated PDF and source-hash record live under `output/pdf/`; rendered
review pages live under `tmp/pdfs/exp77_model_note/`. This is documentation
only: it changes neither the predeclared verdict nor any fitted result.


Status: discovery and attribution controls completed, full QA inspected,
2026-09-08. The declared candidate fails the size and conditional-growth
safeguards; no production adoption or replacement of that verdict.

## Result in plain language

Explicitly asking the correction fit to reproduce annular masses fixes much
of the outer-envelope error without adding a halo feature or a profile
coordinate. It does not yet satisfy all the predeclared safeguards. The
scientific lesson is to improve what the fitting loss asks for before
expanding the model family; this is not a recommendation to adopt this fit.

All results compare the same 842 held-out discovery galaxies at five epochs
with Exp75's measured-history hybrid: the physical deposition model plus
four corrections predicted from measured halo masses and concentration.
The frozen physical baseline was not refitted. The reference was reproduced
exactly, with zero difference in every saved prediction across all five folds.

| Quantity against measured CoGs | Exp75 cumulative-loss correction | Exp77 annular-loss correction | Interpretation |
| --- | ---: | ---: | --- |
| RMS outer-annular mass residual / true total | 0.03298 | 0.02843 | 13.81% better; paired 95% interval 11.59--16.22% |
| Mean galaxy-epoch radial log-CoG RMS (dex) | 0.10771 | 0.10786 | 0.135% worse, within the allowed 2% |
| z=2 median log 100--148.22 kpc mass prediction/data (dex) | +0.27343 | +0.02231 | Median envelope excess falls from 87.7% to 5.27% |
| z=1 median log R90 prediction/data (dex) | +0.06322 | -0.00573 | Typical outer size becomes much closer to the data |
| z=1.5 median log R50 prediction/data (dex) | +0.00252 | +0.02173 | Typical half-mass radius worsens from 0.58% to 5.13% too large |

The outer RMS equally weights the 50--100 and 100--148.22 kpc annuli,
galaxies and epochs; each mass residual is divided by that galaxy-epoch's
measured total. True total is used only in loss/evaluation, never prediction.
The z=2 positive-envelope comparison uses 829 of 842 galaxies; nonpositive
measured annuli are retained in the finite transformed fitting loss and the
linear-mass RMS. The named endpoint is the actual radial-grid boundary.

All five inner-calibration selections chose annular weight 0.25 and a linear
halo map. The fitted profile family, halo feature choice and regression
settings ultimately match the corresponding Exp75 fold. Attribution controls
confirm that fixing every original regression setting gives bitwise-identical
predictions to Exp77, while changing map selection alone gives an outer RMS
of 0.03303 versus the reference 0.03298 (0.14% worse, paired 95% interval
from 1.52% worse to 1.52% better). Thus the outer gain comes from the changed
fitted profile targets, not the regression settings. The first reading
mistook a penalty difference between folds for a change from that fold's
reference; the saved choices and explicit controls correct that mistake.

## Why advancement still fails

- The z=1.5 R50 absolute median bias increases by 0.01921 dex, exceeding the
  predeclared allowance of 0.01 dex. All R80 and R90 epoch checks pass;
  R90 bias improves at every epoch. This is a specific half-mass-radius
  trade-off, not a claim that all sizes became worse.
- At z=2, after controlling quadratically for current M200c, the model-minus-
  data response of log stellar mass inside 100 kpc to later log halo growth
  changes from -0.04374 to -0.05002 dex/dex. The increase in absolute error
  is 0.00628 dex/dex, with paired 95% interval [0.00366, 0.00852]. This is
  small in absolute terms but fails the declared no-significant-increase
  rule. These pointwise intervals condition on trained models and do not
  resample fitting uncertainty or correct for multiple epochs. Full halo
  history remains legitimate; this is not stellar-information leakage.
- The z=2 inner--outer mass plane is still too steep: slope 2.045 for
  log Mstar(50--100 kpc) against log Mstar(<30 kpc), versus 2.118 in Exp75
  and 1.565 in the data. Better outer-envelope normalization does not mean
  every conditional relation is right.
- Fixed and ranked individual profiles retain large failures. Narrower
  predicted mass planes are expected from point predictions and are not
  alone a failure of the conditional mean; no scatter distribution or
  coherent multi-epoch sampling is fitted or qualified here.

The halo-bin radial-bias safeguard passes: its maximum absolute median
bias changes from 0.05233 to 0.05605 dex, less than the 0.01 dex allowance.
Four of six advancement checks pass; the two failures above retain their
original thresholds. Do not select a favorable subset of safeguards afterward.

## What the representation diagnostic teaches

Directly fitting four coefficients to every true profile is a diagnostic,
never a halo prediction. On all 842 galaxies, adding annular weight 0.25
reduces its outer RMS from 0.01971 to 0.00949 of the true total, while mean
radial log-CoG RMS increases from 0.00693 to 0.01128 dex. Weight 1 gives
0.00918 outer RMS and 0.01285 dex CoG RMS. Thus the same four coordinates
can represent much better envelopes, but not at zero cost to inner structure.
Both positive weights fail the diagnostic's strict 2% relative CoG safeguard;
the absolute cost is small and is explicitly reported, not used to relax it.
The halo-prediction trade-off differs because predicting coefficients from
halos is not the same task as fitting each measured galaxy separately.

The 30-galaxy operational path passed in 5.79 seconds (limit 60 seconds).
All five 150-galaxy pilot folds passed; full-discovery fold wall times were
8.10--8.44 seconds, with no coefficient-fit retries. The full direct-fit
diagnostic took 31.28 seconds and had no retries. Runs were serial and
single-threaded in a private environment. Four new focused tests and the
18 inherited Exp75 tests pass; repository-wide pytest was not collected
because the unrelated missing Exp07 input remains outside scope.

## Decision and next step

Close Exp77 as an informative but not qualifying loss test. Keep Exp75 and
Exp77 as diagnostic references; do not change production defaults. Next,
predeclare a joint CoG/annular/size fitting and calibration-selection test
on the fixed family, with conditional-response safeguards applied before
outer evaluation. The point is to make the fit respect several required
observables together, not repeatedly replace one single score with another.
Review that plan alongside Claude's completed Exp76 findings without
modifying its experiment. Independent validation still requires a separately
declared role for protected data; none was opened overnight.

## Artifacts and reproduction

- `run.py prepare`: copies and hash-verifies 14 named Exp75 input/checkpoint
  files into this private worktree. The source is closed Exp75, not Claude's
  active worktree. Reproduction requires those preserved gitignored inputs.
- `run.py gate`, `run.py pilot --fold N`, `run.py discovery --fold N`:
  frozen-baseline correction fits and immutable JSON/NPZ checkpoints.
- `run.py pilot_representation` / `discovery_representation`: labeled
  direct-to-stellar-profile diagnostics, separate from all halo predictions.
- `report.py summary` / `figures` / `standard`, then `diagnostics.py`:
  paired scores, full standard QA and matched figures; no model selection.
- `attribution.py`: predeclared post-result controls, not new advancement.
- `outputs/summary.json` and `outputs/standard_qa.json`: numerical source
  records; `outputs/manifest.json`: final recursive artifact inventory.
- `QA.md`: captions and index for all 22 PNG figures, with PDF companions.
  Source code uses the repository plotting style. Every figure was displayed
  and visually reviewed before closeout.

Use `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1 uv run --no-sync python` before each script path.
The overnight cutoff deliberately stops future fitting invocations after
September 8 06:00 Shanghai; a later reproduction needs a newly declared run
window and a fresh output location. No existing checkpoint is overwritten.

## Original predeclaration

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
the initial reading suggested one changed the ridge penalty (the completed
control above corrects this: it was unchanged from that fold's reference).
To avoid assigning a map-selection gain
entirely to the annular targets, evaluate two diagnostic controls on the same
fixed folds: (a) weight 0.25 with each fold's original Exp75 feature, degree
and penalty held fixed; (b) weight-zero coefficient targets with degree and
penalty selected on the same inner outer-error criterion and 2% CoG safeguard.
For (b), prefer lower degree then larger penalty within 1% of the best eligible
outer score. These are post-result attribution checks, not new candidates for
advancement, and cannot override any failed predeclared criterion. No new
weight, feature or radial basis is introduced. Save predictions, paired
outer-error intervals and a direct profile/annular comparison figure.
