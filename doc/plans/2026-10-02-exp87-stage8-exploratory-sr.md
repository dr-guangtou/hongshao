# exp87 Stage 8 — exploratory symbolic regression across epochs

Requested by the user on 2026-10-02, after Stage 7. An addendum to
`doc/plans/2026-10-02-exp87-inverse-halo.md`; same branch
(`exp87-inverse-halo`), same development folds, the lockbox untouched.

## What is asked

Use symbolic regression to EXPLORE the functions that turn the z = 0.4
stellar mass distribution into the halo's mass at each of the five epochs
(z = 0.4, 0.7, 1.0, 1.5, 2.0), and present the three best formulas per epoch.
The Stage 4 robustness rule (a formula is accepted only if it recurs across
folds) is dropped on purpose. Searches may run longer and use a richer
operator set. Two ways of writing the stellar mass distribution:

1. **The profile**: the stellar mass in a central aperture and in a series of
   annuli (not the cumulative curve of growth).
2. **A parametrised curve of growth**: fit a few-parameter model to each
   galaxy's curve of growth and use the parameters as the inputs.

## Design

**Inputs** (all from the z = 0.4 curve of growth; stellar masses as
log10(M / 10^10 Msun), radii as log10(R / kpc)):

| set | approach | variables |
| --- | --- | --- |
| `annuli6` | 1 | mass inside 10 kpc; annuli 10–30, 30–50, 50–100, 100–132, 132–148 kpc |
| `annuli9` | 1 | mass inside 4.9 kpc; eight annuli on the grid's own radii out to 148 kpc |
| `sersic` | 2 | one Sérsic fit: total mass, effective radius, index |
| `hill` | 2 | a logistic in log R: asymptotic mass, half-mass radius, steepness |
| `double` | 2 | inner Sérsic + outer exponential: total mass, outer fraction, inner radius, inner index, outer radius |
| `logpoly` | 2 | a cubic in log(R / 20 kpc): four coefficients |
| `sizes` | 2 | M*(<148 kpc) and the radii enclosing 20, 50, 80 per cent (non-parametric reference) |

The fits are in `cogparams.py`; their quality (rms residual of the fit) is
reported with the results.

**Targets**: h = log10(M200c / 10^13 Msun). z = 0.4 on the complete parent
(3380; truncated at 10^13, so the search runs on Stage 4's pseudo-latent
response and the score is the truncated CRPS); z ≥ 0.7 is the main
progenitor's mass on the curated sample (no cut, plain normal), as in Stage 7.

**Search**: PySR, one search per (input set, epoch) = 35 searches, 900 s each
on 4 threads (Stage 4's longest was 420 s on one), maximum size 45 (Stage 4:
40), operators `+ − × ÷ pow max min` and `square cube sqrt log exp tanh`,
`log(exp + exp)` allowed so that masses can be added in linear units.

**An honest score without the recurrence rule**:

- folds 0–2: the search; fold 3: ranks the Pareto front; fold 4: scores the
  formulas as found. The fold-4 number is the strict one.
- the "refit" score: the formula's shape frozen, its constants refitted on
  four folds and scored on the fifth, over all five folds. Comparable with
  Stage 7's tables; mildly optimistic because the shape was chosen on folds
  0–3. Both numbers are reported, next to the linear and linear + quadratic
  models on the same inputs and the 24-shell linear model.
- "three best" = the three lowest errors on the ranking fold among formulas
  that are different functions (different shape, predictions differing by
  more than a quarter of the residual scatter), per epoch and per approach;
  plus the simplest formula within 2 per cent of the best.

**Labelled exploratory.** Nothing here is an accepted formula in the sense of
Stage 4; no claim of recurrence is made.

## Steps

1. `cogparams.py`: fit the families, report the fit quality.
2. `stage8_explore.py bench`: the reference models per epoch.
3. Smoke (30 s searches on two configurations), then a 60 s / 300 s
   convergence check on one configuration to see what the longer search buys.
4. `stage8_explore.py search --part 0/2` and `--part 1/2`, detached, two jobs
   of four threads (about 5 h).
5. `stage8_explore.py report`; a figure; README section; lessons, todo,
   handover addendum; commit on the branch. No merge.

## Changes made on the way (2026-10-02, recorded before the write-up)

- **PySR's sympy export is bypassed.** After a search ends, PySR converts
  every formula to sympy; on deep `pow` / `max` / `min` forms that step
  overflowed the recursion limit (two jobs died) or ran for 50 minutes
  without returning (one job). The search itself was finished and saved each
  time. `stage8_explore.py` now skips the export and reads PySR's own
  `hall_of_fame.csv`; the three interrupted searches were recovered from
  those files, not re-run. The direct parser of the formula strings had been
  checked against PySR's own predictions on the first 24 searches.
- **A pole filter was added to the "three best" rule** after the first full
  report ranked, at z = 1.5, two formulas that divide by a quantity crossing
  zero (fold-4 RMSE 0.234 and 0.345 against 0.196 linear). A formula is
  ranked only if both its refit error and its fold-4 error stay below 1.1
  times those of a straight line in total stellar mass. Fold 4 is thereby
  used once, as a blow-up filter.
- **Two plain readings were added to the report** so the formulas have
  something to be compared with: each of the nine annuli alone and every
  pair of them (`stage8_explore.py annulus`), and the fraction of the
  near-best formulas that use each variable.
- **A convergence check**: the nine-annulus search at z = 0.4 and z = 1.0
  repeated at 3600 s (`search --deep`), reported next to the 900 s runs and
  not mixed into the "three best".
