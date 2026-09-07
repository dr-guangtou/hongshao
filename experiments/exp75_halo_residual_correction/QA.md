# Exp75 measured-history QA: figure captions

All paths below are relative to `figures/continuation`; each figure has PNG
and PDF versions. All science panels use the same 842 held-out discovery
galaxies at z=0.4, 0.7, 1, 1.5 and 2 and their measured X--Y isophotal CoGs.
The physical baseline integrates deposition along measured halo histories;
the hybrid adds four halo-predicted corrections; the direct model predicts
five profile coordinates without the deposition equation. Evaluated stellar
masses are references only, never prediction inputs. These are point
predictions, not draws from a calibrated scatter distribution.

## Matched comparisons

- `measured_discovery_mass_binned_cogs`: mean measured and predicted CoGs
  with residuals in final measured-halo-mass bins; the hybrid reduces average
  offsets without eliminating the large errors of individual galaxies.
- `measured_discovery_stellar_mass_planes`: historical-grid aperture planes,
  with actual grid radii labeled; use `exact_stellar_mass_planes` for the
  standard 30 and 50--100 kpc comparison. The high-z hybrid remains too steep.
- `measured_discovery_ranked_individuals`: best, typical and worst hybrid
  galaxies by mean radial log error, all epochs kept together; large tails
  remain despite the pooled improvement.
- `exact_stellar_mass_planes`: exact interpolated 30 and 50--100 kpc stellar
  masses at z=0.4 and 2 for all three models against the data. The hybrid
  improves but does not fix the z=2 tilt; the direct model's plane is closer.
- `fixed_individual_cogs`: the original operational examples, IDs 3362,
  1802 and 9, now predicted by full discovery fits; correction is not an
  object-by-object guarantee of improvement.
- `most_changed_individuals`: ID 1893 gains most and ID 9 loses most in
  galaxy-averaged radial RMS after adding the correction; the same IDs are
  followed at z=0.4, 1 and 2 to expose coherent failures.
- `representation_vs_halo_prediction`: fitting four corrections directly to
  true profiles removes most cumulative error but not the outermost annular
  error. Purple is a representation diagnostic, never a halo prediction.
- `future_growth_response`: partial response of log stellar mass inside
  100 kpc to later log M200c growth, controlling quadratic epoch halo mass;
  shaded paired galaxy-bootstrap intervals condition on trained models.
  Agreement with the measured response is the target. The hybrid slightly
  under-responds at z=0.7 and 2; pointwise intervals are not multiple-test adjusted.
- `pre_epoch_input_audit`: supplied halo fits for IDs 2582 and 1256 can have
  negative growth between snapshots. This stops that input arm, not the
  measured-history experiment; no clipping or sample removal is applied.
- `gate_measured_fold0_individual_cogs`: capped 30-galaxy operational test,
  not a scientific score or a fitted-model qualification.

## Full standard battery

For each `model` in `baseline`, `hybrid`, `direct`, the path pattern is
`model/qa_KIND_exp75_measured_model.png`. These 12 captions apply to all three
versions; model-specific interpretations follow the table. The corresponding
`outputs/continuation/standard_qa_MODEL.json` preserves numerical summaries.

| KIND | What is shown and how to read it |
| --- | --- |
| bins | Mean CoGs and residuals in measured final-halo-mass bins, against measured profiles; systematic radial errors survive averaging. |
| bins_ms | CoGs and residuals binned by reference stellar mass; exposes the conditional errors hidden by averaging unlike galaxies together. |
| cases | Best, typical and worst individual CoGs and residuals against measurements; plotted `idx` is the local snapshot row, mapped to global ID in `inputs.npz`. |
| cdf | Cumulative distributions of radial prediction/data residuals; compare offsets and error tails, not only the central score. |
| dens | CoG-derived one-dimensional density profiles compared with the same operation on data; outer derivatives expose errors hidden in cumulative mass. |
| growth | Same-galaxy stellar masses across epochs compared with the measured evolution; overly coherent point predictions are not a calibrated temporal scatter model. |
| mass_Re_aper | Enclosed stellar masses at size-scaled apertures versus measured counterparts; the reference and predicted sizes follow the established QA convention. |
| mass_Re_diff | Size-scaled annular and outskirt masses versus data; subtraction can amplify small cumulative errors. |
| mass_kpc_aper | Exact interpolated physical-aperture masses versus data; inspect both identity relation and residuals. |
| mass_kpc_diff | Physical annular and outskirt masses versus data; final grid coverage is 148.22 kpc, not an observed 150 kpc point. |
| planes | Observational stellar-mass and size planes overlaid on measurements; high-z slope and location are distinct from population width. |
| size | Enclosed-fraction radii versus data, with bias and distribution comparisons; out-of-grid or invalid sizes follow the standard QA exclusions. |

Baseline interpretation: high-z profiles are typically too extended and the
inner--outer mass plane is too steep; large envelope errors survive a modest
cumulative score. Hybrid interpretation: typical size offsets and average
CoGs improve substantially, but the outer envelope and worst objects remain
problematic. Direct interpretation: the outer-mass plane is closer to the
data, but z=2 cumulative accuracy is worse; outermost density slopes and
profile tails also need scrutiny. Narrow point-prediction planes alone are
not evidence against their conditional means. None passes production QA.

The standard half-light-radius helper rejects any strictly decreasing input
step, including tiny floating-point decreases in a few measured CoGs; no
stellar input was silently repaired. Exact-annulus summaries additionally
report positive-mass comparison counts, and enclosed-radius summaries report
extrapolation counts. Use those masks when comparing individual diagnostics.
