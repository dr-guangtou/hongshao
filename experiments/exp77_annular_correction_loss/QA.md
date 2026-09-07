# Exp77 figure index and captions

Every entry below is under `figures/`, with PNG and PDF companions. Science
figures compare the same 842 held-out measured CoGs at five epochs. The
reference is Exp75's measured-history deposition model plus four halo-only
corrections fitted to cumulative mass; the candidate changes the correction
loss to include annular masses. All amplitudes and shapes are predicted from
halos. The direct-to-true-profile diagnostic is explicitly separate.

## Matched and diagnostic figures

- `gate_individual_cogs`: 30-galaxy operational check, not a scientific model
  qualification; three held-out profiles establish that the complete path runs.
- `mass_binned_cogs`: mean measured and predicted CoGs and median log errors
  in final-M200c thirds at z=0.4 and 2; mean cumulative profiles remain close
  to the reference despite substantial changes in their outer differences.
- `exact_stellar_mass_planes`: log mass inside 30 kpc versus in 50--100 kpc,
  exact interpolated apertures; the z=2 relation remains too steep.
- `radial_and_annular_errors`: cumulative and annular RMS versus radius at
  z=0.4 and 2; the outermost annulus improves most, but some inner annuli worsen.
- `fixed_individual_cogs`: IDs 479, 1869 and 3386 from the operational plot,
  now predicted with full calibration folds; large individual errors remain.
- `most_changed_individuals`: IDs 1431 and 24 have the largest improvement
  and deterioration in outer-annular squared error; follow each across epochs
  to see that cumulative appearances can obscure outer-mass changes.
- `representation_tradeoff`: weights 0, 0.25 and 1 fitted to each true CoG,
  never halo predictions; the fixed four-coordinate family can represent much
  better envelopes at a small absolute cost to cumulative accuracy.
- `residuals_sizes_and_growth`: actual prediction-error CDF, R50/R90 median
  offsets and paired conditional-growth response changes; the candidate
  fails the z=1.5 R50 and z=2 response safeguards despite outer gains.
- `ranked_individuals`: best ID 921, typical ID 366 and worst ID 3079 by
  galaxy-averaged radial RMS, each shown at its worst epoch; even a typical
  galaxy can have a substantial inner-profile error at one epoch.
- `attribution_controls`: fixed-regression annular targets versus outer-
  selected regression with cumulative-only targets; the former reproduces
  Exp77, the latter has no established outer gain. This is post-result
  attribution, not a new selection or a way around the failed safeguards.

## Full standard QA

The following 12 figures use prefix `standard_qa/qa_` and suffix
`_exp77_annular_selected.png` (PDF alongside). They retain the established
QA conventions and masks; the corresponding Exp75 hybrid figures are frozen
references in that experiment's `figures/continuation/hybrid` directory.

| Middle part of filename | Caption and interpretation |
| --- | --- |
| bins | Median CoGs and raw/shape-normalized residuals in final-halo-mass thirds; typical radial offsets are small but structured. Shape normalization is diagnostic only, never an inference input. |
| bins_ms | The same diagnostics binned by measured stellar mass; regression to the mean contributes to opposite offsets at the mass extremes. |
| cases | Best and worst individual profiles across all epochs; `idx` is the local sample row, not global galaxy ID, and large error tails survive. |
| cdf | Measured and predicted aperture/annular mass distributions, plus their CDF differences; not a prediction-error CDF. |
| dens | CoG-derived surface-density profiles and log residuals by halo mass; outer structure improves but inner and low-halo-mass high-z offsets remain. |
| growth | Stellar mass and half-mass radius of the same galaxy across epochs; point predictions are more coherent than the measured population, not a validated stochastic model. |
| mass_Re_aper | Stellar mass in size-scaled apertures versus measured counterparts; broad individual errors remain despite modest median offsets. |
| mass_Re_diff | Size-scaled annular and envelope masses versus data; cumulative accuracy does not guarantee each interval. |
| mass_kpc_aper | Exact physical-aperture stellar masses versus data and fractional residuals; the overall amplitude remains close to the reference. |
| mass_kpc_diff | Physical annular/envelope masses versus data; the 100--150 kpc panel is explicitly unavailable beyond the 148.22 kpc grid, not silently extrapolated. |
| planes | Observational stellar-mass planes against data; z=2 inner--outer slope remains too steep even though outer-envelope normalization improves. |
| size | Enclosed-fraction size planes and mass-binned biases; R20 exposes an inner-profile issue not captured by the R50/R80/R90 aggregate safeguards. |

These are point predictions, not sampled populations; width differences alone
cannot reject a conditional mean. Reference and predicted size masks follow
the existing QA helper. The exact-annulus report separately records positive
comparison counts and measured/predicted below-grid enclosed-radius counts.
No stellar profile or sample membership was silently repaired.
