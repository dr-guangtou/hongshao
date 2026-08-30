---
date: 2026-08-30
repo: hongshao
branch: exp62-cog-fit-atlas
tags:
  - journal
  - exp62
  - cog-profiles
---

## Progress

- Completed the all-galaxy, five-epoch CoG fitting atlas and recorded the family, objective, robustness, parameter-scaling, and halo-information comparisons in `experiments/exp62_cog_fit_atlas/README.md`.
- Found the five-parameter cubic-logit CoG: 0.00249 dex median full-CoG RMS across epochs under the log-CoG objective, compared with 0.00263 dex for unrestricted double Sérsic.
- Promoted the family to `hongshao/cubic_logit.py` with exact enclosed radii, projected density, two-dimensional rendering, numerical Hankel transform, and Gaussian/Moffat PSF transfer functions.
- Verified the promoted family with 26 focused checks; the 90-profile mathematical path took 15.85 seconds and the 16,900-profile audit took 21.50 seconds.
- Measured a modest held-out DiffMAH association with individual scale-free CoG diversity, mainly inside R50; more than 90% of that diversity remains unexplained by current halo mass, main-branch DiffMAH, and concentration.
- Committed the completed experiment and reusable profile as `abd9ba9`.

## Lessons Learned

- More than 90% unexplained by the supplied DiffMAH parameterization does not mean that final CoG diversity is mostly unrelated to assembly; relevant information may not be encoded by the smooth main-branch MAH.
- A cumulative profile can fit the measured CoG accurately while its differentiated density becomes unphysical outside the fitted radial support.
- The cubic-logit central density maximum lies below 2 kpc for 16,896 of 16,900 fits, but the exact-center density is zero for every allowed parameter set.
- Measured aperture mass must remain separate from analytic infinite total: the median correction beyond 148 kpc is 0.00563 dex, while its 99th percentile is 0.401 dex.
- Concurrent halo mass reproduces the average normalized CoG; DiffMAH supplies only modest information about galaxy-to-galaxy deviations from that average.

## Key Issues

- The cubic-logit family is approved as a finite-resolution projected-CoG interpolator, not as a physical central density law or spherical three-dimensional profile.
- The next experiment should search for a comparably accurate and well-conditioned CoG or projected-density family whose central density remains finite or cusped rather than approaching zero.
- The new search must retain the Exp62 all-epoch atlas, unrestricted double-Sérsic fit, cubic-logit fit, density recovery, characteristic radii, objective sensitivity, and parameter-degeneracy tests as common references.
- Repository-wide pytest collection remains blocked by the pre-existing Exp07 dependency on absent gitignored `data/processed/tng300_072_z0p4.fits`; the focused Exp62 suite passes.
