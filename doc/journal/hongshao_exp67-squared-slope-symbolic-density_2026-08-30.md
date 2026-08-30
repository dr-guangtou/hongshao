---
date: 2026-08-30
repo: hongshao
branch: exp67-squared-slope-symbolic-density
tags:
  - journal
  - exp65-exp67
  - symbolic-density
---

## Progress

- Completed Exp65's restricted projected-density grammar; all 13 expressions passed the mathematical checks, but none reproduced the 15-profile unrestricted-double-Sersic teacher bank within both frozen accuracy limits.
- Completed Exp66's relaxed 88-expression grammar on 250 discovery galaxies at all five epochs; no bounded-linear slope family passed the combined convergence, boundary, conditioning, and accuracy gates.
- Completed Exp67's positive squared-slope search on 1,200 discovery galaxies at all five epochs, or 6,000 profiles and 1,056,000 profile-start optimizations; no family passed every frozen discovery gate.
- Found the elementary free damped-cosine family, whose epoch-median full-CoG and shell-density RMS are lower than unrestricted double Sersic at every epoch on the same 6,000 discovery profiles.
- Added a discovery-only comparative QA driver and five PNG+PDF figure pairs with the double-Sersic and new-family formulas printed in the figures.
- Verified the complete Exp65--Exp67 focused suite: 32 checks passed in 2.44 seconds, and Ruff passed for all three experiment directories.
- Committed the comparative QA as `7e85e4a` after the experiment commits `ce6b2cb`, `ea9c058`, and `f743efe`.

## Lessons Learned

- An algebraically positive squared logarithmic density slope removes the central-hole defect and permits stronger contrast than the bounded-linear wrapper.
- The free damped cosine is a genuinely better profile reconstructor than double Sersic on the discovery population, not merely a better aggregate score; the resolved residual and mass-binned CoG figures show the gain across radius and population bins.
- High per-profile reconstruction accuracy does not guarantee useful latent coordinates: damping, frequency, contrast, baseline slope, and core radius can compensate over the finite measured radial interval.
- Cumulative-mass integration suppresses density-level differences, so multi-start profile agreement and weakest-Jacobian scans are necessary even when the best-fit CoGs look excellent.
- The fixed sine packet remains the strongest accuracy--conditioning compromise, but rare branch ambiguity prevents production promotion under the frozen worst-case rule.

## Key Issues

- The free damped cosine has only 99.50% optimizer convergence, 17.83% boundary incidence, and a 0.00443 dex maximum CoG difference between near-optimal solutions; its weakest scaled-Jacobian direction is substantially weaker than double Sersic and cubic-logit.
- The current figures demonstrate accuracy clearly, but they do not directly overlay alternative near-optimal solutions or scan the weakest parameter direction.
- The next experiment should test targeted damped-cosine reparameterizations, fixed or discretized damping, finite-window orthogonalization, and canonical branch rules before any new broad grammar search.
- Judge a reparameterization by decoded-profile stability, coordinate continuity across galaxies and epochs, synthetic recovery, multi-start agreement, and retained CoG and shell-density accuracy.
- Exp67 selection and validation galaxies remain untouched; a new experiment must predeclare whether and when they may be used.
- Repository-wide pytest collection remains blocked by the unrelated missing gitignored Exp07 input; use the focused checks.
- Do not modify `/Users/shuang/Dropbox/work/project/massive/hongshao`, where `exp63-analytic-growth` remains active.
