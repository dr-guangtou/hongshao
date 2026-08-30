---
date: 2026-08-30
repo: hongshao
branch: exp64-no-central-hole
tags:
  - journal
  - exp64
  - cog-profiles
---

## Progress

- Predeclared and completed two searches for a finite, positive central projected density while retaining the Exp62 double-Sersic, cubic-logit, and PCA references.
- Tested the one-break and two-break cored density laws on the 90-profile development sample; the complete path took 30.35 seconds.
- Tested the five-coordinate monotone-density envelope on all 16,900 galaxy--epoch profiles under three fitting objectives; the full calculation took 712.96 seconds.
- Verified all 37 focused checks and generated direct accuracy, conditioning, density-profile, and residual figures.
- Closed Exp64 as a measured null because neither family passed the predeclared combined accuracy, conditioning, boundary, and robustness gates.

## Lessons Learned

- Positive annular masses should be integrated separately before cumulative summation so numerical roundoff cannot create a decreasing model CoG.
- Repairing cubic-logit's unresolved central density leaves most measured 2--148 kpc densities unchanged and therefore does not improve its resolved accuracy trade space.
- Experiment suites that import same-named local modules must run in separate Python processes to avoid false module-collision failures during test collection.

## Key Issues

- The monotone-density envelope has a 4.94% fitted-boundary incidence, above the predeclared 1% maximum, and its worst near-optimal decoded CoG spread is 0.00171 dex, above the 0.001 dex maximum.
- The envelope exceeds the allowed double-Sersic error in multiple epochs for CoG, shell-density, or size metrics; it is not promoted.
- The broader promising-model QA battery was not triggered because no candidate cleared the numerical decision gate.
- Repository-wide test collection remains blocked by the unrelated missing gitignored Exp07 input; the focused Exp64 checks pass.
