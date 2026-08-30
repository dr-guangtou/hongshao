---
date: 2026-08-31
repo: hongshao
branch: exp70-global-damping
tags:
  - journal
  - exp69-exp70
  - damped-cosine
---

## Progress

- Completed Exp69's targeted free damped-cosine reparameterization test; coordinate relabeling and a canonical branch rule did not remove the damping degeneracy.
- Predeclared Exp70's four population-wide damping values, blind sample roles, objectives, diagnostics, gates, and complete operational path before implementation.
- Passed the Exp70 all-in 25-galaxy operational gate in 20.88 seconds; all twenty synthetic cases and seven focused checks passed.
- Fit the free reference and four fixed-damping families with eight starts to 6,000 discovery profiles from 1,200 galaxies in 1,284.26 seconds.
- Recorded a negative production decision after none of the four fixed values passed all discovery gates.
- Generated and inspected five PNG+PDF discovery figure pairs covering synthetic recovery, the accuracy--conditioning trade, multi-start solutions, weakest Jacobian directions, and coordinate continuity.

## Lessons Learned

- Fixing damping globally made the weakest scaled Jacobian 17.9--20.0 times stronger than newly refit free damping in the weakest epoch, but the best fixed value was still 20.9% worse in worst-epoch median full-CoG RMS.
- Removing one weak coordinate did not ensure unique remaining coordinates; maximum near-optimal CoG spread remained 0.00682--0.01275 dex against the 0.003 dex limit.
- Shared damping improved median descriptor continuity, but no nearest-profile or adjacent-epoch 95th-percentile continuity ratio reached the required 0.75.
- Continuity diagnostics must use descriptor bounds derived from the frozen native parameter box rather than approximate hand-written scales.
- Local conditioning, multi-start uniqueness, coordinate continuity, and population accuracy are separate production requirements.

## Key Issues

- Do not refine the fixed-damping grid after this result or continue this family to production through another coordinate relabeling.
- Exp70 selection and validation were not evaluated because discovery froze no candidate.
- A future production profile experiment would need a genuinely different family rather than a globally fixed damping constant.
- Repository-wide pytest collection remains blocked by the unrelated missing Exp07 input; continue using focused checks.
