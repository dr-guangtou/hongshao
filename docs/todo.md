# Documentation task: Exp77 model explanation

## September 9 direction review

- [x] Review the scientific context, recent handover, open questions and
  relevant Exp46/52/57/63/74/76/77 records without running scientific fits.
- [x] Mirror the agreed two-agent rule in AGENTS.md in a private worktree.
- [x] Propose a complementary deposit-reach and aperture-budget diagnostic;
  leave Exp78's size objective and experiment ID with Claude.

Review: the proposal in `doc/next_direction_review_20260909.md` introduces no
new fitted parameters in its first phase and makes no improvement claim.
Execution requires a separately declared experiment, sample and test plan.

- [x] Explain the implemented correction, parameter accounting, calibration,
  and prospective forward-model use in a reproducible PDF note.
- [x] Render and visually inspect every page; verify formulas against code.

Scope: documentation only in the Exp77 worktree. Preserve science artifacts,
sample membership, fitted models, library interfaces, and other worktrees.
The architecture specification is unchanged: no integration is implemented.

Review: the six-page note distinguishes four per-epoch predicted coordinates
from 140/220 fitted affine-map coefficients and from downstream deformation
parameters. It documents the existing annular loss and the missing production
integration/scatter layer. No scientific fits or sample changes were made.
