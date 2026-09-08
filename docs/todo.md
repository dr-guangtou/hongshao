# Documentation task: Exp77 model explanation

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
