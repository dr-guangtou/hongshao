# exp54 — remove the truth pin: predict the stellar mass, don't copy it

**Status: DATA PREP ONLY.** The fitting stages have not started. Plan:
`doc/plans/2026-08-22-exp54-unpinned-amplitude.md` (revision 2).

## What this directory holds so far

### `halo_structure.py` — the full TNG Halo Structure record

Downloads and reduces the TNG supplementary "Halo Structure" catalog
(Anbajagane et al., arXiv:2109.02713) for our 2397 galaxies, at **every
snapshot at or before our selection epoch**: 2, 3, 4, 6, 8, 11, 13, 17, 21,
25, 33, 40, 50, 59, 67, 72 — z = 11.98 down to 0.40.

Why it supersedes `exp46_highz_ridge/conc_history.py`: that kept five epochs
and seven hand-picked fields, enough for a question about the five profile
epochs. exp54 conditions each DEPOSIT on the halo state at its own formation
time (plan §4.4), and **39% of z=0.4 stellar mass is deposited at z>2**, so it
needs the whole history — and every field, discovered from the file rather than
hard-coded.

Snapshots 78/84/91/99 exist in the catalog and are **deliberately excluded**:
they postdate the SubLink tree root (snap 72), so there is no main-branch entry
to join on, and they are after the epoch we observe.

```
PYTHONPATH=. uv run python experiments/exp54_unpinned_amplitude/halo_structure.py plan
PYTHONPATH=. uv run python experiments/exp54_unpinned_amplitude/halo_structure.py fetch
PYTHONPATH=. uv run python experiments/exp54_unpinned_amplitude/halo_structure.py extract
PYTHONPATH=. uv run python experiments/exp54_unpinned_amplitude/halo_structure.py verify
```

**Raw files (~45 GB) stay OUTSIDE the repo**, at `$HONGSHAO_HALO_STRUCTURE_DIR`
(default `~/Desktop/tng300_halo_structure`). Only the few-MB reduced table is
written here. The fetch streams in 16 MB chunks so a 3 GB file never enters
memory, and renames from `.part` so a truncated file never appears under the
real name. Needs `~/.tng_api_key`.

`verify` re-runs the join validation: `c200c` at snap 72 must reproduce the
project's own `c200c` exactly, and the five overlapping epochs must agree with
exp46's `conc_history.npz`.

**DONE (2026-08-22).** 27 GB fetched, all 16 snapshots, extracted to
`outputs/halo_structure_history.npz` (2.9 MB, all 14 fields). Verification is
exact: max |diff| = **0.000e+00** against the project's own `c200c` at snap 72
(100% exact, all 2397) and **0.000e+00** against exp46's five epochs.

> **KEEP THE RAW FILES until exp54 is finished.** `experiments/*/outputs/` is
> gitignored, so `halo_structure_history.npz` is NOT in version control, and
> regenerating it needs the 27 GB cache — a ~3 hour download over a link that
> stalled 14 times. Either keep `~/Desktop/tng300_halo_structure`, or copy the
> 2.9 MB npz somewhere durable before deleting it.

Coverage by valid NFW fit, on the full set: 2397/2397 at z=0.4, 2384 at z=3,
1812 at z=5, 1083 at z=6, 482 at z=7, 158 at z=8, 5 at z=10, **zero at z>=11**.
Usable to z ~ 5, unusable beyond z ~ 6, and the high-z survivors are the
best-resolved haloes, so statistics on them are mass-selected.

## Measured before any fitting (see plan §3.2 and §4.4)

- The **absolute deposition law works**: `eps = 0.00318 (Mh/10^13.5)^-0.190
  (1+z)^0.817`, 3 global parameters, 5-fold CV **ties** the epoch-matched
  halo-mass baseline with <=0.004 dex bias.
- **Diemer & Joyce (2019)** (`colossus`, now a dependency) adds nothing over
  `logMh(z_k)` because it is deterministic in `(M, z)`. The measured
  concentration's entire gain lives in the **residual**, so the adopted feature
  is the dimensionless excess `log10[c_measured / c_D19(Mh, z)]`.
- The **deposit-size clock is badly posed**: `f = R50/R200c` runs 0.0013 at
  z=8-20 to 0.49 at z=0.5-1, spread 0.913 dex. Proposed replacement
  `R50_i = f0 * R200c(t_i) * (1+z_i)^b`, where `b = 0` is a testable null.

Reproduce the amplitude measurements with
`doc/plans/2026-08-22-exp54-evidence-probe.py` (~4 min, writes nothing).
