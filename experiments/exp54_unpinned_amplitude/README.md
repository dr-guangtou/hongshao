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

## Stage 0 — the contract (`contract.py`), PASSING

```
OK   1 poison / current kernel (expected FAIL)     FAIL   300/300 non-finite
OK   2 poison / absolute law                       PASS   0/300 non-finite
OK   3 mass conservation                           PASS   1.4e-10
OK   4 horizon bookkeeping                         PASS
OK   4b tail convergence (diagnostic)              PASS
OK   5 equivalence bridge                          PASS   0.000e+00
OK   6 pin-radius offset                           PASS   -0.027233 (-14.69%)
```

Horizon bookkeeping, nothing renormalized away: of each galaxy's deposited
stellar mass, median **51.7%** lands inside the 100 kpc anchor, 8.4% between
100 and 148, 21.9% between 148 and 500, and **18.0% beyond 500 kpc** — what
the pin used to restore for free.

Tail convergence: the fitted Moffat's gam = 1.378 gives an outer density slope
of -2.76, so enclosed mass converges only as R^-0.76. **A deposit with a 50 kpc
half-mass radius needs 9.6 Mpc to enclose 99% of its own mass**, 201 Mpc for
99.9%. A property of the PROFILE FAMILY, and the direct cause of the 18% lost.

## Stage 1 — the amplitude scoreboard (`scoreboard.py`)

Target `log10 M*(<100 kpc)`, n=2397, 5-fold galaxy-level CV on identical folds,
all five epochs, scored with RMS **and** CRPS **and** 90% interval coverage.
Official DiffMAH re-fit recovers 2397/2397 at median 5.8e-4 dex.

| row | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 | CRPS |
|---|---|---|---|---|---|---|
| do-nothing | 0.2562 | 0.2754 | 0.2872 | 0.3074 | 0.3657 | 0.1635 |
| catalog `logMh(z=0.4)` | 0.1401 | 0.1815 | 0.2031 | 0.2392 | 0.3039 | 0.1147 |
| official DiffMAH curve at z_k | 0.1209 | 0.1459 | 0.1488 | 0.1552 | 0.1820 | 0.0777 |
| official DiffMAH 4 params | 0.1130 | 0.1436 | 0.1516 | 0.1642 | 0.1916 | 0.0790 |
| + concentration excess | 0.1051 | 0.1364 | 0.1451 | 0.1604 | 0.1870 | 0.0752 |
| **+ fz2  (best halo-only)** | **0.1047** | **0.1359** | **0.1417** | **0.1456** | **0.1744** | **0.0713** |
| our exp10 DiffMAH [robustness] | 0.1068 | 0.1460 | 0.1718 | 0.2027 | 0.2286 | 0.0897 |
| 3p absolute law | 0.1231 | 0.1463 | 0.1473 | 0.1504 | 0.1802 | 0.0770 |
| 4p + mass x redshift | 0.1176 | 0.1431 | 0.1456 | 0.1510 | 0.1795 | 0.0759 |
| 4p + concentration excess | 0.1223 | 0.1452 | 0.1460 | 0.1493 | 0.1783 | 0.0764 |
| **5p + both (best law)** | 0.1178 | 0.1426 | 0.1448 | 0.1498 | 0.1777 | 0.0756 |
| 3p law, SHUFFLED MAHs | 0.1247 | 0.1647 | 0.1897 | 0.2372 | 0.3133 | 0.1110 |
| DiffMAH 4 params, SHUFFLED | 0.1228 | 0.1601 | 0.1799 | 0.2192 | 0.2870 | 0.1030 |

Coverage runs 0.907-0.933 against a nominal 0.90 — mildly over-covered, so the
predictive spreads are honest.

**Three findings.**

1. **The physical law is close but still behind the best regression**: 0.1178
   vs 0.1047 at z=0.4 (0.013 dex), narrowing to 0.1777 vs 0.1744 at z=2.0
   (0.003 dex). CRPS 0.0756 vs 0.0713. It does BEAT the epoch-matched halo mass
   at z=1.0 and z=1.5.

2. **THE LAW THROWS AWAY MAH-SHAPE INFORMATION THAT A LINEAR REGRESSION
   EXTRACTS.** Under the shuffle control at z=0.4, the DiffMAH-4-parameter
   regression degrades 0.1130 -> 0.1228 (**0.0098 dex of genuine history
   information**) while the 3p law degrades only 0.1231 -> 0.1247 (**0.0016
   dex**). The law recovers about **one sixth** of the shape information the
   same MAH supports. Its single-integral-with-a-smooth-efficiency form is too
   rigid to use the history — a direct argument for the redesign.

3. **Our exp10 DiffMAH fit is tuned to z=0.4 and degrades with redshift**:
   0.1068 at z=0.4 (marginally better than the official 0.1130) but 0.2286 at
   z=2.0 against the official 0.1916. Confirms the plan §4.4.5 decision to make
   the official fit primary.

**A protocol note worth carrying.** This shuffle control REFITS on the
shuffled data. An earlier probe froze the parameters and got 0.1324 rather than
0.1247 at z=0.4. Both are valid but answer different questions, and the refit
version is the correct permutation test — the same frozen-vs-refitted
distinction that decided exp52/exp53's mechanism claim.

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
