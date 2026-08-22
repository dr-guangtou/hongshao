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

## Stage 2 — re-baselining the record (`rebaseline.py`)

Four products, **identical shape, no refit anywhere**, differing only in what
the profile is scaled to: `oracle-500` (the exp53 incumbent), `oracle-100`
(same, pinned at the measured anchor), `hybrid-mean` (out-of-fold predicted
amplitude), `hybrid-draw` (+ a cross-epoch-correlated draw). Amplitude provider
is Stage 1's best halo-only row; its residual correlation across epochs runs
0.60 (0.4-0.7) down to 0.12 (0.4-2.0), which is why the draw must be correlated.

**Read the column that matches the metric — getting this backwards inverts the
conclusion.** Paired metrics (tier 1+2 scatter, tier 3) compare a model galaxy
to its own truth, so read `hybrid-mean`; a draw is a realization, not a
prediction of that galaxy, and inflates paired error by ~sqrt(2). Distribution
metrics (tiers 2b/2c/2d/2e, all energy ratios) are pairing-blind, so read
`hybrid-draw`; `hybrid-mean` is under-dispersed by construction and is
penalized for it.

| metric | oracle-500 | hybrid | change |
|---|---|---|---|
| tier 3 max\|rel\| z=0.4 (paired) | 0.1951 | **0.2690** | **+38%** |
| M(<100) dex scatter z=0.4 (paired) | 0.029 | **0.105** | was pinned |
| tier 2b plane z=0.4 (distribution) | 2.17 | **2.20** | **+1%** |
| tier 2d mass-size R50 z=0.4 (distribution) | 5.09 | 5.17 | +2% |
| **tier 2c Mtot growth 0.4->1.0** | **0.27** | **1.23** | **the pin's hiding place** |

**Three findings.**

1. **The plan's headline cost estimate was wrong, and in an instructive way.**
   It predicted the tier 2b z=0.4 plane would move 2.1 -> 2.9 under a 0.137 dex
   amplitude error. Measured: **2.17 -> 3.38 if the error is injected into the
   MEAN, but 2.17 -> 2.20 with a properly cross-epoch-correlated DRAW.** The
   predicted degradation is largely an artifact of applying a paired operation
   to a pairing-blind metric. Population-level conclusions in exp38/40/47/48/53
   are far more robust to unpinning than the plan assumed.

2. **Per-galaxy conclusions are NOT robust.** Tier 3 profile max|rel| degrades
   +38% at z=0.4 and +91% at z=2.0 (0.1648 -> 0.3150). Any exp38-53 claim about
   an individual galaxy or a paired residual needs the `hybrid-mean` column.

3. **The pin's real hiding place is the cross-epoch GROWTH tier.** `Mtot`
   growth energy ratios sit at **0.27-0.33** under the pin — *below the truth's
   own split-half floor*, i.e. "indistinguishable from reality", achieved by
   construction rather than merit. Unpinned they are **1.22-1.60**. `R_half`
   growth is identical in all four columns (9.19-12.29), a clean sanity check
   that only the amplitude moved.

## Stage 3.0 — pre-fit floors (`stage30.py`), COMPLETE

**The monotonic-addition ceiling.** Any sum of static positive deposits makes
`M*(<R)` non-decreasing in time; TNG violates this in 50-58% of galaxies at
4.9 kpc per epoch interval. In production-objective units the best possible
monotone model scores **0.045590** against the incumbent's 0.158196 — 28.8% of
the loss, 8.3% of its variance. **But it is entirely an inner-profile
phenomenon**: median isotonic correction 0.0485 dex at 2 kpc, 0.0163 at 4.9,
0.0059 at 10.2, and **0.0000 beyond ~20 kpc**.

**Signal or noise? RESOLVED — do not build a survival term.** The violations
are ~2x the CoG measurement sigma (not instrumental), but at 4.9 kpc the sign
agreement between consecutive intervals is **52.1% against 50% for chance**,
with step-to-step r = **+0.013**. They are INCOHERENT. (At 103.5 kpc: 76.8%,
+0.163 — the outer profile does grow coherently.) They do not track MAH
burstiness (-0.101, wrong sign) or halo shape (+0.037), and galaxies losing
~4e9 Msun inside 10 kpc simultaneously GAIN 3-8e9 in total. A survival term is
smooth and monotone in dt and cannot produce incoherent sign-flipping
inner-only fluctuations — it would absorb the floor as a spurious mass-loss
signal.

**Tail diagnostics** confirm the review's caution: Moffat's `R99/R50 = 195` but
its observable fraction beyond 500 kpc is only 9.29%. The discriminating
observable is the fraction beyond the measured horizon — moffat 23.1%,
loglogistic 21.1%, richards 18.9%, gompertz_log 16.9%, sersic 15.1%, expo 3.9%,
gauss 0.19%.

**Endpoint-preserving shuffles** displace `Mh(z_k)` by a median 0.0037-0.0039
dex against **0.1884 dex** for the old `Mh(z=0.4)`-matched control at z=2 —
47.9x cleaner.

## Stage 3.1 — the minimal absolute model (`stage31.py`), COMPLETE

E1 x sersic x {S2, S2a}, n=300 mass-stratified, fitted jointly at **z=0.4 and
z=2.0**. Six parameters, no per-object freedom, no pin.

| | loss | score_A | score_F | n_eff | bootstrap |
|---|---|---|---|---|---|
| **S2** `R50 = f0 R200c` | 1.90521 | 1.0891 | 0.8480 | 6/6 | 0.9-15% |
| **S2a** `R50 = fs R200c/c200c` | **1.85863** | 1.0822 | 0.8291 | 6/6 | **b at 81%** |

**Amplitude, truth vs prediction: rms 0.123 dex at z=0.4 and 0.174 at z=2.0.**
Against Stage 1's best halo-only regression (0.1047, 0.1744) the physical model
is 17% worse at z=0.4 and **exactly equal at z=2.0** — with 6 global parameters
against a per-epoch refitted regression, while also predicting the profile.

**The identifiability protocol earned its place on its first use.** S2a has the
lower loss but its `b` is **not identified**: 81% bootstrap scatter, a profile
rise of 1.3e-2 against S2's 6.8e-2, and a smallest singular value of 2.8e-3
against S2's 1.1e-2 — 4x closer to degenerate. Ranking on loss alone would have
promoted a model with a meaningless parameter. `b` is degenerate in S2a because
`c200c` carries its own redshift dependence, which `(1+z)^b` then duplicates.

**Pre-registered predictions:**
- **2 CONFIRMED**: `b = -0.4527 +- 0.068` in S2, far smaller in magnitude than
  the incumbent's implied -3.
- **11 WRONG**: I predicted S2a would fit at least as well at low z and worse
  at high z. It fits better overall at both — but with a degenerate parameter,
  which is the more useful finding.

**The shape residual is structured and worsens with redshift** (figure
`s31_minimal_model`): at z=0.4 the model over-predicts at 2 kpc (+0.037 dex),
under-predicts at 4-6 kpc (-0.048), and recovers by 20 kpc; at z=2.0 it
under-predicts by -0.079 dex inside 4 kpc and over-predicts at 20-40 kpc. The
shape loss (0.131) is **2.9x the monotonic floor (0.0456)**, so the floor is not
yet binding and the family is still the limitation — which is what Stage 3.2
tests.

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
