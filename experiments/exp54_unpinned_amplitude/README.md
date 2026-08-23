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

## Stage 3.2 — the controlled factorial (`stage32.py`), COMPLETE

45 cells (5 efficiencies x 3 families x 3 size laws), n=240 mass-stratified,
z=0.4 and z=2.0, **ranked by a seven-criterion judge, never by loss**.
205.9 min.

| rank | model | mean | sA(0.4) | sA(2.0) | sF(0.4) | sF(2.0) | ident | tilt |
|---|---|---|---|---|---|---|---|---|
| 1 | **gompertz_log-E2-S2** | 13.29 | 1.183 | 0.907 | 0.886 | 0.703 | **8.0e-03** | **-0.0007** |
| 2 | sersic-E5-S2+S3 | 14.00 | 1.202 | 0.864 | 0.856 | 0.717 | 8.8e-04 | -0.0004 |
| 3 | moffat-E2-S2 | 14.29 | 1.182 | 0.907 | 0.884 | 0.709 | 7.8e-03 | +0.0028 |
| 4 | moffat-E2-S2+S3 | 15.57 | 1.183 | 0.905 | 0.872 | 0.702 | 1.1e-03 | +0.0033 |
| ... | | | | | | | | |
| 43-45 | **all three E4 cells** | 33.6-34.0 | | | | | 1e-07 | |

**THE JUDGE AND THE LOSS DISAGREE, EXACTLY AS THE RULE ANTICIPATES.** Ranking
by loss would pick `moffat-E5-S2+S3` (1.7010), which the judge places **7th**,
and `gompertz_log-E5-S2+S3` (1.7116), which it places **11th**. The
loss-optimal models are precisely the ones with degenerate identifiability
(8.2e-04 to 2.4e-03). The judge's winner has **8.0e-03** — an order of
magnitude better conditioned — for a loss only 0.04 higher.

**Five findings.**

1. **The transport-replacement question is ANSWERED, and not by the size law.**
   E1 leaves a halo-mass tilt of +0.024 to +0.041 dex/dex. Adding the
   mass x redshift cross-term (E2) collapses it to **-0.009 to +0.009** across
   every family and size law. Explicit mass conditioning of the size law (S3)
   does NOT close it (+0.035 to +0.041 under E1). exp53 concluded transport
   supplies the halo-mass-dependent concentration; what actually supplies it is
   a halo-mass-dependent *evolution of the efficiency*, which shifts WHEN mass
   is deposited and therefore WHERE, since deposit radius tracks `R200c(t_j)`.

2. **E4, the free spline, is the worst of all five efficiency forms** — the
   bottom three cells overall, with identifiability 1.2e-07 to 7.5e-07, five
   orders of magnitude worse than E1/E2, for a loss gain of ~0.007 over three
   extra parameters. **Prediction 4 confirmed: no peak in the efficiency's
   redshift dependence is identified.** The data prefer a power law.

3. **E1+E3 (concentration in the efficiency) does not earn its place** — it
   appears nowhere in the top 15, gains ~0.009 in loss, and leaves the tilt
   unchanged. Stage 1 showed concentration excess helps an amplitude
   *regression* by 0.011 dex; inside the physical model it does not.

4. **E2 subsumes S2a.** At Stage 3.1 with E1, the NFW scale radius beat the
   halo boundary. With E2 present, plain `S2` wins and `S2a` falls to ranks 9,
   14, 15 — both encode a mass-dependent size/timing and E2 does it better and
   far more identifiably.

5. **The profile family barely matters** — gompertz_log, moffat and sersic all
   appear in the top 4, with a loss spread of ~0.05 across families against
   ~0.08 across efficiency forms. This is the exp48 verdict ("the objective is
   the lever, the profile is not") reproduced in an unpinned model.

**Short list for Stage 3.3**: `gompertz_log-E2-S2`, `sersic-E5-S2+S3`,
`moffat-E2-S2`, `moffat-E2-S2+S3`.

## Stage 3.3 — five epochs (`stage33.py`), COMPLETE

Short list on n=1199 at all five epochs. `score_A` = 1 means "as good as the
best halo-only regression"; `score_F` is the raw shape loss over Stage 0's
incumbent reference.

| model | p | loss | sA(0.4) | sA(0.7) | sA(1.0) | sA(1.5) | sA(2.0) | growth bias |
|---|---|---|---|---|---|---|---|---|
| **gompertz_log-E2-S2** | 7 | 1.6547 | 1.128 | **0.907** | **0.899** | **0.929** | **0.970** | +0.032 |
| sersic-E5-S2+S3 | 11 | 1.6472 | 1.151 | 0.921 | 0.906 | 0.912 | 0.924 | +0.044 |
| moffat-E2-S2 | 7 | 1.6619 | 1.127 | 0.907 | 0.899 | 0.929 | 0.971 | +0.035 |

**Three headline results.**

1. **The physical model BEATS the best halo-only regression at four of five
   epochs** (`score_A` = 0.90-0.97 at z >= 0.7), losing only at z=0.4 (1.13).
   Seven global parameters, no per-object freedom, no pin — against a
   regression refitted separately at every epoch, and it predicts the profile
   as well.

2. **Mass growth is right to +0.032 dex.** The pinned incumbent needed a
   **+0.398 dex** correction to its own assembly. That is a **12x** improvement
   in the one quantity the pin was entirely concealing.

3. **The monotonic floor is NOT the limitation.** Raw shape loss runs 0.145 at
   z=0.4 to 0.113 at z=2, i.e. **3.2x to 2.5x the floor of 0.0456**. The model
   is limited by its functional forms, not by the deposition-only assumption.

**The Stage 3.4 diagnosis — the residual has exactly two components.**

*(a) An inner deficit at every epoch.* All three models under-predict inside
~7 kpc, reaching **-0.10 dex (-21%) at 3 kpc**, over-predict by +0.02 at 20-70
kpc, and recover by 148 kpc. This is the project's standing compact-central
defect (exp47/48/53), now at -21% instead of -39% to -47%, but unmistakably the
same shape. **No profile family in the shootout fixes it** — gompertz_log,
moffat and sersic give the same curve.

*(b) A z=2 amplitude deficit that tilts with halo mass.* Median amplitude bias
is small at z <= 1.5 (-0.008 to +0.010 dex) and falls to **-0.026 dex at z=2**,
with a halo-mass tilt of **-0.061 to -0.122 dex/dex at every radius** —
radius-independent, so it is an amplitude failure, not a shape failure.

**The two candidate fixes now have evidence, and they address different
components:**

- **E5 partially fixes (b) and worsens (a)**: it closes the z=2 outer tilt
  (-0.030 [-0.060, +0.003] at 148 kpc, CI includes zero, against E2's -0.078)
  but opens an inner tilt of -0.042 to -0.057 at 5 kpc **at all five epochs**,
  and costs 4 extra parameters, multi-start spread, and 10x worse
  identifiability.
- **A COMPACT second channel** (§4.5) targets (a) directly.
- **A DELIVERY-DELAY kernel** (§4.6) targets (b): if stars arrive later than
  the halo mass that formed them, the z=2 budget is reduced in exactly the
  mass-dependent way observed.

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

---

## Stage 3.4 (2026-08-23) — two corrections and the selection control

Two defects in the SCORING were found before any model extension was built.
Both change what the earlier stages mean; neither touches the model's physics.

### Correction 1 — `M200c` is already log10 (`dc226b0`)

`halo.py` and `scoreboard.py` read the halo-structure catalog's `M200c` column
as a linear mass in `1e10 Msun/h` when it is already `log10(M/Msun)`, so the
Diemer19 reference concentration was evaluated **2.0006 dex too low** (verified
against `population.npz["logmh"]`, which the column reproduces to max |diff| =
0.0000 at snap 72). The concentration excess `log10[c200c / c_Diemer19]` gained
a spurious 0.15 dex redshift ramp and a halo-mass slope that changes sign at
z = 0.4 (−0.053 as coded against +0.047 correct, dex/dex).

**Scope.** After removing a linear trend in `log10 Mh`, the buggy and correct
variables agree to correlation 0.9987–0.9995, so any form with a linear mass
term absorbs most of it. What consumes the variable at all is `E3`
(`log_eps` adds `k*dlogc`), **`S2a`** (`r50_of` uses `R200c/c_eff` for every
size except exactly `S2`) and `S3` (`r50_of` adds `dlogc` again) — **33 of 45
factorial cells**. `_assert_m200c_is_log10` now makes the units a test rather
than a comment: it checks the column against the project's independently
derived halo mass and refuses to proceed above 1e-3 dex.

### Correction 2 — `SIGMA_A` was inflated by one galaxy (`91a26f3`)

Row 181's measured `M*(<100 kpc)` collapses from 10^11.712 at z = 0.4 to a flat
~10^7.96 at all four higher-redshift epochs — a broken cross-match, 2.5 dex
below the sample's 0.1st percentile. It is in Stage 1's 2397, which sets
`SIGMA_A`, but in **none** of the model-fitting subsamples (300 / 240 / 1199,
all of which take every second galaxy; 181 is odd).

| best halo-only row | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| as published | 0.1047 | 0.1359 | 0.1417 | 0.1456 | 0.1744 |
| row 181 rejected | 0.1044 | 0.1120 | 0.1204 | 0.1279 | 0.1611 |

| `gompertz_log-E2-S2` score_A | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| as published (1199) | 1.128 | 0.907 | 0.899 | 0.929 | 0.970 |
| corrected, same 1199 | 1.131 | 1.100 | 1.059 | 1.058 | 1.050 |
| corrected, 1197 held out | 1.114 | 1.079 | 1.052 | 1.070 | 1.077 |

**The adopted model does not beat the best halo-only regression at any epoch;
it is 5–13% worse.** `scoreboard.py` now rejects such histories at source
(`MAX_BACKWARD_DEX = 3.0`, against a largest genuine backward growth of 2.03
dex) and stores the `rows` it kept.

**There is no generalization gap.** The NMAD of the amplitude residual is
identical between the fitted and held-out halves at every epoch
(0.1136/0.1178/0.1226/0.1310/0.1530 against 0.1156/0.1202/0.1218/0.1305/0.1507).
Seven global parameters generalize, as they should.

### `selection.py` — the progenitor-selection control

**Completeness** = `N(ours) / N(all TNG300 central haloes)` in bins of
`log10 M200c`, read from the raw catalogs (346k–460k centrals per epoch). The
z = 0.4 ceiling is **0.781**, set by the CoG and cross-match quality flags,
which are evaluated on the z = 0.4 galaxy and therefore cost the same at every
epoch. The **mh-complete cut** is where completeness first reaches 60% of that
ceiling and stays there.

| | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| cut, `log10 M200c` | >13.00 | >13.00 | >13.00 | >12.90 | >12.80 |
| galaxies kept | 2397 | 1780 | 1435 | 1144 | 839 |

At z = 0.4 the cut keeps everything — the right sanity check, since there is no
progenitor bias at the selection epoch.

**The z = 2 halo-mass tilt is an artifact of the incomplete regime.** With
theta frozen:

| tilt at z=2 [dex/dex] | 5 kpc | 10 kpc | 75 kpc | 148 kpc |
|---|---|---|---|---|
| full sample | −0.135 | −0.075 | −0.083 | −0.094 |
| mh-complete sample | −0.011 | **+0.066** | **+0.099** | **+0.089** |

It does not shrink, it flips — onto the same positive tilt the mh-complete sample
shows at z = 0.7–1.5, so removing the incomplete regime makes z = 2 consistent
with the other epochs rather than an outlier.

**The amplitude deficit is real and larger than reported**: the median of
`log10[model/truth]` at 100 kpc moves from −0.0255 to **−0.0421 dex** at z = 2.

**The mechanism is confirmed, not assumed.** A selection can only bias a
residual if the residual depends on the variable it truncated. That variable is
future growth `G_k = log10 Mh(z=0.4) − log10 Mh(z_k)`, which the z = 0.4
selection cuts on and the model — reading only epoch-local halo quantities —
cannot see. The partial correlation of the residual with `G_k` at fixed halo
mass is **+0.19 / +0.22 / +0.23 / +0.21** with **dy/dG = +0.18 / +0.16 / +0.15
/ +0.15 dex per dex** at z = 0.7 / 1.0 / 1.5 / 2.0.

**A delivery-delay kernel fitted to the z = 2 tilt would have been fitting the
survey.** The amplitude deficit is the part that survives the control.

### A caveat on the high-redshift ceiling

Completeness at the massive end settles near 0.61–0.65 at z = 1–2 rather than
returning to 0.781. Halo mass only grows, so a 10^13.5 halo at z = 2 certainly
clears the z = 0.4 threshold and the shortfall is not the growth selection: it
is that the sample follows **one main progenitor per galaxy**, so when two
massive high-redshift haloes merge into a single z = 0.4 galaxy only one is
counted. Those haloes do end up inside a massive z = 0.4 galaxy, which is a far
more benign kind of missingness.

### How far the `SIGMA_A` correction reaches — the shape side is intact

Row 181's curve of growth is broken in shape as well as amplitude: `F(<R)` pins
to exactly 1.0000 from 23 kpc outward at every epoch above z = 0.4, i.e. all of
its mass is inside 23 kpc. But its leverage on the two metrics differs by three
orders of magnitude:

| metric | shift when row 181 is removed |
|---|---|
| `SIGMA_A` (amplitude benchmark) | −0.3% / **−17.6%** / **−15.0%** / **−12.2%** / **−7.6%** |
| population shape loss | **+0.043%** |

The asymmetry is structural. The amplitude residual is `log10(model/truth)`,
unbounded, and row 181's is 3.76 dex, so its square swamps 2396 others. The
shape residual is `(m−d)/d` on a curve of growth normalized to 1 at 100 kpc, so
it lives near unity and no single galaxy can move the mean far.

**So `L_F_REF` and every `score_F` stand**, including the Stage 3.3 shape scores
(0.919 / 0.916 / 0.875 / 0.797 / 0.716). The correction reaches the amplitude
comparison and nothing else.

### The 45-cell factorial, re-judged under both corrections

204.7 min, all 45 cells refitted (`stage32_rejudge.py`). Three predictions were
put on record in the script's docstring before the run; all three held.

**1. `gompertz_log-E2-S2` is still the judge's winner** (mean rank 12.86, was
rank 1 before). The adopted model survives both corrections.

**2. E1+E3 improves sharply but does not reach the top six.** The four largest
upward rank moves in the whole factorial are all E1+E3 cells:

| cell | was | now |
|---|---|---|
| `moffat-E1+E3-S2+S3` | 29 | **13** |
| `gompertz_log-E1+E3-S2+S3` | 34 | 18 |
| `moffat-E1+E3-S2` | 37 | 23 |
| `gompertz_log-E1+E3-S2` | 36 | 22 |

So the concentration excess WAS being unfairly penalised by the corrupted
variable — the earlier verdict "E1+E3 never reaches the top 15" was partly an
artifact — but correcting it does not make concentration competitive with the
mass x redshift cross-term.

**3. E2 still replaces transport; E3 still does not.** The halo-mass tilt at
z = 0.4 is **+0.003 to +0.012 for every E2 cell** and **+0.0396** for the best
E1+E3 cell. A concentration term does not collapse the tilt; the cross-term
does. (This tilt is measured at z = 0.4, where `selection.py` shows there is no
progenitor bias, so it is clean.)

**Unchanged verdicts.** E4, the free spline, is still the worst family in the
factorial — `sersic-E4-S2` last at 35.14, the E4-S2a cells at 33.57, with
`ident` ~1.5e-07. No peak in the efficiency's redshift dependence is identified.

**The judge still overrules the loss.** Loss-ranking picks `moffat-E5-S2+S3`,
`gompertz_log-E5-S2+S3` and `sersic-E5-S2+S3` — `ident` 7.8e-04, 8.9e-04 and
8.7e-04 against the winner's **7.8e-03**, an order of magnitude better
conditioned. Judge ranks 2, 10 and 6.

Every cell's loss rose by a near-uniform 0.07–0.09, which is the expected
signature of the smaller `SIGMA_A` rather than of the concentration fix: a
smaller denominator raises `score_A`, and the loss is
`score_A**2 + score_F**2`. A uniform shift cannot reorder a ranking, which is
why the rank moves above are attributable to the concentration fix.

**Short list for Stage 3.3**: `gompertz_log-E2-S2`, `moffat-E5-S2+S3`,
`moffat-E2-S2`, `moffat-E5-S2`.

### The per-epoch mh-complete mask — S3 is fitting the selection, and so is E5

`stage33_perepoch.py` repairs the intersection's assembly bias by scoring each
epoch on its own fair galaxies: **7599 galaxy-epochs** against the
intersection's 3755, and no galaxy required to be massive at an epoch where it
is not scored. `MaskedProblem` subclasses `fit.Problem` and is gated by an
equivalence check — with an all-true mask it must reproduce the parent, and does
(|Δ| = 0.00e+00 on `score_A`, 4.4e-16 on `score_F`).

**1. THE S3 COLLAPSE IS REAL.** Two mh-complete samples of different construction, one
assembly-biased and one not, both drive all three size-law conditioning slopes
to near zero from large full-sample values:

| parameter | full | intersection | per-epoch |
|---|---|---|---|
| `f:logMh` | 0.0992 | 0.0032 | **0.0202** |
| `f:dlogc` | −0.3380 | −0.0165 | **−0.0242** |
| `f:fform` | −0.3342 | +0.0043 | **+0.0051** |

**S3 is absorbing progenitor-selection structure, not physics.** With the
factorial's separate finding that S3 does not collapse the halo-mass tilt the
way E2 does, that is two independent reasons to drop the size-law conditioning.

**2. AND SO, APPARENTLY, IS E5.** On the per-epoch fit, `moffat-E5-S2+S3` has a
z=2 tilt of +0.005 / +0.033 / −0.002 / −0.015 at 5 / 10 / 75 / 148 kpc, against
`gompertz_log-E2-S2`'s −0.126 / −0.076 / −0.089 / −0.099. E5's known advantage
was that it closes the z=2 tilt — but `selection.py` shows that tilt is the
survey, not the galaxies. **E5's four extra parameters buy a fit to a selection
artifact.** The case for the seven-parameter `E2-S2` is stronger after the
control than before it.

**3. THE MODEL IS WEAKEST WHERE THE DATA ARE BEST.** Scoring each epoch's fair
galaxies against a regression refitted on those same galaxies:

| `gompertz_log-E2-S2` score_A | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| full (progenitor-selected) sample | 1.131 | 1.100 | 1.059 | 1.058 | 1.050 |
| mh-complete sample, vs the same regression | 1.116 | **1.356** | **1.339** | **1.360** | **1.371** |

On a complete sample of massive haloes the model is ~36% worse than a halo-only
regression at z ≥ 0.7, against 5–10% on the full sample. The regression barely
notices the restriction (its scatter falls only 3–15%); the model degrades
sharply. **The progenitor-selected sample was flattering the model**, and the
deployment case is weaker than either the original or the corrected full-sample
numbers suggested.

The adopted model's own parameters barely move under the per-epoch fit
(`a0` −0.004, `a_z` +0.012); what does move is the mass dependence
(`a_M` −0.110, `a_Mz` +0.080), which is exactly the part the selection acts on.

---

## Stage 3.4 — the representational ceiling (`stage34_ceiling.py`, `stage34_objective.py`, `stage34_basis_scan.py`), COMPLETE (v2)

Stage 3.0 bounded the AMPLITUDE and killed a survival term before anyone built
it. This asks the same question of the SHAPE. **This section describes v2**,
rebuilt after an independent review found four defects in v1; v1's output is
kept at `outputs/stage34_ceiling_v1_provisional.npz` and its conclusions are
listed as withdrawn at the end.

### What was computed

For each galaxy the model's deposits are a fixed **basis**: one truncated
unit-mass cumulative profile per MAH step, `R50` from the fitted size law,
truncated at `3 R200c(t_j)`. Replace the seven-parameter weight law with a free
non-negative weight per deposit, respecting causality, and the residual bounds
what any deposition model with this basis could do given a perfect efficiency
law.

Three gates make the comparison legitimate, all at `0.00e+00`: the scorer
reproduces `stage33_perepoch.MaskedProblem.per_epoch`; `B @ (eps dMh)`
reproduces `model.forward`; and an all-true epoch mask reproduces the unmasked
solve.

**The sample quoted below is the fixed z=2-threshold cohort** — the 840
galaxies above the *z* = 2 mh-complete cut, followed at all five epochs, so
that an evolution statement is about the same objects. ("mh-complete" = above
the halo mass where our completeness first reaches 60 per cent of its z=0.4
ceiling of 0.781, i.e. absolute completeness ~0.47. A threshold, not
completeness 1.) The run also reports an `all5` and a `masked` definition;
the ladder's ordering is identical in all three.

**The objective matters and is not optional.** NNLS minimises a fractional
residual on the *absolute* curve of growth, not the production shape loss.
Because `score_F` and `score_A^2` are means over galaxies of per-galaxy terms,
each can be minimised galaxy by galaxy and the population minimum obtained
EXACTLY; `stage34_objective.py` does that with analytic gradients (checked
against finite differences to 1e-8).

### The ladder — `score_F`, cohort, lower is better

| bound | freedom/galaxy | mean over epochs |
|---|---|---|
| monotone in time, shape loss only | basis-free | **0.009** — *not a floor* |
| monotone in time, amplitude + shape | basis-free | 0.195 |
| monotone in time, absolute-fractional (exact PAVA) | basis-free | 0.286 |
| each epoch fitted alone (not causal) | 5 x ~71 | <= 0.164 |
| **one causal weight vector, EXACT shape loss** | **~71** | **0.282** |
| the same under the NNLS surrogate | ~71 | 0.538 |
| *control*: fitted to ANOTHER galaxy, exact loss | ~71 | **0.319** |
| 8 deposit-group amplitudes (surrogate) | 8 | 0.614 |
| 5 epoch-interval amplitudes (surrogate) | 5 | 0.762 |
| **the fitted model** | 0 (7 global) | **0.898** |

`score_A` on the cohort: exact amplitude bound 0.022, surrogate ceiling 0.150,
permuted control 0.200, **model 1.278**.

### What it says

**1. THE BASIS CAN DRAW THE PROFILES.** With independent weights per epoch it
reproduces every galaxy's curve of growth to a median 0.9–2.2 per cent RMS.
The profile family, the size law and the truncation are not the limitation, and
the fitted `(log_f0, b)` is already within 0.4 per cent of the ceiling-optimal
size law.

**2. THE COST IS THAT ONE CAUSAL VECTOR MUST SERVE ALL FIVE EPOCHS** — the
whole gap from <=0.164 to 0.282, with no change of basis.

**3. BUT ALMOST NONE OF THE MODEL-TO-CEILING GAP IS INFORMATION A GLOBAL LAW
COULD USE.** Under the exact objective, this galaxy's deposits fitted to a
DIFFERENT galaxy's measured profile reach **0.319** against the true ceiling's
**0.282**. The wrong galaxy costs 13 per cent. The dictionary is overcomplete;
halo-specific basis information is about an eighth of the ceiling's advantage.
(The control has no matching halo basis but full access to the target's stellar
data — it measures dictionary flexibility, not what a predictor could learn.)

**4. MONOTONICITY DOES NOT FLOOR THE SHAPE LOSS.** The production shape loss
renormalises at 100 kpc while monotonicity constrains the absolute profile, and
any set of shapes can be made monotone by inflating the later epochs. Optimised
against the shape loss alone the basis-free monotone bound is **0.009**. Stage
3.0's isotonic projection scored ~0.29 because it matched the amplitude.

**5. WHERE MONOTONICITY DOES BIND, PRICED IN ITS OWN UNITS.** RMS error in the
cumulative curve as a fraction of `M*(<R)` at the later epoch, 400-sample
bootstrap:

| interval | radii where the truth FALLS | cost of clipping | cost of the basis on the rise |
|---|---|---|---|
| z=2.0 to 1.5 | 32.2% | 3.68% [3.44,3.83] | 3.17% [3.08,3.26] |
| z=1.5 to 1.0 | 33.1% | 4.65% [4.43,4.97] | 2.43% [2.27,2.58] |
| z=1.0 to 0.7 | 38.2% | 2.35% [2.18,2.57] | 1.93% [1.80,1.98] |
| z=0.7 to 0.4 | 36.6% | 2.67% [2.53,2.87] | 1.76% [1.69,1.83] |

**The decline costs more than the basis at every interval.** v1 reported the
opposite because its solve minimised a differently-weighted quantity from the
one it printed.

**6. THE z=2 CENTRAL DEFICIT: THE TRADE-OFF IS STRUCTURAL, THE NUMBER IS NOT.**
Re-solving the basis-free monotone bound with the z=2 epoch up-weighted:

| z=2 weight | 0.2 | 1.0 | 2.0 | 5.0 | 20.0 |
|---|---|---|---|---|---|
| `score_F` | 0.209 | **0.188** | 0.193 | 0.289 | 0.477 |
| z=2 at 2 kpc | −23.2% | **−17.2%** | −13.2% | −6.0% | −0.9% |
| z=0.4 at 2 kpc | +4.6% | **+9.0%** | +10.4% | +25.4% | +40.4% |

A deposition-only model can put the z=2 residual anywhere on that curve. At the
production weighting the optimum is −17.2%; the model sits at −25.4%.

**7. AND THE MECHANISM SPLIT IS SHARPER THAN ANY NUMBER.** Preregistering
"declined" as a fall in `M*(<4.92 kpc)` from z=2 to z=0.4, median
`100 (pred-truth)/truth` at 2 kpc, z=2:

| group | n | model | ceiling | epoch-alone | monotone |
|---|---|---|---|---|---|
| centre declined | 443 (53%) | **−37.4** | −22.4 | +3.6 | −25.9 |
| centre did not | 397 | **−11.5** | −5.9 | +3.6 | −10.2 |

The population's −25.4% is mostly the declining half, which no static
non-negative deposition model can follow. Among non-decliners the model is only
11.5 per cent low and the ceiling recovers half of that.

**8. HOW MUCH TIME RESOLUTION A GLOBAL LAW WOULD NEED.** K contiguous deposit
groups, within-group distribution held at the fitted law's:

| K | 1 | 2 | 3 | 5 | 8 | 12 | 20 | free |
|---|---|---|---|---|---|---|---|---|
| mean `score_F` | 0.898 | 0.817 | 0.661 | 0.645 | 0.614 | 0.598 | 0.570 | 0.538 |
| at z = 2 | 0.777 | 0.777 | **0.584** | 0.565 | 0.568 | 0.605 | 0.603 | 0.581 |

K = 1 and K = 2 are an IDENTITY, not a result: every pre-z=2 deposit is in one
group, so rescaling it cannot change a profile normalised at 100 kpc, and the
z=2 score must equal the model's. **Most of the reachable improvement arrives
by K = 3–8**, a resolution a richer global redshift dependence could plausibly
express. This is the strongest positive lead the ceiling produced.

**9. THE FREEDOM AUDIT.** A median of **6 of ~71** weights are active — a
sparse, bursty history, not an even use of the nominal freedom. Capping
`eps_surv` at `f_b = Omega_b/Omega_m` costs **+0.002** in mean `score_F`, so
the ceiling **passes the universal-baryon mass-budget check** — a necessary
condition, not a sufficient one.

`moffat-E2-S2` reproduces the whole ladder to within 0.03.

### The contingent extension, rejected at the bound (`stage34_basis_scan.py`)

`c = c0 + c_z ln(1+z)` was scanned on a 7x7 grid of GLOBAL basis parameters
with the per-galaxy freedom held fixed, at two freedom levels. The optimum is
at **`c_z` = 0** in both, on an interior minimum; re-measured on all 2397
galaxies and on the 840 cohort the best non-zero `c_z` is worse by +0.001 and
+0.019. The ceiling-optimal size law is worth +0.4 per cent. **Not built** —
strong negative evidence against the proposed mechanism, not a proof of
impossibility.

### Figures

`figures/stage34_ladder_*`, `stage34_temporal_*` (the K ladder and its
identity), `stage34_galaxies_*` (four individual galaxies, two declining and
two not), `stage34_rms_*` (the distribution of per-galaxy error), and
`stage34_pareto` (the trade-off). All are generated from the saved full-sample
output; the smoke path writes a `_smoke` suffix so it can never overwrite them
again.

### WITHDRAWN from v1 of this section

1. ~~"Two thirds of the z=2 central deficit is irreducible."~~ It compared two
   medians from two particular compromises. See item 6.
2. ~~"The monotone floor is 0.29 in `score_F`."~~ 0.009 under the shape loss;
   the projection was matching the amplitude. See item 4.
3. ~~"Zero headroom at z=2."~~ An identity of the five-interval control. See
   item 8.
4. ~~"The basis cannot draw the changes between epochs."~~ With the increment
   solve in the units it reports, the decline costs more than the basis at
   every interval. See item 5.
5. ~~"The compact second channel is retired."~~ Deprioritised. See item 7.
6. The v1 ladder figure was regenerated from a 60-galaxy smoke sample.

The one v1 conclusion that survives unchanged is that a redshift-dependent
deposit shape does not earn a fit — and it now survives a full-sample
confirmation it did not have.
