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

## Stage 3.4 — how good could a model of this kind ever be? (`stage34_ceiling.py`, `stage34_objective.py`, `stage34_basis_scan.py`), COMPLETE (v2)

> **Read `doc/tech_note/deposition_model_2026-08.md` Section 8 instead if you
> want this in plain language.** What follows is the working record, written for
> someone editing the code: it uses the internal metric names (`score_F` is the
> profile-shape error divided by the previous generation's 15.8 per cent, so
> multiply by 15.8 to get a typical per-cent error of the normalised cumulative
> profile; `score_A` is the total-mass error divided by a plain statistical
> fit's scatter, so 1.30 means 30 per cent more scatter than a regression).

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

| bound | freedom/galaxy | `score_F` | = shape error |
|---|---|---|---|
| monotone in time, shape loss only | basis-free | **0.009** — *not a floor* | 0.1% |
| monotone in time, amplitude + shape | basis-free | 0.195 | 3.1% |
| monotone in time, absolute-fractional (exact PAVA) | basis-free | 0.286 | 4.5% |
| each epoch fitted alone (not causal) | 5 x ~71 | <= 0.164 | 2.6% |
| **one causal weight vector, EXACT shape loss** | **~71** | **0.282** | **4.5%** |
| the same under the NNLS surrogate | ~71 | 0.538 | 8.5% |
| *control*: fitted to ANOTHER galaxy, exact loss | ~71 | **0.319** | **5.0%** |
| 8 deposit-group amplitudes (surrogate) | 8 | 0.614 | 9.7% |
| 5 epoch-interval amplitudes (surrogate) | 5 | 0.762 | 12.1% |
| **the fitted model** | 0 (7 global) | **0.898** | **14.2%** |

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

### RE-DERIVED 2026-08-25 on the cleaned sample

Every number in this Stage 3.4 section was originally computed with the broken
cross-match at row 181 inside the sample, and with model parameters from a fit
that also contained it. Both corrected and the whole stage re-run
(`stage34_objective.py`, `stage34_ceiling.py`, `stage34_basis_scan.py`; pre-fix
outputs kept as `outputs/stage34_*_PRE_ROW181.npz`).

**Every rung moved by 0.01 to 0.03 percentage points.** At the precision quoted
here only two cells change: the free-deposit bound 4.5% -> 4.4%, and the
three-interval rung 10.5% -> 10.4%. The wrong-galaxy control is unchanged at
5.0%, the fitted model at 14.2%, the ceiling at 8.5%. The decline split is
unchanged at 443 declining of 839 (52.8%), with the model -37.2% inside 2 kpc
at z=2 for those and -11.5% for the rest. No ordering, ratio or conclusion
changes.

**`c_z` stays rejected**, and now on a clean sample: the best non-zero `c_z`
makes the ceiling WORSE by +0.0017 in mean shape score on all 2397 galaxies and
by +0.0201 on the 839-galaxy cohort, against +0.001 and +0.02 before.

Worth stating because it was not obvious in advance: row 181's profile is
essentially all inside the innermost aperture, the shape a deposit basis
reproduces worst, so it could have inflated the ceiling rather than merely
perturbing it. It does not, because a ceiling is a MEAN over 839 galaxies. The
contrast with the amplitude headline it *did* destroy (see the row-181 section
below) is the difference between a mean over a large sample and a mean over one
the completeness cut had already shrunk to 840.

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

---

## Stage 3.5 (2026-08-24) — a richer time dependence for the efficiency law (`stage35_time_law.py`, `stage35_figures.py`), COMPLETE

**The result is negative, and it is a clean one.** Giving the efficiency law a
richer dependence on redshift is worth **0.18 percentage points** of the
incumbent's 13.80% profile-shape error, and the new coefficients are not
determined. This was the leading positive lead out of Stage 3.4. It is now
closed.

### What was fitted

The adopted model `gompertz_log-E2-S2` writes the effective surviving-central
mass fraction as a single power law in `ln(1+z)`:

```
log10 eps_surv = a0 + a_M (logMh - 13.5) + a_z ln(1+z) + a_Mz (logMh - 13.5) ln(1+z)
```

Five nested enrichments were fitted, all sharing every parameter across
galaxies, on all five epochs, scored on the per-epoch mh-complete mask (2397 /
1781 / 1436 / 1145 / 840 galaxies, 7599 galaxy-epochs), ranked by the
seven-criterion judge:

| form | what it adds | extra parameters |
|---|---|---|
| `E6p1` `E6p2` `E6p3` | shifted Legendre polynomials in `ln(1+z)`, degrees 2, 2-3, 2-4 | 1, 2, 3 |
| `E7k2` `E7k3` | hinges at z = 1, 3 and 6 | 2, 3 |

Every basis function is **orthogonal to a constant and to `ln(1+z)`** over the
deposits' own redshift range and normalized so its largest value is 1, so its
coefficient reads directly as the biggest swing in `log10 eps_surv` that term
can contribute. That orthogonalisation is not cosmetic. Stage 3.2's `E4` —
a free piecewise-linear curve whose knot VALUES were the parameters — was the
worst of the 45 variants, and the reason is structural rather than statistical:
`a0 + interp(x, knots, k)` is **exactly degenerate**, because adding a constant
to `a0` and subtracting it from every knot gives an identical model. E6 and E7
cannot do that.

`model.py`'s self-check asserts the nesting **bit for bit**: with the new
coefficients at zero, E6, E7 and E8 reproduce E2 to `max |dM*(<R)| = 0.0 Msun`
over random thetas, five epochs, several galaxies and 24 radii. Every fit also
included the incumbent's own optimum with the new coefficients zeroed as one of
its five starts, so no variant could come out worse than E2 unless the
optimiser failed.

### The result

| variant | extra | shape error | vs E2 | loss | smallest singular value |
|---|---|---|---|---|---|
| `E2` (incumbent) | — | **13.803%** | — | 2.279273 | 7.5e-3 |
| `E6p1` | 1 | 13.722% | -0.08 | 2.272124 | 3.9e-3 |
| `E7k2` | 2 | 13.700% | -0.10 | 2.272731 | 1.9e-3 |
| `E7k3` | 3 | 13.638% | -0.17 | 2.264012 | 1.4e-3 |
| `E6p3` | 3 | 13.642% | -0.16 | 2.263729 | 2.6e-3 |
| `E6p2` | 2 | **13.623%** | **-0.18** | 2.267094 | 3.1e-3 |

The judge's order is `E6p3` > **`E2`** > `E6p1` > `E7k3` > `E6p2` > `E7k2`: the
incumbent comes **second**, ahead of three of the five enrichments, because
what the extra terms take off the shape error they give back in
identifiability. The target was 10-11%. Nothing came within 2.6 percentage
points of it.

### Why it stops there: the ceiling on ANY shared time law

`E8` gives the time dependence a **free** piecewise-linear curve on 6 or 8
knots, with the constant and linear directions projected out analytically. It
is not a candidate model and is kept out of the judge; it exists to bound every
shared time law from above.

```
incumbent 13.803%  ->  best low-order law 13.623%  ->  FREE shared curve 13.643%
```

A free curve with four extra coefficients does **not** beat two Legendre terms.
The 0.02-point difference is the size of the optimiser's own spread in 11-13
dimensions — `E8f8`, which has strictly more freedom than `E8f6`, came out
0.018 points **worse**, which can only be Nelder-Mead. So `E8`'s optimum is a
LOWER bound on the ceiling, stated rather than buried; the conclusion survives
it, because three independent constructions with one to six extra shared
coefficients all land between 13.62% and 13.66%.

**A shared time law of any shape is worth about 0.2 percentage points. Two
extra parameters already collect all of it.**

### The correction is spent where there is no mass

`figures/stage35_time_law.png` panels (a) and (b). Every variant leaves the
efficiency law essentially unchanged below z = 4 and then turns sharply upward,
reaching **+1.4 dex above the incumbent by z = 15**. Under the incumbent law
only **7.3%** of the deposited stellar mass is made above z = 4 and **0.98%**
above z = 7; of the accreted halo mass, 2.9% and 0.47%. The extra freedom is
being spent almost entirely on mass that is not there, which explains the
negligible gain and the poor identifiability at once.

### The identifiability report is the binding constraint, as predicted

`figures/stage35_identifiability.png`. The separation is complete — every new
coefficient is flat and unstable, every pre-existing parameter is steep and
stable, with no overlap in either coordinate:

| | conditional loss rise | bootstrap spread / \|value\| |
|---|---|---|
| the 7 parameters E2 already had | 0.14 to 39.4 | 0.9% to 8.1% |
| the 11 new time coefficients | 0.0035 to 0.047 | 10.6% to 201% |

Three coefficients are outright unstable: `E7k2:s0` at **201%** of its own
value, `E7k3:s1` at 61%, `E7k3:s0` at 53%. The best-behaved new coefficient
anywhere is `E6p3:c2` at 10.6%, still worse than every one of E2's own.

The enrichment also **degrades the parameters that were already there**. The
redshift slope `a_z`, which is the term the new coefficients are competing
with, goes from a 5.0% bootstrap spread in E2 to 8.9-12.1%; `a0` goes from 0.9%
to 2.4-3.0%. Adding the new terms buys 1.3% off the shape error and costs a
doubling of the uncertainty on the law's own redshift dependence.

### What did NOT get worse

The halo-mass tilt does not reopen. Measured on Stage 3.3's footing — full
sample, epoch-matched halo mass, `stage33.tilt_report` — the outer residual's
slope at 148 kpc moves from `-0.031` (E2) to `-0.039` (E6p2) at z = 0.4 and
**improves** at z = 2, from `-0.099` to `-0.078`. At 5 kpc and z = 2 it
improves more, `-0.126` to `-0.095`. The amplitude scores are unchanged to
within 0.01 at every epoch.

### The premise this stage was launched on was wrong

Stage 3.4's temporal ladder was read as "most of the reachable improvement sits
at a time resolution a shared law can express", because freeing the deposited
mass over K time groups moves the shape error 14.2% (K=1) -> 10.4% (K=3) ->
9.7% (K=8). **Every rung of that ladder is solved PER GALAXY**
(`stage34_ceiling.py`, `run_sample` loops over galaxies calling
`solve_galaxy`, and every rung inside it is a per-galaxy non-negative least
squares). So K = 3 buys its gain by giving each object its own three numbers,
not by finding a better universal curve. The tell was already in the output and
was not read: `bins1` is **identical** to the fitted model at every epoch
(13.80% on the masked sample), because one free amplitude per galaxy cancels
out of a profile renormalised at 100 kpc.

The right comparison for a shared law is `E8`, and it says the budget was 0.2
points, not 3.3.

### The speed change, checked rather than assumed

`StackedProblem` evaluates all 2397 galaxies as one array instead of looping in
Python — every galaxy carries the same 72 merger-tree steps, so the deposits
stack into (2397, 72). `check_stacking` compares the whole predicted profile
array against `fit.Problem.predict` galaxy by galaxy at several thetas and
refuses to run if they differ: **max relative difference 1.6e-15**, and the
masked scores agree with `stage33_perepoch.MaskedProblem` to 8.9e-16. The run
also verifies that Stage 3.3's stored E2 optimum reproduces Stage 3.3's stored
loss here **exactly** (2.2792734407262394, difference 0.0) before fitting
anything. `model.solve_u` now caches the truncation-inversion grid, which was
being rebuilt identically once per galaxy.

### Reproducing

```
export HONGSHAO_DATA_DIR=/path/to/tng300_mah_mprof
# one variant per process; each writes outputs/stage35_fits/<label>.npz
PYTHONPATH=. uv run python -u experiments/exp54_unpinned_amplitude/stage35_time_law.py \
    --only gompertz_log-E6p2-S2 --starts 4
# then, with every fit on disk, score and rank them
PYTHONPATH=. uv run python -u experiments/exp54_unpinned_amplitude/stage35_time_law.py
# the identifiability report for one variant (~25 min: 12 bootstrap refits)
PYTHONPATH=. uv run python -u experiments/exp54_unpinned_amplitude/stage35_time_law.py \
    --only gompertz_log-E6p2-S2 --ident
PYTHONPATH=. uv run python -u experiments/exp54_unpinned_amplitude/stage35_figures.py
```

`--smoke` writes nothing at all, and `--only` writes only its own per-variant
file, so neither can touch `outputs/stage35_time_law.npz` or the figures.

### The verdict

**Do not adopt any of these.** The incumbent `gompertz_log-E2-S2` stands. If
one were adopted anyway it should be `E6p2` — two extra parameters, the lowest
shape error, and the least badly determined of the enriched forms — but it buys
1.3% off one score at the price of doubling the uncertainty on `a_z`, and the
judge already ranks the incumbent above it.

**What this closes.** The efficiency law's dependence on time is not where the
remaining error lives. The next steps in `doc/todo.md` are re-ordered
accordingly: the objective's blindness beyond 52 kpc at z = 2 is now the
leading item, because a shared law cannot be shown to be inadequate by a loss
that cannot see where it fails.

---

## Stage 3.6 (2026-08-25) — repairing the loss, one cause at a time (`stage36_objective.py`), COMPLETE

**Negative, and it relocates the problem.** The loss really is blind in the
outskirts; repairing that blindness does not improve the outskirts. What the
comparison quantity does control is the overall LEVEL of central mass, and it
has no purchase whatever on that level's dependence on epoch — which is the
actual defect.

### The blindness, measured

Perturb the model's mass in one radial shell by ±1% and watch the loss move.
Reported as "how many times bigger must an error in this shell be to cost the
loss as much as one in the shell it watches most":

| objective | worst ratio over all shells and epochs | the innermost 2 kpc |
|---|---|---|
| production — cumulative profile, fractional residual, pinned at 100 kpc | **175,570** | the most-watched shell |
| surface density in log (exp48's winner) | ~60 over 2–148 kpc | **invisible** |
| shell decomposition in log (new) | 234, typically under 20 | seen |

The production loss's shape term divides each profile by its value at 100 kpc,
which PINS it there, so the three shells beyond the pin are nearly free. A few
isolated large cells elsewhere (a few hundred) are turning points where the
population gradient changes sign, not extra blind spots.

### exp48's winner is disqualified, and the reason was never checked

Building a surface density means differencing the curve of growth between grid
radii: 24 radii give 23 differences, and **the mass inside the innermost radius
is never one of them**. Worse, adding mass inside that radius raises every
later cumulative value by the same constant, so every difference is unchanged.

```
production  cog x frac      base 0.05349625   x5.0: 1.09025543
candidate   density x log   base 5.60673286   x5.0: 5.60673286
candidate   density x frac  base 9.52135432   x5.0: 9.52135432
```

Multiplying `M*(<2 kpc)` by five leaves it unchanged to a relative **1e-13** —
floating-point noise from a log10 round-trip inside `density_from_cog`, not a
response. `hongshao/objective.py` now asserts this in its self-check. That
aperture holds **12% of the stellar mass at z=0.4 and 32% at z=2**, and it is
where the model's largest known defect lives.

### The new comparison, and the two causes varied one at a time

`hongshao.objective` gains `quantity="shells"`: compare
`[M(<R_0), M(R_0..R_1), ...]`, which is **bijective** with the curve of growth,
so unlike a density it drops nothing. Paired with a log residual it holds a
shell carrying 1% of the mass to the same relative accuracy as one carrying 30%.

| cell | pin | quantity | what it changes |
|---|---|---|---|
| `production` | M*(<100 kpc) | cumulative, frac | nothing — the CONTROL |
| `pin-total` | the total | cumulative, frac | the pin only |
| `shell-log` | M*(<100 kpc) | shells, log | the quantity only |
| `shell-log-tot` | the total | shells, log | both |
| `density-log` | M*(<100 kpc) | density, log | exp48's winner |

Each cell gets its OWN `score_F` reference, fixed so the incumbent scores the
same in every cell, because dividing objectives on different raw scales by one
reference would silently change how hard the fit works on the amplitude.

### The result

**The pin is not a cause of anything.** `pin-total` moves the profile-shape
error by 0.005 percentage points and the central mass by 0.002 dex.

**The quantity moves the LEVEL of central mass and nothing else.** Median
log10(model/truth) inside 2 kpc:

| cell | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 | **span** |
|---|---|---|---|---|---|---|
| production | +0.020 | −0.003 | −0.021 | −0.062 | −0.126 | **0.145** |
| pin-total | +0.020 | −0.002 | −0.022 | −0.064 | −0.128 | 0.148 |
| shell-log | +0.116 | +0.095 | +0.074 | +0.031 | −0.035 | 0.150 |
| shell-log-tot | +0.117 | +0.096 | +0.076 | +0.033 | −0.033 | 0.150 |
| density-log | +0.149 | +0.128 | +0.108 | +0.065 | −0.003 | 0.152 |

Each shift is essentially **constant across epochs** — it varies by 0.006 dex
for `shell-log` and 0.008 for `density-log`, against shifts of 0.09 and 0.13.
A constant offset is absorbable by the efficiency normalisation and is not the
defect. **The span is the defect, and no objective moves it.**

### And the repair made the outskirts worse

| | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| median error beyond 52 kpc, production | −0.001 | −0.007 | −0.006 | −0.002 | −0.008 |
| the same, `shell-log` | −0.011 | −0.023 | −0.034 | −0.028 | −0.033 |
| outer density slope error, production | +0.12 | +0.10 | +0.09 | +0.03 | +0.21 |
| the same, `shell-log` | +0.16 | +0.14 | +0.12 | +0.06 | +0.24 |

Amplitude scores are identical across cells to within 0.005. **Nothing
adopted.** The production objective stands.

**The reading**: giving the loss eyes in the outskirts does not help because
the model appears to have no freedom that moves the outskirts independently of
the centre. That is a hypothesis, recorded in `doc/open_questions.md` as C1, and
it is not yet tested.

### Two gates that caught real bugs

Neither was decoration. The reference-consistency check caught the loss
normalisation being applied twice. The floor check caught the library's 1 Msun
shell floor silently truncating dimensionless profiles — every shell clipped,
every residual exactly zero — and then, on the full sample, caught **row 181**
(below). The production cell reproduces the incumbent's loss to 0.00e+00.

---

## 2026-08-25 — ONE CORRUPT GALAXY WAS INSIDE EVERY FIT FROM STAGE 3.3 ONWARD

Found by Stage 3.6's floor gate, which refused to run because a measured shell
contained 2e-4 Msun.

**Row 181's measured `M*(<100 kpc)` falls 3.76 dex** — from 10^11.72 at z=0.4
to a flat ~10^8 at all four earlier epochs. A broken cross-match, not a
progenitor. The next largest drop anywhere in the sample is 2.04 dex, so it is
isolated rather than the tail of a distribution.

**The project already knew, and already had the rule.** `MAX_BACKWARD_DEX = 3.0`
lives in `scoreboard.py` and `selection.py`, and `fit.py` carries a comment
explaining that `SIGMA_A` was recomputed without this galaxy. **The rule was
never applied in the model-fitting path.** `fit.py`'s statement that row 181 is
"in NONE of the model-fitting subsamples" was true when written and expired the
moment Stage 3.3 moved from a 1199-galaxy subsample to all 2397.

### What it cost

| | with row 181 | without |
|---|---|---|
| total loss, incumbent | 2.279273 | **1.874570** (17.8% was one galaxy) |
| amplitude score, overall | 1.2321 | **1.0557** |
| profile-shape error | 13.803% | 13.792% |

**Stage 3.5's conclusion is unaffected** — it turned on the shape error and on
identifiability, and the shape error moves by 0.011 percentage points, far
below the 0.18 under discussion. All variants were equally affected.

### The headline it manufactured

Section 5.3 of the technical note reports the model as ~36% worse than a plain
statistical fit at total stellar mass on completeness-controlled samples, and
calls it "the strongest argument against the model as it stands". Ratio of the
model's error to the statistical benchmark:

| | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| with row 181 | 1.12 | 1.31 | 1.29 | 1.30 | 1.16 |
| **without it** | 1.12 | **1.06** | **1.03** | **1.04** | **0.94** |

At redshift 2 the model **beats** the benchmark. The mechanism the note read as
physics — "the regression barely notices the restriction; the model degrades
sharply" — is arithmetic: the completeness cut shrinks the sample from 2397 to
840, so one fixed contaminant grows as a fraction of a mean square as the
sample shrinks. Meanwhile `SIGMA_A` had already been recomputed WITHOUT row
181, so the model was charged for a galaxy the benchmark was not.

### The fix, and what it obliges

`selection.sane_history_mask` promotes the existing rule to a shared function
and it is now applied in `stage35_time_law.py` and `stage36_objective.py`.
Stage 3.5's gate now has two halves — with the pre-fix mask it must reproduce
Stage 3.3's loss exactly, and with the mask it must differ — which proves the
sanity mask is the only thing that changed.

**Pre-fix outputs are archived, not deleted**, at
`outputs/stage35_fits_PRE_ROW181_FIX/` (with a `WHY.txt`),
`outputs/stage35_ident_PRE_ROW181_FIX/` and
`outputs/stage35_time_law_PRE_ROW181_FIX.npz`, because the Stage 3.5 conclusion
was drawn from them and the re-run must be comparable with them.

**Still owed** (`doc/open_questions.md`, A1 and A3): Stage 3.4's ceiling numbers
and Stage 3.3's reported scores were both computed on the contaminated sample
and have not been re-derived.

---

## Stage 3.7 (2026-08-25) — the size law's dependence on epoch (`stage37_size_epoch.py`), COMPLETE

Plan: `doc/plans/2026-08-25-exp54-stage37-size-epoch.md`.

**The defect is reachable — and reaching it empties the outskirts.** That trade,
not the candidates, is the result, and it is a direct quantitative argument for
a second deposit component.

### The defect, and the pre-check that licensed the stage

The model's median error in `M*(<2 kpc)` runs from **+0.020 dex at z = 0.4 to
−0.126 at z = 2** — right in the centre today, 25% too light in the centre
early. Stage 3.5 showed the efficiency law's time dependence cannot move that
span; Stage 3.6 showed no objective moves it either.

Before fitting anything, `--precheck` asked whether the span is real or is the
sample's mass composition moving — the completeness-controlled sample runs from
2397 galaxies to 839 and the survivors are more massive
(`figures/stage37_precheck.png`).

| | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 | span |
|---|---|---|---|---|---|---|
| every admitted galaxy | +0.020 | −0.003 | −0.021 | −0.062 | −0.126 | 0.145 |
| at fixed halo mass, 13.1–13.4 | +0.030 | +0.009 | −0.021 | −0.092 | −0.146 | **0.175** |
| at fixed halo mass, 13.4–13.7 | −0.001 | −0.014 | −0.053 | −0.132 | −0.193 | **0.192** |
| the same 750 galaxies at all five epochs | +0.001 | −0.030 | −0.048 | −0.093 | −0.129 | 0.131 |

**The span survives at fixed halo mass and grows.** The check was worth running
— the cohort's median halo mass does fall from 13.66 to 12.94 — but the
composition is not the cause.

**A second reading of the same table, not asked for.** The error also depends
on halo mass at fixed epoch, and that gradient **steepens with redshift**:
0.058 dex at z = 0.4, then 0.092, 0.153, 0.192. That is an interaction between
mass and epoch, which is the functional form of `S4` and not of `S5`. It was
recorded as a prediction before any fitting.

### The candidates, fitted under the production loss

`model.r50_of` gives every deposit `R50 = 10^(log_f0 + b log10(1+z)) * R200c`,
one exponent shared by every galaxy and halo mass. Three nested epoch axes were
added (all reproducing the plain law bit for bit at zero coefficients, asserted
in `model.py`): `S4`, the exponent depends on halo mass; `S5`, a low-order
polynomial in redshift; `S6`, a free curve.

| model | extra | **span** | vs base | shape% | >52 kpc | tilt | ident |
|---|---|---|---|---|---|---|---|
| incumbent | — | **0.145** | — | 13.787 | −0.005 | −0.0278 | 7.6e-03 |
| `+S4` | 1 | 0.141 | −0.004 | 13.728 | −0.005 | **−0.0215** | 3.6e-03 |
| `+S5p1` | 1 | 0.133 | −0.012 | 13.750 | −0.006 | −0.0297 | 2.6e-03 |
| `+S5p2` | 2 | 0.133 | −0.012 | 13.750 | −0.006 | −0.0296 | 3.0e-04 |
| `+S4+S5p2` | 3 | **0.132** | −0.013 | 13.724 | −0.005 | −0.0239 | 2.2e-04 |

The preregistered target was **below 0.10 dex**. The best reaches 0.132, a 9%
reduction. `S5p2`'s cubic coefficient comes out at **+0.0038** and its span,
shape error and outskirts match `S5p1`'s to three decimals — the extra term
buys nothing, exactly as in Stage 3.5.

**The pre-check's prediction was half right, and the half that failed is the
one that mattered.** `S4` wins the loss and the seven-criterion judge, and is
the only variant that *improves* the halo-mass tilt. But `S5` moves the span
further, and the span is this stage's declared criterion.

### THE BOUND WAS THE WRONG BOUND — an error, and its correction

`S6f6`, the free shared curve fitted under the production loss, returned the
**best loss of any cell** (1.8659) and a span of **0.139 — worse than a
one-parameter polynomial's 0.133**. That is not the optimiser failing. It is
this project's own standing lesson, re-learned: *a bound must be optimised on
the same quantity it is reported on.* The free curve was fitted for the loss and
read as a ceiling on the span, and nothing in the fit ever asked it to reduce
the span.

Refitted to minimise the span directly, with the loss required to stay within
5% of the incumbent's so the answer cannot be bought by making the model
uniformly bad:

```
minimise  span^2 + 100 * max(0, loss/loss_base - 1.05)^2
```

| | span | shape% | loss | tilt | ident |
|---|---|---|---|---|---|
| incumbent | 0.1454 | 13.787 | 1.874253 | −0.0278 | 7.6e-03 |
| `S6f6` fitted for the LOSS | 0.1390 | 13.729 | 1.865870 | −0.0288 | 1.2e-05 |
| **`S6f6` fitted for the SPAN** | **0.0605** | 14.216 | 1.968294 | +0.0365 | 1.2e-05 |

**A shared size law CAN remove 58% of the span.** The defect is reachable. The
candidates missed it because the loss barely feels it.

### What reaching it costs, which is the actual finding

| | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| central error, incumbent | +0.020 | −0.003 | −0.021 | −0.062 | −0.126 |
| central error, span solution | +0.012 | +0.012 | −0.001 | −0.025 | **−0.048** |
| beyond 52 kpc, incumbent | −0.001 | −0.007 | −0.006 | −0.002 | −0.008 |
| beyond 52 kpc, span solution | −0.039 | +0.003 | +0.023 | −0.035 | **−0.112** |
| outer slope error, incumbent | +0.12 | +0.10 | +0.09 | +0.03 | +0.21 |
| outer slope error, span solution | −0.01 | +0.02 | +0.03 | −0.05 | **+0.12** |

Buying the centre at redshift 2 — closing 0.078 dex of central deficit — costs
**0.104 dex of outskirt mass at the same epoch**, taking the model from 2% to
23% too light beyond 52 kpc. The outer *slope* improves markedly at the same
time, so the deposits are the right shape and in the wrong place, not the wrong
shape.

The solution is also **not adoptable on its own terms**: `log_f0` sits at its
bound (−0.0002 against a bound of 0.0), the smallest singular value is 1.2e-05
so the parameters are undetermined, the halo-mass tilt flips sign, and the
profile-shape error rises 0.43 points.

### The conclusion, and what it sets up

**With one radial scale per deposit, the central mass at redshift 2 and the
outer mass at redshift 2 cannot both be right.** Making early deposits compact
enough to fill the centre necessarily removes them from the outskirts, because
there is only one scale to move. The span is reachable, the outskirts are
reachable, and the same parameter controls both.

That is a direct, quantitative argument for **a second component with its own
scale** — an early compact channel that adds central mass while an extended
channel keeps the outskirts. It also matches, independently, what a parallel
experiment (exp55) finds from the profiles alone: an exponential core plus an
extended Moffat reproduces measured curves of growth as well as a
three-component decomposition. Two cautions carried into that next stage:

- exp55 constrains a **single epoch**; this model serves five, so only the hint
  about the inner component's shape transfers, not the conclusion.
- "Two components" need **not** mean every deposit carries both. The form that
  matches this stage's defect is an **early exponential deposition and a late
  Moffat-like deposition** — a mixture weighted by *when* the mass was laid
  down. That is a different object from the redshift-dependent deposit SHAPE
  (`c_z`) already rejected, and must not be conflated with it.

**Nothing adopted.** `S4` is the only variant the judge prefers to the
incumbent and it moves the span by 0.004 dex, which is not worth a parameter.

---

## Stage 3.8 (2026-08-25) — a second deposit channel with its own scale (`stage38_two_channel.py`), COMPLETE

**A flat negative, and it confirms the limit by a route that could have refuted
it.** The fit was free to use the compact channel at any strength and chose
zero, in all five fits, in both modes.

### What was added

Each deposit's unit profile became a mixture of two blobs at two sizes,
weighted by WHEN the mass was laid down:

```
F_j(<R) = (1 - w_j) * F_extended(R ; R50_j)  +  w_j * F_compact(R ; rho * R50_j)
```

`F_compact` is an exponential (Sersic n = 1, no free shape parameter), `rho < 1`
is the second radial scale, and `w_j` is the compact fraction — the share of
that deposit's stars placed in the small blob rather than the big one. `P1`
holds `w` constant; `P2` and `P3` let it rise toward early times through a
logistic in `ln(1+z)`. Both channels are truncated at the same boundary and are
individually unit-mass, so the mixture is unit-mass and Stage 0's
mass-conservation contract is untouched. `w0 = 0` reproduces the single-channel
model bit for bit, asserted at random parameters AND random `rho`.

### The result

| | mode | fitted `w0` | `rho` | loss |
|---|---|---|---|---|
| `P1expo` | production loss | +0.0002 | 0.197 | **1.874253** |
| `P2expo` | production loss | +0.0002 | 0.210 | **1.874253** |
| `P3expo` | production loss | +0.0000 | 0.198 | **1.874253** |
| `P1expo` | minimise the span | +0.0004 | 0.738 | 1.968106 |
| `P3expo` | minimise the span | +0.0014 | 0.474 | 1.968716 |

The three loss-mode fits return **exactly the incumbent's loss to six decimal
places**, and every diagnostic is numerically identical to the incumbent: span
0.145, non-declining span 0.045, outskirts −0.008, shape error 13.787%. The
`ident` column reads 0.0e+00 because with `w0 = 0` the size ratio `rho` has no
effect at all, making the Jacobian exactly singular — which is itself the
diagnostic.

The two span-mode fits reach 0.089 and 0.094, **worse than Stage 3.7's
one-scale size law at 0.061**, and reach it by moving `log_f0` (−0.843 →
−0.533) and `b` (−0.896 → −1.553) — the size law — not by using the channel.

### Why, and this is the point

Splitting on Stage 3.4's preregistered flag — did the galaxy's MEASURED mass
inside 4.92 kpc fall between redshift 2 and 0.4:

| median error inside 2 kpc | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 | span |
|---|---|---|---|---|---|---|
| all admitted | +0.020 | −0.003 | −0.021 | −0.062 | −0.126 | 0.145 |
| **centre DECLINED (42%)** | +0.068 | +0.007 | −0.042 | −0.127 | −0.202 | **0.270** |
| **centre did NOT decline (58%)** | −0.008 | −0.008 | −0.012 | −0.034 | −0.053 | **0.045** |

Per-galaxy scatter is 0.16–0.20 dex in both groups, so on the non-declining
majority the systematic trend is about a quarter of the scatter and is nearly a
uniform offset — absorbable by `a0`. The declining galaxies genuinely lose
central mass: median −0.066 dex, a **14% loss**, from redshift 2 to 0.4.

A compact channel ADDS central mass. It cannot make a centre decline, and
adding it to the 58% whose centres are already right would only overfill them.
So there was nothing for it to win, and it won nothing.

### THE LIMIT, stated as what it is

The model is `M*(<R,t_k) = sum_{t_j <= t_k} dM*_j F_j(<R)` with every
`dM*_j >= 0` and `F_j` fixed once the deposit is made. Every term is
non-negative and the sum is over a set that only grows with `t_k`, so

```
t_k > t_k'   =>   M*(<R, t_k) >= M*(<R, t_k')   for every R
```

**A deposition-only model cannot produce a declining enclosed mass at any
radius, at any parameter value.** Not "does not" — cannot. For 42% of these
galaxies the required change has the unreachable sign, which is why Stages 3.5,
3.6, 3.7 and 3.8 all failed against the same defect: they were aimed at a
target outside the reachable set.

It also explains Stage 3.7's trade. To reduce the POPULATION span the size law
had to compact every early deposit, including for the galaxies whose centres
were already right, and that is why it emptied the outskirts.

**The user's decision (2026-08-25): report this as the honest limit of the
model class, and do NOT refit on the non-declining subset.** This is the best
this kind of model can do under this assumption, and we now know why.

### What it points at

An empirical description in which already-deposited stars move outward — the
"expansion" of central stellar profiles seen in massive galaxies — would supply
the missing sign while keeping mass conserved, non-negative, and free of any
stellar input. Sketched in `doc/journal/` and deliberately NOT started here.
