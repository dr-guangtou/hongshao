# exp38 — Rethink the deposition primitive

The exp36 verdict left one critical unphysical symptom: the extended
channel's Gaussian width scale sits AT its allowed maximum (log s0 = 3.0,
~1000-kpc deposits) in every fit — the parameter compensates for a wrong
model form. Suspect: the centred-Gaussian deposit has no outer wings, so it
can only supply outskirt light by inflating its scale. This experiment is a
staged ladder (plan agreed 2026-07-15): cheap diagnostics -> single-epoch
primitive shootout -> multi-epoch / halo-conditioning promotion, with user
checkpoints after stages 0+1 and before adoption.

Judged criteria (pre-registered): the held-out 148-pinned shape marks
(16.4% z=0.4 / 15.6% statistical wall / 17.7-14.7% per-epoch), NO parameter
at a bound (joint fit included), the differential-deposition curve
(deposition models), the low-mass outskirt terciles, and — for profile
families — z=2-vs-z=0.4 fit parity plus smooth-in-z parameters.

## Stage 0 — the three data diagnostics (2026-07-15, n=2397)

**0.1 The profile shape is nearly self-similar in half-mass-radius units.**
Per galaxy, the normalized log growth curve's shape drift between z=2.0 and
z=0.4 over the 1-4 R_half band is 0.025-0.028 dex rms in R/R_half
coordinates vs 0.107-0.177 dex at the same fixed physical radii — a 4-6x
collapse, largest for the most massive tercile. The population shape
scatter at fixed epoch is likewise ~3x smaller in R/R_half coordinates
(0.019-0.032 vs 0.070-0.089 dex). The median R_half grows 3.0 -> 11.6 kpc
from z=2 to z=0.4. An evolving-size, nearly-fixed-shape family (candidate
1j) is strongly supported. Figure: `stage0_similarity`.

**0.2 The measured added light is centrally peaked with Sersic n ~ 2-3
wings at a few tens of kpc — NOT a shell, and nothing like a 1000-kpc
Gaussian.** Stacked median added surface density between adjacent epochs
(3 stellar-mass terciles x 4 epoch pairs): best Sersic index 2.1-3.1 in
every cell (a Gaussian is n=0.5); kernel half-mass radius 15-47 kpc,
growing with cosmic time and mass; the peak sits at 3-4 kpc (centrally
concentrated outside the softening). The core added light turns NEGATIVE
for both z<1 pairs (the exp26 core-drop signal, now in the stacked
kernel). The Gaussian could only mimic an n~2.5 wing by inflating its
scale — the rail, explained by measurement. The off-axis/shell idea is not
supported. Figure: `stage0_autopsy`; kernels saved for the empirical
candidate (`outputs/stage0_kernels.npz`).

**0.3 Freeing the wing shape substitutes for the inflated scale.**
Refitting the exp29 single-epoch deposit model on the massive tercile of
the dev subsample (n=34): at z=0.4 the fitted Sersic index is 0.84 (median;
Gaussian = 0.5) improving the relative-RMS 0.42% -> 0.34%; at z=2.0 the
Gaussian needs log s0 = 2.73 (median, 8/34 at the loosened 3.5 bound)
while the Sersic variant takes n = 2.0 and drops the scale to log s0 =
1.87 at a better fit (0.62% -> 0.38%). The shell exponent collapses to
p = 0 (= Gaussian) for 29/34 galaxies at z=2 — shells rejected.

**Stage-0 verdict: GREEN for heavy-winged deposits (Sersic-n family) and
for the evolving-R_half self-similar profile family; RED for shell/off-axis
deposits. Proceed to the stage-1 shootout with all candidates (the shell
stays in as a falsification control).**

## Stage 1 — single-epoch shootout (2026-07-15, dev100; CHECKPOINT)

Harness note: the deposit branch uses a SIMPLIFIED population model
(pure additive deposits + lognormal efficiency + M(<500) normalization,
one shared theta per epoch, fitted independently per epoch) so that only
the deposit shape varies between candidates — comparisons are internal to
this table; the winner is re-tested inside the full exp36 structure at
stage 2. Loss = mean per-galaxy relative RMS of the model growth curve
against the measured one (smaller = better). Figure:
`stage1_deposit_tracks_dev`.

| candidate | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 | scale at a bound? |
|---|---|---|---|---|---|---|
| gauss (incumbent) | 0.2061 | 0.1959 | 0.1840 | 0.1621 | 0.1514 | UPPER rail, every epoch |
| sersic (n free) | 0.2025 | 0.1870 | 0.1724 | 0.1458 | 0.1429 | LOWER rail 3/5 epochs (n-scale degeneracy; needs an R50 reparameterization) |
| shell (p free) | 0.2069 | 0.1960 | 0.1840 | 0.1639 | 0.1525 | upper rail + p collapses to 0 — REJECTED (as stage 0 predicted) |
| **moffat (power-law tail)** | **0.2001** | **0.1841** | 0.1697 | **0.1416** | **0.1384** | **NO scale bound at any epoch**; gamma ~ 1.3 flat in z; efficiency peak mu railed at 2/5 epochs (watch) |
| **gausswing (core + n=1 wing)** | 0.1997 | 0.1838 | **0.1692** | 0.1418 | 0.1389 | off-rail 4/5 (z=1.5 rails); w falls 0.55 -> 0.24 smoothly; wing at ~5-7x the core scale |
| empirical (stage-0 kernel) | 0.1858 | 0.1736 | 0.1661 | 0.1496 | 0.1507 | stretch railed at max every epoch — the adjacent-pair kernel is too compact for the cumulative role as-built (inconclusive, not promoted) |

Family branch (per-galaxy direct fits per epoch; max|rel| over R>5 kpc of
the fit itself; parity = z=2 median / z=0.4 median):

| family | direct fit by epoch | parity | held-epoch closure (quad-in-z params) |
|---|---|---|---|
| slope-sigmoid (5p, exp03) | 0.6 / 0.6 / 0.6 / 0.6 / 0.5 % | 0.83 | 62-112% — raw params too degenerate to interpolate |
| Sersic CoG (3p) | 12.7 -> 8.8% | 0.69 | 22-45% |
| evolving-Re template (2p) | 6.4 / 6.5 / 6.7 / 6.4 / 6.1 % | 0.94 | 15-18% mid-epochs, 31/43% at the endpoints (extrapolation) |

Reads: (1) the slope-sigmoid family has the CAPACITY (0.6% at every epoch
including compact z=2 — one family fits both extremes), but its raw
per-epoch parameters cannot be interpolated (the exp27/exp30 degeneracy
lesson realized) — the stage-2b move is a JOINT all-epoch fit with
parameters forced smooth in z through the data. (2) The 2-parameter
evolving-Re template already reaches 6-7% at EVERY epoch with the best
parity (0.94) — self-similarity is real per galaxy, and its two tracks
(total mass, half-mass radius vs z) are exactly what [DiffMAH, c200c]
should condition. (3) In the deposit branch, the two heavy-tail
candidates (moffat, gausswing) beat the Gaussian at every epoch WITH the
scale off the rail — the stage-0 diagnosis confirmed at population level.

**Promotion (user-approved 2026-07-15): (a) moffat and (b) gausswing to
stage 2a; (c) the family branch to stage 2b (joint smooth-in-z fits +
halo conditioning).**

## Stage 2 — the promotion round (2026-07-15, full n=2397; CHECKPOINT)

### 2a — the primitives inside the full multi-epoch harness

Both variants keep the exp35/36 structure (lognormal efficiency, dyntrans
transport, M(<500) normalization, joint 5-epoch fit, 10-fold CV) and swap
only the deposit shape. Held-out 148-pinned shape max|rel| R>5 [%]
(incumbent multi 2ch-fa: 17.7/17.1/16.3/16.0/14.7):

| model | params | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 | avg | bounds (joint fit) |
|---|---|---|---|---|---|---|---|---|
| 2ch-exp (wide channel exponential) | 16 | 17.8 | 17.0 | 16.4 | **15.8** | **14.5** | **16.3** | log_s0_ex at 3.0 — but the stress test shows a FINITE interior optimum (below) |
| 1ch-mof (single-channel power-law tail) | **12** | 18.5 | 17.6 | 16.7 | 16.2 | **14.2** | 16.6 | **NONE** |

The four judged tests:
1. **Differential deposition** (massive tercile z0.7->0.4, data 0.37/0.11;
   incumbent 0.40/0.14): 2ch-exp 0.39/0.14 (pass); **1ch-mof 0.39/0.12 —
   the best pass in the program**, and it tracks the measured curve at
   EVERY epoch pair (z2.0->1.5: 0.18/0.05 vs data 0.23/0.06, where the
   incumbent overshot at 0.36/0.10).
2. **Outskirt terciles** (dlog Sigma model-data, 30-60 / 60-148 kpc;
   incumbent T1 +0.029/+0.070): 2ch-exp +0.023/+0.062 (comparable);
   **1ch-mof +0.026/+0.019 with T2/T3 at the +-0.01-0.02 level — flat
   residuals at every mass, no compensating undershoot.**
3. **Bounds-stress, 2ch-exp** (log_s0_ex 3.0 -> 3.5): loss gain 0.02%
   (0.1476 -> 0.1473) and the scale settles INTERIOR at 3.19 with the
   observables unmoved (f148 0.873 vs data 0.883; differential 0.38/0.13).
   Unlike the Gaussian — which re-railed at every box and bought its loss
   through horizon deletion — the exponential's preferred wide scale is
   finite and data-set; the 3.0 rail was a box clipping a nearly-flat
   optimum. Not load-bearing.
4. **Bounds-stress, 1ch-mof** (mu 3.0 -> 4.5): the mu box is irrelevant
   (the peak stays at z~3.4), BUT the nudged start exposed a SECOND basin
   at 4% better loss (0.1493 vs 0.1556) with the width time-exponent AT
   its bound (g=4.0) — and there the physics degrades (differential
   0.42/0.13, massive f148 overshoots to 0.892). The loss-optimal basin
   pays for its loss in the physics tests; the bound-free basin is the
   adopted one. (The program's core pattern, reproduced within one model.)

### 2b — the smooth-in-z profile family + halo conditioning

Joint per-galaxy fits (each family parameter quadratic in z, all 5 epochs
at once, R>=5 kpc): the slope-sigmoid keeps 4.0-6.1% at every epoch
(capacity survives the smoothness constraint; unconstrained per-epoch
ceiling 0.6%); the 2-parameter template 7.6-11.6%. Held-epoch closure:
mid-epochs 18-26% (sigmoid) / 15-20% (template); endpoints 46-58% / 32-46%
(quadratic extrapolation, expected).

Conditioning the per-galaxy coefficients on [DiffMAH(4), c200c] (poly2
heteroscedastic cores, 5-fold held-out), scored as 148-pinned shape R>5:

| family | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| sigmoid-z (15 coeffs) | 82.8 | 50.7 | 31.0 | 43.8 | 46.7 — FAIL (coefficients not feature-reachable) |
| **template-z (6 coeffs)** | **16.3** | **16.0** | **15.3** | 17.0 | 18.8 |

**The 6-coefficient evolving-Re template — one frozen population shape,
two smooth tracks (total mass, half-mass radius vs z) predicted from the
halo — matches or beats the 16-parameter kernel at z<=1 and approaches
the 15.6% statistical wall at z=0.4.** Its weakness is z>=1.5 (17.0/18.8
vs the kernel's 16.0-14.2), where the frozen shape drifts (stage 0.1:
Re-coordinate shape scatter grows 0.019 -> 0.032 dex toward z=2).

### Stage-2 verdict (pre-adoption; user checkpoint)

- **1ch-mof is the recommended kernel successor**: 12 parameters (vs 16),
  no split channel, NO parameter at a bound, the best
  differential-deposition pass in the program, flat outskirt residuals at
  every mass, and held-out accuracy within 0.5 points of the incumbent
  (better at z=2). The Gaussian-era pathologies — the railed width scale,
  the two-channel patch, the low-mass outskirt overshoot — all trace to
  the missing power-law tail.
- **2ch-exp** is the accuracy pick among deposition models (avg 16.3,
  finite wide scale) if the two-channel architecture is retained.
- **template-z** earns a place as the minimal statistical description:
  massive-galaxy profile evolution z=2 -> 0.4 is, to 15-19%, "one shape,
  two halo-predictable growth tracks".

## Stage 3 — the inner-mass deficit (user-raised 2026-07-15; full n=2397)

Both stage-2 winners under-predict the observation-facing inner masses
(1ch-mof in-sample: M(<5 kpc) -11.7% / M(<10) -5.4% at z=0.4; -8.0/-5.8%
at z=2) while over-filling 10-30 kpc — a placement error. Diagnosis and
fixes, all measured:

1. **Provenance** (model M(<10) decomposed by deposit epoch, fitted vs
   FROZEN migration clock): at z=0.4 the frozen clock would deliver
   1.47x the observed inner mass — compact material exists; the alpha=1
   clock drains it (a retention knob is missing). At z=2 even frozen
   reaches only 0.97x — deposits are BORN too wide there (the narrow
   lognormal efficiency caps the z>4 ultra-compact budget).
2. **Re-aimed objective alone** (fit R>5 shape + M(<5)/M(<10) integral
   terms, the user's option 2): halves the deficit (M<5 -4.8/-3.6%) and
   improves the outer shape — but rails the efficiency peak (mu = 3.0):
   it buys the core by over-weighting the earliest deposits. Partly
   allocational, but not a clean fix. (2ch-exp re-aims worse: -6.8%,
   wide-channel scale re-rails.)
3. **Retention floor alone** (fc' = f_ret + (1-f_ret) fc, plain loss):
   f_ret = 0.084, overall loss 0.1556 -> 0.1494 (-4%, the physical
   counterpart of the pathological g=4 basin) — a real model upgrade,
   but the inner bias barely moves (the plain loss spends the retained
   mass elsewhere).
4. **The dissipative-core channel** (the user's option 1, physically
   read as the compaction-formed in-situ core: a fraction f_core of
   every deposit lands in a NON-migrating component at scale rc_core;
   fit under the inner-aware objective): **f_core = 0.185, rc_core =
   1.55 kpc, NO parameter at a bound, deficit halved (M<5 -5.3/-4.2%,
   M<10 -4.3/-3.2%) and the in-sample pinned shape improves
   (19.2 -> 16.1% at z=0.4, 14.1 -> 12.1% at z=2).**

Next (the untested factorial cell): core + retention combined, fit under
the inner-aware objective, then the held-out CV + physics tests for the
adoption candidate.
