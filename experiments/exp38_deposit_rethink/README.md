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

**Promotion recommendation (user decision pending): (a) moffat and (b)
gausswing to stage 2a (the exp36 multi-epoch harness, shape swapped);
(c) the family branch to stage 2b as a joint smooth-in-z slope-sigmoid
fit anchored by the template's (Mtot, Rhalf) tracks.**
