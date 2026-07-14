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

## Stage 1 — single-epoch shootout (pending)

Deposit branch: population-level per-epoch fits (M(<500)-normalized, the
regime where the Gaussian rails): gauss / sersic / shell / moffat /
gausswing / empirical-kernel. Family branch: per-galaxy fits per epoch
(slope-sigmoid, Sersic CoG, evolving-Re template) + the held-epoch
quadratic-in-z closure test.
