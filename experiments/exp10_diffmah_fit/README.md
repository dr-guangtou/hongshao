# exp10 — Our own DiffMAH fits (portable halo features)

The emulator's MAH features so far — MAH-PCA(4) — are defined on the *TNG sample*
(the PCA basis is fit to our galaxies), so they don't transfer to another
simulation or to observations. DiffMAH (Hearin et al. 2021, arXiv:2105.05859)
describes each halo's mass accretion history with four **intrinsic** numbers, so
an emulator trained on them is portable. We fit DiffMAH ourselves (the released
fits are not cross-matched to our galaxies, but we have the MAH curves),
visualize the fits the way we did the curves of growth, and check the features.

## Question

Can DiffMAH parameters replace MAH-PCA(4) as the halo features — i.e. do four
portable, intrinsic numbers carry the same assembly signal for predicting the
stellar profile?

## Method

- **Model** (`hongshao/diffmah.py`): `log M(t) = logmp + α(logt)·(logt − logt0)`
  with a rolling index `α = early + (late − early)·sigmoid(k·(logt − logtc))`,
  transition speed **k fixed at 3.5** (DiffMAH default). Four per-halo params:
  `logmp` (normalization), `logtc` (transition time), `early`, `late`.
- **Anchor** at the z=0.4 epoch (`logt0 = log10 t(z=0.4)`), so `logmp` ≈ the
  final peak mass M0.
- **Fit range** `t ≥ 2 Gyr` (z ≲ 3.2): the earlier history is sparsely sampled
  and poorly resolved (tiny halo mass), and a smooth rolling power law cannot
  follow it — the analog of the 5-kpc inner cut on the curve of growth. This also
  matches the exp06 MAH-PCA grid (2.2–9.0 Gyr).
- **Feature check**: 5-fold CV linear prediction of the 4 aperture/annulus masses
  from M0-only vs M0+MAH-PCA(4) vs the 4 DiffMAH params, scored by CRPS.

Driver: `run.py` (`EXP10_NMAX=400` for a sub-minute pass). Figures:
`exp10_fit_quality` (best/median/worst example fits + RMS histogram),
`exp10_fits_by_mass` (individual MAHs + fits and residuals in 3 mass bins —
the MAH analog of the CoG-by-mass figure), `exp10_feature_check`. Sample:
n = 2545; per-halo params saved to `outputs/diffmah_params.csv`.

## Key results

**1. DiffMAH fits the resolved history to ~0.06 dex.** Median fit RMS 0.063 dex
(90th 0.099). This is looser than the curve-of-growth fits (~0.001 dex) because
the MAH is intrinsically wigglier — mergers and bursts that a smooth four-number
model cannot capture (consistent with exp06: the MAH is only ~93% compressible in
3 modes, vs 99.7% for the CoG). The worst fits are halos with a sharp accretion
jump (a major merger); the median fit tracks the data cleanly.

**2. The portable DiffMAH params carry ~88% of the MAH-PCA assembly signal.**
Aperture-mass prediction CRPS (n=2533):

| features | mean CRPS [dex] | improvement over M0 |
|---|---|---|
| M0 only | 0.1117 | — |
| M0 + MAH-PCA(4) | 0.0849 | 0.0268 |
| **DiffMAH (4 params)** | **0.0882** | **0.0235 (88% of MAH-PCA)** |

DiffMAH is marginally behind MAH-PCA (4% higher CRPS) — the cost of the smooth
approximation (it discards the ~0.06 dex of MAH wiggle, some of which correlates
with the profile) plus using four numbers instead of M0 + four PCs. But it is
**portable**: the same emulator can run on any halo with a DiffMAH fit (N-body,
other sims, or — in principle — observational MAH proxies), which MAH-PCA cannot.

## Interpretation & caveats

- The `t ≥ 2 Gyr` cut is the MAH analog of the 5-kpc CoG cut: the early history is
  unresolved, and it was the dominant source of fit residual (median RMS 0.094 →
  0.063 when applied). It keeps the epochs that matter (z=2 is t≈3.3 Gyr).
- `logmp` ≈ M0 by construction; the *assembly* information is in
  `early`/`late`/`logtc`. The 4% gap vs MAH-PCA is the price of portability +
  smoothing, not a failure — DiffMAH recovers the bulk of the signal.
- We fit DiffMAH ourselves (no `halo_id` match to the released fits); the
  parameterization and k=3.5 follow the published model.

## Decision

Adopt the **four DiffMAH parameters as the portable halo features** for the
emulator going forward. They nearly match MAH-PCA(4) on the assembly signal and,
unlike the sample-defined PCA basis, make the model applicable beyond TNG — the
generalizability the Ultimate-SHMR goal requires. Next: build the emulator on
these features and refine the scatter model (exp08 covariance) under the exp07
suite.
