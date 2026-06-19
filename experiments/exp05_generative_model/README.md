# exp05 — Generative profile model: paint a valid profile from halo + history

## Question

exp04 predicted the 24 curve-of-growth points directly. Here we predict the
compact, interpretable **5-parameter** radial-DiffMAH vector and *reconstruct*
the profile from it — a true generative "painting" model whose outputs are
guaranteed valid (monotonic). Three questions:

1. Is generating-via-parameters as accurate as direct point prediction?
2. Which parameters does assembly history help predict?
3. Does the model paint realistic profile *diversity* at fixed halo mass?

## Method

The radial-DiffMAH parameters (`rdm_*`) are now cached in the shared dataset
(`hongshao.tng_data.build_dataset(fit_profiles=True)`). We predict
`theta = (logMstar0, beta_in, beta_out, log R_c, Delta)` from halo features via
5-fold CV linear regression (M0 vs M0+MAH vs shuffle), reconstruct the curve of
growth with `hongshao.profiles.cog_from_physical`, and compare to exp04's direct
point prediction. n = 2538. Driver: `run.py`.

> Model fix: the radial-DiffMAH fit is now **bounded** (transition radius kept
> within the measured range, slopes within physical limits) so the parameters
> are identifiable. The previous unbounded fit let R_c / beta_in diverge for
> near-power-law profiles — fine for exp03's rank statistics, fatal for a
> regression that reconstructs from the parameters. The cached `rdm_*` values
> were refit; exp03's rank-based conclusions are unchanged.

## Key results

**1. Generating via 5 parameters costs almost nothing.**

| model | reconstructed-CoG RMS |
|---|---|
| generative, M0 only | 0.160 dex |
| generative, M0 + history | **0.128 dex** |
| generative, shuffled | 0.161 dex |
| direct point prediction (exp04) | 0.118 dex |

History improves the generative profile by **19.8%** (shuffle ~0%). Compressing
to 5 physical numbers loses only ~0.01 dex versus predicting all 24 points — so
we can paint guaranteed-valid profiles at essentially no accuracy cost.

**2. History helps the normalization most** (per-parameter scatter reduction):
`logMstar0 +13.2%`, `Delta +3.5%`, `beta_in +2.2%`, `beta_out +0.4%`,
`R_c −0.1%`. So assembly mainly sharpens the predicted inner/total mass, with a
smaller effect on the transition width — consistent with exp03/exp04.

**3. The model paints ~half the real diversity** (Panel D). In a narrow halo-mass
bin, the true galaxy-to-galaxy profile-shape scatter is ~0.20 dex at small radii;
the history-informed generator reproduces ~0.09 dex (~45% of the spread), while
the mass-only model paints almost none (~0.015 dex). So assembly history
deterministically explains roughly half the profile-shape diversity at fixed
halo mass; the rest is intrinsic scatter (secondary halo properties + the
single-projection / triaxiality noise we cannot model here).

## Decision

We have a working **generative profile-painting model**: halo mass + history →
5 parameters → a valid monotonic profile, recovering ~45% of the intrinsic
profile diversity. The clear next step (Phase 5) is the **probabilistic**
version — add the residual scatter so painted mock catalogs have the *full*
realistic diversity, not just the mean trend — and/or add **secondary halo
properties** to push past the ~45% deterministic ceiling.
