# exp03 — An interpretable 5-number profile model (radial-DiffMAH)

## Question

exp02 showed profiles are ~2–3 dimensional, but PCA modes are abstract. Can we
describe each curve of growth with a few *physically meaningful* numbers, and
which of those numbers carry the halo-assembly signal?

## Method

The **radial-DiffMAH** model (reused from the DiffMAH idea, applied in radius):
the logarithmic slope of the curve of growth transitions smoothly from a steep
inner slope to a shallow outer slope.

    beta(R) = d ln M*(<R)/d ln R = beta_out + (beta_in - beta_out)·sigmoid((ln R_c - ln R)/Delta)
    ln M*(<R) = ln M*0 + ∫ beta d ln R

Five parameters per galaxy: normalization `M*0`, inner slope `beta_in`, outer
slope `beta_out`, transition radius `R_c`, transition width `Delta`. Monotonic by
construction. Fit to all 2545 `use` galaxies (`hongshao.profiles.fit_cog`), then:
mapped onto the exp02 PCA modes, and tested for assembly information via partial
Spearman correlation at fixed M0.

Driver: `run.py`. Figures: `exp03_fit_quality.png`, `exp03_params.png`.

## Key results

**1. Five interpretable numbers reproduce every profile.**
Median fit RMS = **0.005 dex**, 90th percentile 0.010 dex; essentially all
galaxies (>99%) fit below 0.02 dex (a handful with real profile wiggles reach
~0.045). This matches the PCA 3-mode accuracy (0.005 dex) but with physical
parameters. **A single sigmoid is enough** — no two-component model is needed for
these z=0.4 curves of growth.

**2. The physical parameters are a rotated version of the PCA modes.**
No clean one-to-one mapping (Spearman): `M*0 ↔ PC2 (0.67)`, `beta_out ↔ PC3
(−0.64)/PC2 (0.51)`, `beta_in ↔ PC3 (−0.47)`, `R_c ↔ PC3 (0.37)`, `Delta ↔ PC1
(−0.41)`. Both descriptions span the same low-dimensional manifold.

**3. The assembly signal lives mainly in the transition width `Delta`**
(partial Spearman r at fixed M0):

| param | Mpeak(z=1) | Mpeak(z=2) | z50 | z90 |
|---|---|---|---|---|
| beta_in | +0.04 | +0.01 | +0.03 | +0.04 |
| beta_out | −0.08 | +0.03 | −0.10 | −0.04 |
| R_c | −0.06 | +0.01 | −0.05 | −0.05 |
| **Delta** | **+0.21** | +0.00 | **+0.19** | +0.15 |

At fixed M0, earlier-assembled halos (more mass by z=1, higher z50) have a
**broader, more gradual** inner→outer transition. The inner/outer slopes and
transition radius individually carry little assembly memory.

## Interpretation & caveats

- radial-DiffMAH is an excellent, interpretable compression and is the natural
  profile parameterization for the conditional model.
- **The single-parameter assembly correlations (max 0.21) are slightly weaker
  than exp02's best PCA direction (PC2 vs Mpeak(z=2), 0.28).** The
  assembly-correlated direction is a particular *combination* of parameters, not
  aligned with any single physical one. Implication for the emulator: predict the
  full parameter vector jointly, not one parameter at a time.
- Magnitudes remain modest, as expected (partial MAH information + projection
  noise; see `AGENTS.md`).

## Decision

We now have two equally accurate ~5-number profile bases (PCA modes,
radial-DiffMAH params), both reaching ~0.005 dex and both carrying a modest
assembly signal. Profile compression (Phase 2) is **done**: single-component
radial-DiffMAH is the chosen interpretable parameterization.

Next — **Phase 4 / exp04:** build the first conditional model
`P(theta_prof | M0, theta_MAH)`: predict the profile parameters from halo mass +
assembly, compare to an `M0`-only baseline and a shuffle control, and quantify
the information gain (extending exp01's ~20% scatter result to the full profile).
