# exp01 — Which radial zones remember which epochs of halo growth?

## Question

At fixed **final halo mass** `M0 = Mpeak(z=0.4)`, does the halo's assembly
history carry information about the central galaxy's stellar mass profile? And
specifically: do the **inner** regions track **early** halo growth while the
**outskirts** track **late** growth (the two-phase assembly picture)?

## Method

- Sample: 2545 galaxies passing the clean `use` cut (`data/processed/tng300_072_z0p4.fits`).
- Galaxy side: stellar mass in 7 radial zones (`<10, 10–30, …, 120–150 kpc`,
  differential shells) and the full 24-point curve of growth, at z=0.4.
- Halo side: `Mpeak` at z=0.7/1/1.5/2 and formation redshifts z50/z75/z90.
- **Control for M0** via partial Spearman (rank) correlation — isolates
  assembly information beyond final halo mass.
- **Scatter test**: predict inner `M*(<10 kpc)` and outer `M*(50–100 kpc)` from
  `M0` alone vs. `M0 + halo history`, scored by 5-fold cross-validated RMSE.
- **Shuffle control**: permute the halo-history features among halos in the same
  M0 bin; a real signal should vanish.

Driver: `run.py` (run `PYTHONPATH=. uv run python experiments/exp01_aperture_mah_corr/run.py`).

## Key results

**1. Halo history reduces stellar-mass scatter by ~20% at fixed M0 — and it's real.**

| target | RMSE (M0 only) | RMSE (M0 + history) | shuffled | reduction |
|---|---|---|---|---|
| `M*(<10 kpc)` inner | 0.143 dex | 0.116 dex | 0.143 dex | **19.4%** |
| `M*(50–100 kpc)` outer | 0.230 dex | 0.183 dex | 0.231 dex | **20.6%** |

The shuffle control gives ~0% reduction, so the gain is genuine assembly
information, not hierarchical averaging.

**2. The inner galaxy uniquely remembers early halo growth.**

All correlations are positive: at fixed M0, earlier-forming halos host more
massive centrals at every radius. The *radial structure* is the interesting part
(partial Spearman r, controlling for M0):

- Correlation with **early** mass `Mpeak(z=2)` is strongest in the center
  (`<10 kpc`: r = 0.46) and falls by ~2× outward (`120–150 kpc`: r = 0.20).
- Correlation with **recent/intermediate** mass `Mpeak(z≈0.7–1)` is high at all
  radii (r ≈ 0.5–0.6), peaking at intermediate radii (~30–75 kpc).
- So the inner zone is the only place where early- and late-time halo mass
  matter comparably; everywhere outside, recent growth dominates.

This matches the two-phase expectation — early collapse builds the dense inner
body, later growth builds the envelope — though the *dominant* axis at all radii
is overall formation time (z≈0.7–1), with the inner early-time sensitivity as a
clear second-order, physically meaningful signal.

Figures: `figures/exp01_results.png` (headline — Panel A: correlation vs radius;
Panel B: scatter reduction), `exp01_corr_matrix.png` (full zone × feature map).

## Why ~20% and not more (expected)

The residual scatter is not a failure — it is expected:

- The main-branch MAH is only part of the halo information. Secondary properties
  (initial tidal/tensor field, concentration) carry independent signal we did
  not use here.
- The profiles are single random 2-D projections of triaxial galaxies, so
  viewing angle and intrinsic shape inject noise that *no* halo model can remove
  from this dataset.

So ~20% is a meaningful lower bound on the assembly signal accessible from the
main-branch growth history alone. The aim is a phenomenological model that
captures this part, not zero scatter.

## Caveats

- `M0` is the peak mass at the latest available snapshot (~z=0.42), not the
  exact snap-72 mass (see `doc/tng300_data.md`).
- `Mpeak(z)` at different epochs are strongly correlated at fixed M0, so the
  per-epoch correlations are not independent — read the *radial trend* within a
  column, not absolute column-to-column differences.
- Outer shells beyond ~50 kpc still share correlated large-scale structure.

## Decision

**Positive result → proceed.** The ~20% scatter reduction and the
radius-dependent early/late signal justify building a compact profile
representation and a conditional model `P(theta_prof | M0, theta_MAH)`. Next:
**exp02** — compress the curves of growth (PCA + radial-DiffMAH) and test which
profile-shape modes carry the assembly signal seen here.
