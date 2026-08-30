# The 1-D profile data: what is in it, what it means, and how to use it

*Written 2026-08-30, design settled with the user 2026-08-31, on branch
`exp68-profile-data`. Code: `hongshao/profile_data.py`. Figure:
`figures/profile_data_qa.png` (`scripts/profile_data_qa.py`). Products:
`data/processed/tng300_072_profiles_v2.npz` (the current product) and
`..._v1.npz` (kept readable as a reference point). Both gitignored; v1 rebuilds
from the raw drop in about ten minutes and v2 derives from v1 in one second.*

**Sections 0-6 describe what was measured and are version-independent.
Section 7 is the v1 data model. Section 10 is the v2 data model, the settled
design, and the changelog — read that one if you are writing new code.**

This note is the reference for the second half of the raw data drop. Everything
this project has fitted so far came from one of its two directories; this note
opens the other one, works out exactly how the two are related, and turns the
pair into a single consolidated file with a measured error model.

Read this before using `sigma_iso_dex`, `sigma_shape_dex` or `sigma_proj_dex`
in any objective.

---

## 0. Vocabulary (nothing here is assumed known)

| term | what it means |
|---|---|
| **isophote** | An ellipse fitted to the 2-D stellar mass map at one semi-major axis `a`. photutils' `Ellipse` fitter returns, for each one, the mean surface mass density along the ellipse and the shape of the ellipse. |
| **surface mass density** `Sigma(a)` | Stellar mass per unit area along that isophote, in solar masses per square kiloparsec. A *local* quantity. Stored as the column `intensity`. |
| **`intens_err`** | The **standard error of the azimuthal mean** of `Sigma` along the ellipse — how much the density varies as you go around the isophote, divided by the square root of the number of sampled points. It is *not* a Poisson particle error, and this project never invents one. |
| **ellipticity** `e` | `1 - b/a`. A circle has `e = 0`. |
| **curve of growth (CoG)** | Stellar mass enclosed by an aperture of semi-major axis `R`, `M*(<R)`, on the 24-radius grid. The *cumulative integral* of `Sigma`. |
| **constant isophotal shape** | One ellipticity and position angle for a galaxy's whole curve of growth, as opposed to the radius-by-radius shape the isophote fit measures. |
| **`flag`** | The drop's own per-epoch validity flag. |
| **`test`** | The drop's own per-epoch flag meaning *the shape measurement did not have to fall back to its default starting radius*. `test = False` is a **failed** fit, and this note shows it is the single most useful quality cut in the dataset. |

---

## 1. What the raw drop actually contains

Two per-galaxy directories, 3388 files each, indexed identically.

**`save_tng300_072_hist_aper_dir/*.npy`** — what every experiment so far has
used. Per epoch: `cog` (24 enclosed masses) and `aper` (7 enclosed masses at
fixed semi-major axes 10, 30, 50, 75, 100, 120, 150 kpc).

**`save_tng300_072_hist_prof/*.npy`** — the directory nobody had opened. Per
epoch, four entries:

- `prof` — an astropy table with **seven** columns, in physical units:
  `r_kpc`, `intensity`, `intens_err`, `ellipticity`, `ellip_err`, `pa`,
  `pa_err`. **So yes: the ellipticity and position angle PROFILES are there,
  radius by radius, each with its own error** — not just an average.
- `other` — the same isophote list in raw map/pixel units, plus photutils'
  own diagnostics: `ndata`, `nflag`, `niter`, `stop_code`, `grad`,
  `grad_error`, `grad_rerror`, `x0`, `y0` and their errors.
- `flag` and `test` — the two booleans above.

The radial grid is *not* the CoG grid and *not* the same for every galaxy. Three
families were found: a 33-point geometric ladder with ratio 1.2 from 0.5608 to
159.74 kpc (the great majority); the same ladder scaled up by 4.17 per cent
(0.5841 to 166.40 kpc); and a 35-point degenerate ladder starting at `1e-4` kpc
with ratio 16, which is what the fitter writes when it gives up — always
together with `test = False`.

A third file, `galaxies_tng300_072_hmc_13_aperture_mass.txt`, is a table of
3388 galaxies × **3 sky projections** (`xy`, `xz`, `yz`) carrying aperture
masses at six radii, halo properties, and **one ellipticity and position angle
per projection**.

---

## 2. The stored curve of growth is the `xy` projection

The seven aperture masses stored beside each z = 0.4 curve of growth are
**bit-identical to the `xy` rows** of the projection table, and to no other
projection (`xz` and `yz` are 7–8 per cent away in the median, never exact).

This matters twice over. It tells us the whole five-epoch history is one fixed
sky projection of the box — so a model with no orientation is being asked to
reproduce one particular viewing angle. And it means the `xz`/`yz` rows are a
**directly measured projection-to-projection scatter** of the same galaxies,
which is the calibration for the projection term in section 6.

About 2 per cent of galaxies do not match `xy` exactly. They are flagged
per galaxy as `cog_matches_xy = False` with the deviation in `cog_xy_max_dev`,
and they are almost entirely the same galaxies as `test = False`.

---

## 3. The stored curve of growth uses a CONSTANT isophotal shape

This was the user's expectation and it is now established. Five candidate
geometries were tested against the stored curve, on the same profiles:

| geometry | median `|log10(rebuilt / stored)|` at z = 0.4 |
|---|---|
| **constant shape, `R` = semi-major axis** | **0.0027 dex** |
| radius-by-radius ellipticity profile | 0.0383 dex |
| circular aperture | 0.182 dex |
| `R` = circularized radius | intermediate, radius-dependent |
| `R` = semi-minor axis | intermediate, radius-dependent |

The winner, by a factor of fourteen, is

```
M*(<R) = ∫₀^R 2 π a (1 − e_const) Σ(a) da
```

with `R` the **semi-major** axis and `e_const` a **single ellipticity for the
whole curve**. The radius-by-radius ellipticity profile is clearly *not* what
built the stored CoG. A circular aperture is wrong by the constant factor
`(1 − e_const)` — 10 to 45 per cent depending on the galaxy — which is why the
first naive reproduction attempt looked like a near-constant 0.80–0.85 ratio.

Consistent with the user's reading, the stored CoG was almost certainly measured
by **laying elliptical apertures directly on the stellar mass map**, not by
integrating the photutils profile. Section 5 gives the ~1 per cent residual that
proves those are not the same operation.

---

## 4. The constant shape is recorded at z = 0.4 and RECOVERABLE at every epoch

The projection table records the constant shape, but only at z = 0.4 (it is a
snapshot-72 table). For z = 0.7 to 2 nothing is recorded.

It does not need to be, because `(1 − e_const)` is *by construction* the ratio
of the stored CoG to the **circular** integral of the same profile, at every
radius. Taking the median of that ratio over the 24 radii recovers the shape;
its radial scatter is both the uncertainty and a quality check, because a curve
built with any other geometry would not give a radius-independent ratio.

Checked at z = 0.4 against the recorded value, on galaxies with `test = True`:

- correlation **0.9947**, rms **0.018** in `e`, bias **+0.009** (n = 3213 of 3380);
- the radial scatter of the ratio is **1.1 per cent** at z = 0.4 rising to
  **2.0 per cent** at z = 2 — i.e. the constant-shape model describes the stored
  curve to one or two per cent at every epoch.

`test = False` galaxies are the *entire* outlier population: including them
drops the correlation from **0.9947 to 0.9047** and inflates the rms from
**0.018 to 0.071**. Panel B of the QA figure shows this as two clearly separated populations.

**Use `iso_shape_ok` (the `test` flag) as the quality cut for anything that
touches shape.**

---

## 5. The one per cent that does not reconstruct

Rebuilt with the **recovered** shape, the reproduction is small by construction.
The independent check is z = 0.4 against the **recorded** shape, where nothing
was fitted:

> the map-measured aperture holds **about 1.4 per cent more mass** than the
> smooth isophote model integrates to.

This is physically expected: the isophote fit is a smooth elliptical model of a
clumpy, asymmetric map, and it loses a little mass to the substructure it does
not describe. It is stored as `cog_bias_vs_recorded_shape` and **never corrected
away** — it is the honest size of the gap between "aperture photometry on the
map" and "integral of the 1-D profile", and any likelihood built on the 1-D
profile inherits it as a floor.

---

## 6. The error model: three measured terms

This is the part that matters for open question C17. None of these three terms
is invented; each is measured from something already in the drop.

### (a) `sigma_iso_dex` — the azimuthal / statistical term

`intens_err` propagated through the same integral that builds the CoG. Because
a curve of growth is a cumulative sum, the per-radius error is much smaller than
the per-isophote error: the local density is uncertain by 5–18 per cent, but the
*enclosed mass* is uncertain by only 0.006–0.025 dex, because cumulating
averages the azimuthal scatter down.

**Its radius-to-radius covariance is not a modelling choice.** A CoG is a
cumulative sum of independent annulus increments, so for `i ≤ j`

```
Cov(M_i, M_j) = Var(M_i)
```

exactly — the variance vector determines the entire 24×24 matrix and nothing
extra needs storing. `cog_covariance()` builds it and `--selftest` verifies it
against a Monte Carlo. This is the single most important structural fact for a
CoG likelihood: **an error made inside a small radius propagates to every larger
radius with the same sign**, so the dominant mode is the whole curve shifted, not
24 free points. A likelihood that treats the 24 radii as independent will shrink
its uncertainties by roughly a factor of five.

**What this costs in practice, measured on a real galaxy.** For galaxy 0 at
z = 0.4 the 24×24 covariance has 90.2 per cent of its total variance in a
single eigenvector — the whole-curve-shifted mode — and an effective rank
(participation ratio) of **1.22**. The 24 radii of a curve of growth carry
roughly the information of **one and a bit independent measurements**, not 24.
The correlation between `M*(<2 kpc)` and `M*(<148 kpc)` is 0.41 for that galaxy
even though the outer aperture contains a hundred times more mass.

Any likelihood that sums 24 independent Gaussian terms per galaxy is therefore
overstating its evidence by roughly a factor of twenty in the exponent. This is
the single most consequential number in this note for open question C17.

Independence *between isophotes* is an assumption, and it is the optimistic end:
photutils samples neighbouring ellipses from overlapping pixels, so the true
covariance is somewhat positive and this term is, if anything, understated.

### (b) `sigma_shape_dex` — the constant-shape term

The cost of describing a galaxy whose shape genuinely varies with radius by a
single ellipticity. It multiplies `(1 − e_const)`, so it is **one number per
galaxy and epoch that applies coherently to every radius**: it moves a curve up
or down bodily and never tilts it. About 0.005 dex at z = 0.4 rising to 0.009 dex at z = 2.

### (c) `sigma_proj_dex` — the projection / orientation term

The scatter of the same galaxy's aperture mass across the three sky projections,
at six radii, at z = 0.4. **This is a real measurement, not a mock**, and it is
the term this project has never had a number for.

### What the three terms say

Median over galaxies, in dex, all at z = 0.4 so the comparison is like for like:

| radius | 10 kpc | 30 kpc | 50 kpc | 75 kpc | 100 kpc | 150 kpc |
|---|---|---|---|---|---|---|
| statistical `sigma_iso` | 0.0098 | 0.0073 | — | — | 0.0061 | — |
| projection `sigma_proj` | 0.0241 | 0.0162 | 0.0123 | 0.0094 | 0.0076 | 0.0052 |
| **ratio** | **2.45** | **2.23** | **1.87** | **1.51** | **1.24** | — |

> **From 10 to 100 kpc the projection term is the largest of the three — two and
> a half times the statistical error at 10 kpc — and it is the one term the
> model cannot reduce, because the model has no orientation.**

A deposition model predicts a spherically-symmetric-in-effect mass profile and
is scored against *one* viewing angle of a triaxial galaxy. The difference
between viewing angles is irreducible error for that comparison. Any likelihood
that omits it will attribute a real orientation accident to a failure of the
physics — which is exactly the concern recorded in the memory
`decliner-physics-user-interpretation`.

**Two limits, stated plainly, because they matter for how far this can be
pushed.** The projection term is *not measured inside 10 kpc* — the projection
table's innermost aperture is 10 kpc — and inside 10 kpc is exactly where the
statistical term is largest (0.017 dex at 2 kpc) and where the fits have been
failing. And it is measured **at z = 0.4 only**, from **three** projections, so
it is a three-sample standard deviation: well determined in the population
median, noisy per galaxy, and not extrapolable in redshift. Galaxies are more
irregular at z = 2, so if anything it grows.

Note also that the two epochs' behaviour differs: `sigma_iso` grows with
redshift at every radius (0.0061 to 0.0150 dex at 100 kpc), so by z = 2 the
statistical term alone is twice the z = 0.4 projection term there. Whether the
projection term also grows is unknown and unmeasurable from this drop.

---

## 7. The data model

One compressed `.npz`, `data/processed/tng300_072_profiles_v1.npz`, read with
`hongshao.profile_data.load_profiles()`. Galaxy index runs 0–3387 and matches
every other product in this repo; epoch index runs 0–4 for
z = 0.4, 0.7, 1.0, 1.5, 2.0.

### Grids and provenance
| key | shape | meaning |
|---|---|---|
| `epochs`, `redshifts` | (5,) | epoch names and redshifts |
| `cog_radius_kpc` | (24,) | the CoG grid, semi-major axis in kpc |
| `aper_sma_kpc`, `aper6_sma_kpc` | (7,), (6,) | the fixed-aperture grids |
| `projections`, `cog_projection` | (3,), scalar | `('xy','xz','yz')`; the stored CoG is `xy` |
| `built_utc`, `source_dir`, `git_sha` | scalars | provenance |

### Per-galaxy bookkeeping and quality
| key | shape | meaning |
|---|---|---|
| `file_present` | (3388,) | both raw `.npy` files readable |
| `cog_matches_xy`, `cog_xy_max_dev` | (3388,) | z=0.4 apertures equal the `xy` table row |
| `cog_bias_vs_recorded_shape` | (3388,) | the ~1.3 per cent of section 5 |
| `n_iso` | (3388, 5) | number of isophotes stored |
| `flag_valid` | (3388, 5) | the drop's `flag` |
| `iso_shape_ok` | (3388, 5) | the drop's `test` — **the quality cut** |
| `has_shape_cols` | (3388, 5) | the profile table carries ellipticity/PA |
| `cog_over_circ_spread` | (3388, 5) | radial scatter of stored/circular; the reconstruction quality per galaxy-epoch |

### The isophote profile, padded with NaN to 40 radii
`iso_r_kpc`, `iso_sigma`, `iso_sigma_err`, `iso_ellipticity`, `iso_ellip_err`,
`iso_pa`, `iso_pa_err` — all (3388, 5, 40), physical units
(kpc, Msun/kpc², radians for PA).
Diagnostics, same shape: `iso_ndata`, `iso_nflag`, `iso_niter`,
`iso_stop_code`, `iso_grad`, `iso_grad_err`, `iso_grad_rerr`, `iso_x0`,
`iso_y0`. Plus `pixel_mass_per_area` (3388, 5), the pixel-to-physical
rescaling, kept for provenance only.

### The curves of growth, all on `cog_radius_kpc`
| key | shape | meaning |
|---|---|---|
| `cog_provided` | (3388, 5, 24) | **the stored curve — still the reference** |
| `cog_const_shape` | (3388, 5, 24) | rebuilt with the recovered constant shape |
| `cog_var_shape` | (3388, 5, 24) | rebuilt with the radius-by-radius ellipticity profile |
| `aper_provided` | (3388, 5, 7) | the stored fixed apertures |

### The constant isophotal shape
| key | shape | meaning |
|---|---|---|
| `ellip_const`, `ellip_const_err` | (3388, 5) | recovered, every epoch |
| `ellip_const_table`, `pa_const_table` | (3388, 3) | recorded, z = 0.4, per projection |
| `ellip_const_n_used` | (3388, 5) | radii used in the recovery |

### The error model
| key | shape | meaning |
|---|---|---|
| `sigma_iso_abs`, `sigma_iso_dex` | (3388, 5, 24) | azimuthal/statistical; covariance via `cog_covariance` |
| `sigma_shape_dex` | (3388, 5) | constant-shape, coherent in radius |
| `sigma_proj_dex`, `sigma_proj_dex_n` | (3388, 6) | projection scatter, z = 0.4, on `aper6_sma_kpc` |

### The surface mass density profile
| key | shape | meaning |
|---|---|---|
| `sigma_on_cog_grid`, `sigma_err_on_cog_grid` | (3388, 5, 24) | `Sigma` and its error interpolated onto the CoG grid, for convenience |

The raw `iso_*` arrays are the authority; the interpolated versions are a
convenience and lose information at the inner radii, where the isophote grid is
finer than the CoG grid.

### Also carried through from the projection table
`aper_mass_proj` (3388, 3, 6), `mass_halo`, `catgrp_id`, and the halo
properties `c_200c`, `c_to_a_3d`, `b_to_a_3d`, `v_sigma_3d`, `acc_rate`.

---

## 8. How to use this, and what NOT to do

**Do** keep `cog_provided` as the fitting target. It is what every previous
result was measured against, and switching targets silently would break every
comparison in the programme. The reconstructions exist to explain the stored
curve and to carry its error, not to replace it.

**Do** apply `iso_shape_ok` (and `flag_valid`) before using anything
shape-derived; section 4 shows the fallback galaxies are the entire outlier
population. This is a *profile-quality* cut and is separate from — and does not
replace — THE FITTING SAMPLE rule in `CLAUDE.md`, which is about halo and
stellar histories. Anything new must be composed with that rule, never
substituted for it.

**Do** use `cog_covariance()` rather than a diagonal, or state explicitly that
you are choosing the diagonal and why.

**Do not** treat `intens_err` as a Poisson error. `ndata` is 13 at every inner
isophote because that is photutils' minimum sampling of an ellipse, not a
particle count. There is no particle count anywhere in this drop.

**Do not** assume the projection term is calibrated above z = 0.4. It is not.

**Do not** use `cog_var_shape` as a better curve of growth. It is not better; it
is a *different* measurement (the mass inside the actual isophotes rather than
inside a fixed ellipse), useful for asking how much the shape gradient matters,
and it does not reproduce the stored curve.

---

## 9. Loose end: the stellar mass maps are unreadable

`map_tng100_hist_stellar.hdf5` (852 MB) is **not an HDF5 file**: it has no HDF5
signature, its first 201 MB are zeros, and the region beyond that is plausible
float64 stellar masses (values of order 10⁶ solar masses per pixel). It appears
damaged in transfer, or to be a raw dump with a misleading extension.

This is worth repairing, because the maps would settle by direct measurement
what section 3 established by inference, would give the exact aperture geometry
rather than a recovered one, and would allow the projection term of section 6(c)
to be measured at every epoch instead of only at z = 0.4. **A re-supplied map
file is the single highest-value addition to this dataset.**

---

## 10. Version 2 — the settled data model (2026-08-31)

Section 7 describes v1. This section describes **v2**, the product to write new
code against. v2 is a strict **superset** of v1: every v1 key resolves in it.

```python
from hongshao.profile_data import load_profiles
d = load_profiles()             # v2
d = load_profiles(version=1)    # the v1 reference build
```

### 10.1 The decisions, and who made them

Settled with the user on 2026-08-31 after the measurements in sections 2–6.

| decision | choice | why |
|---|---|---|
| fitting target | **`cog_provided`, unchanged** | it is the measurement every prior result was scored against; nothing here replaces it |
| fiducial density | **the isophote `intensity`** — the measured 1-D stellar mass density | it is the actual measurement, always positive, has its own error bar, and reaches inside 2 kpc |
| secondary density | the CoG derivative, stored as `annulus_mass` | a *different* measurement, not a rival estimate — see 10.2 |
| radial grid | **one shared grid for every galaxy and epoch** | 76% of galaxy-epochs already sit exactly on it, so most are stored unmodified |
| beyond a galaxy's cutoff | density **exactly zero**, `sigma_measured = False` | makes the reconstructed CoG stay flat by construction, not by a special case |
| radius coordinate | semi-major primary, `r_circ` alongside | keeps every prior result valid while making the 6.1% epoch effect visible |
| average ellipticity | CoG-ratio route adopted; profile route as an independent cross-check | their difference is a certificate that works at every epoch |
| inner isophotes | kept in full, flagged `sigma_resolved`, **errors untouched** | the product records what was measured; weighting an unresolved point is the likelihood's decision, not the data model's |
| projection term | raw only — z=0.4, 6 radii, **no extrapolation** | any use inside 10 kpc or above z=0.4 must be an explicit assumption in the experiment that makes it |
| negative annuli | kept, flagged, **not repaired** | the transform must stay exactly invertible; a non-monotone CoG is a real property of the measurement |

### 10.2 The two densities are different measurements

This is the single most important thing to understand before using the density.

- **`sigma_shared`** — the isophote density. Measured *along* the best-fit
  ellipse, and **sigma-clipped**, so it excludes satellites and intracluster
  light. It is the closer thing to *the galaxy*.
- **`annulus_mass`** — the CoG derivative. The mass in a fixed elliptical
  annulus, counting **everything** in the aperture.

They differ by 0.05 dex at 110 kpc at z = 0.4 and **0.64 dex at z = 2**, and
that gap *is* the clipped material. Neither is wrong. They answer different
questions, and a result fitted to one is not comparable to a result fitted to
the other.

In *cumulative* terms the difference is far smaller, because the outer annuli
hold little mass: rebuilding the CoG from the isophote density lands within
**0.010 dex** of the stored CoG at 148 kpc.

### 10.3 What the flat-outskirt prescription costs

Median `log10(CoG rebuilt from the density / stored CoG)`, dex:

| r [kpc] | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| 2.0 | −0.011 | −0.011 | −0.009 | −0.008 | −0.006 |
| 10.2 | +0.003 | +0.004 | +0.005 | +0.006 | +0.008 |
| 52.3 | +0.001 | +0.001 | +0.001 | −0.000 | −0.002 |
| 148.2 | −0.004 | −0.005 | −0.007 | −0.009 | −0.010 |

Split at 148 kpc by whether the density truncates inside the CoG grid:

| | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| truncated | −0.010 | −0.010 | −0.010 | −0.010 | −0.010 |
| full | −0.003 | −0.005 | −0.006 | −0.008 | −0.009 |
| n truncated (of 3388) | 100 | 360 | 734 | 1550 | 2404 |

**The two rows agreeing is the result.** At z = 2, where 71% of galaxies
truncate, truncated and untruncated galaxies land at −0.010 and −0.009 dex. The
residual is the smooth-model reconstruction bias of section 5, *not* the flat
prescription. Zero-filling costs essentially nothing.

### 10.4 The two routes to the constant shape, and the certificate

| | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| CoG-ratio route (`ellip_const`, adopted) | 3380 | 3379 | 3379 | 3378 | 3374 |
| profile route (`ellip_const_prof`, 5–30 kpc unweighted mean) | 3312 | 3201 | 3096 | 2878 | 2430 |
| median \|difference\| | 0.020 | 0.023 | 0.026 | 0.030 | 0.035 |
| fraction agreeing within 0.05 | 0.851 | 0.767 | 0.709 | 0.612 | **0.461** |

The two routes are independent: one uses only the stored CoG and the intensity
profile, the other only the measured ellipticity profile. **Their agreement is a
certificate that needs no recorded reference**, so unlike the z = 0.4 comparison
of section 4 it works at every epoch. The expected spread if both were unbiased
is about 0.031, so z = 0.4's 85% is what agreement looks like; z = 2's 46% says
the shape is genuinely less well determined there.

### 10.5 New keys in v2

| key | shape | meaning |
|---|---|---|
| `schema_version` | scalar | 2 |
| `shared_grid_kpc` | (32,) | 0.5608 → 159.74 kpc, ratio 1.2 |
| `sigma_shared` | (3388, 5, 32) | **the fiducial density**, Msun/kpc²; exactly 0 outside the measured range |
| `sigma_err_shared` | (3388, 5, 32) | its error; 0 outside |
| `sigma_measured` | (3388, 5, 32) | **gate on this.** A zero density means *not measured*, never *empty* |
| `sigma_resolved` | (3388, 5, 32) | False where the isophote sat at photutils' 13-point floor |
| `r_inner_valid`, `r_outer_valid` | (3388, 5) | the measured radial range |
| `cog_from_density` | (3388, 5, 24) | CoG rebuilt from the density, on `cog_radius_kpc` |
| `cog_from_density_shared` | (3388, 5, 32) | the same, on the shared grid |
| `r_circ` | (3388, 5, 24) | `cog_radius_kpc × sqrt(1 − e_const)` |
| `ellip_const_prof` | (3388, 5) | the profile route, 5–30 kpc unweighted mean |
| `ellip_const_agree` | (3388, 5) | \|difference\| between the two routes — the certificate |
| `annulus_mass` | (3388, 5, 24) | `(M*(<2 kpc), ΔM₂ … ΔM₂₄)`; exactly invertible |
| `annulus_sigma_abs` | (3388, 5, 24) | its error — **diagonal**, no covariance needed |
| `annulus_negative` | (3388, 5, 24) | where the stored CoG fails to increase (0.8% of z=2 annuli) |

### 10.6 Verified, not asserted

`--selftest` checks eight structural claims; F–H are v2:

- **F** `cumsum(annulus_mass)` reproduces `cog_provided` to 1.2e-16 relative — the whitened form loses nothing.
- **G** the reconstructed CoG rises by exactly 0.0 beyond `r_outer_valid` — flat by construction.
- **H** for the 75.6% of galaxy-epochs on the dominant native ladder, `shared_grid_kpc` equals their own radii to 4e-16 and `sigma_shared` equals their measured density to 5e-15 — the product stores the measurement itself, not an interpolant.

### 10.7 Changelog, v1 → v2

- **Added** the shared 32-radius grid and everything on it (`sigma_shared` and
  friends). v1's `sigma_on_cog_grid` is retained but superseded: it was on the
  24-radius CoG grid, so it could not reach inside 2 kpc.
- **Added** `annulus_mass` / `annulus_sigma_abs` / `annulus_negative`,
  `r_circ`, `ellip_const_prof`, `ellip_const_agree`, `sigma_resolved`,
  `r_inner_valid` / `r_outer_valid`.
- **Changed** nothing that v1 already contained. `cog_provided` is untouched.
- **Corrected a claim in this document, not a number**: section 4 said to use
  the `test` flag as the quality cut for anything shape-related. That was an
  inference and it was wrong. `test` gates the *fit's convergence*, not the
  *ellipticity profile*: 35% of `test=False` galaxy-epochs carry full
  ellipticity and PA profiles, and where present those profiles are not
  degenerate (median e = 0.265, no zeros, no negatives, radial scatter 0.079
  against 0.0855 for `test=True`) and reproduce the recorded constant shape as
  well as `test=True` ones do. The correct gate for the profile is
  `has_shape_cols`; the correct certificate for the constant shape is
  `ellip_const_agree`. `test` remains a good cut for the CoG-ratio *recovery*
  outliers, which is a different thing.

---

## 11. Is the error model good enough to fit with? Three tests (2026-08-31)

Nothing in this programme has ever fitted with an error bar, so before the C17
likelihood is built the error model has to be shown to be the right SIZE. Code:
`scripts/profile_error_validation.py`. Figure:
`figures/profile_error_validation.png`.

### 11.1 The curve-of-growth error model: complete at z = 0.4, incomplete above it

A flexible five-parameter profile family is fitted to each measured curve of
growth by **generalised least squares** — residuals whitened by the Cholesky
factor of the covariance, so this is a real GLS fit and not a diagonal fit with
a covariance quoted afterwards. With 24 radii and 5 parameters there are 19
degrees of freedom, and a correctly sized error model gives a reduced
chi-square near 1. 800 galaxies, four nested covariances:

| covariance | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| `sigma_iso` only, **diagonal** | 0.4 | 0.3 | 0.2 | 0.1 | 0.03 |
| `sigma_iso` only, **full covariance** | 3.4 | 3.4 | 3.7 | 4.6 | 7.6 |
| `+ sigma_shape` | 3.4 | 3.4 | 3.7 | 4.6 | 7.6 |
| `+ sigma_proj` (z = 0.4 only) | **1.1** | — | — | — | — |

Four things follow, and the last is the headline.

**The diagonal makes the fit look far better than it is.** Reduced chi-square
0.4 falling to 0.03 is not a good fit; it is an error model whose off-diagonal
structure has been thrown away. Under the diagonal, the large common mode is
compared point by point and the model absorbs it in its normalisation. Under
the true covariance the common mode is *cheap* and what is tested is the fine
radial structure — which is the stringent and correct test.

**The coherent shape term buys nothing** — 3.4 against 3.4, identical to three
digits. That is not a null result, it is a structural one: `sigma_shape`
multiplies the whole curve coherently, and a full cumulative covariance already
makes a common mode nearly free. The term is real and should be carried, but it
cannot repair a shape misfit.

**Adding the measured projection term brings z = 0.4 to 1.1.** From 3.4 to 1.1
by adding a term that was *measured*, not tuned: the per-galaxy three-projection
scatter with the radius-to-radius correlation matrix of section 11.3. **At
z = 0.4 the error model is complete**, and the thing that had been missing was
the orientation of the galaxy.

**Above z = 0.4 it is not complete, and we cannot complete it.** Without the
projection term the chi-square rises to 7.6 at z = 2. If projection is the whole
explanation there too, `sigma_proj` at z = 2 would have to be roughly 2.6 times
its z = 0.4 value — plausible, since galaxies are more irregular then, but
**unmeasurable from this drop**. This is the single strongest argument for
re-supplying the stellar mass maps (section 9).

### 11.2 The density profile as an independent check

The stored curve of growth and the curve rebuilt from the 1-D density profile
are two routes to the same quantity, so their difference divided by the
predicted error is a pull whose width should be about 1:

| width of the pull | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| 2 kpc | 1.15 | 1.15 | 1.13 | 1.13 | 1.16 |
| 10 kpc | 0.54 | 0.58 | 0.56 | 0.59 | 0.62 |
| 52 kpc | 0.39 | 0.42 | 0.43 | 0.48 | 0.50 |
| 148 kpc | 0.69 | 0.66 | 0.66 | 0.66 | 0.63 |

The error model is right-sized to within a factor of two at every radius and
every epoch, slightly **generous** at intermediate radii (pull below 1) and
slightly **tight** at 2 kpc (1.15), where section 5's smooth-model
reconstruction bias lives. Notably the width barely moves with redshift, so
whatever is missing at high z in 11.1 is *not* missing from this comparison —
consistent with it being projection, which cancels here because both routes use
the same projection.

### 11.3 The projection effect at z = 0.4

**It is strongly correlated between radii.** Measured from the three sky
projections of the same galaxies:

| corr | 10 | 30 | 50 | 75 | 100 | 150 kpc |
|---|---|---|---|---|---|---|
| 10 | 1.000 | 0.939 | 0.884 | 0.823 | 0.774 | 0.703 |
| 30 | | 1.000 | 0.970 | 0.912 | 0.860 | 0.782 |
| 100 | | | | | 1.000 | 0.946 |

A galaxy seen edge-on is over-dense at **every** radius. Treating this term as
diagonal would badly understate it, and it is why 11.1 works: the term supplies
correlated error, which is exactly the kind the cumulative covariance leaves
uncovered.

**But it is only a few per cent of the signal.** Against the galaxy-to-galaxy
scatter at fixed halo mass — the thing a model is actually asked to reproduce:

| radius [kpc] | 10 | 30 | 50 | 75 | 100 | 150 |
|---|---|---|---|---|---|---|
| projection scatter [dex] | 0.0234 | 0.0157 | 0.0120 | 0.0092 | 0.0073 | 0.0051 |
| scatter at fixed `Mh` [dex] | 0.1300 | 0.1225 | 0.1238 | 0.1245 | 0.1247 | 0.1248 |
| **projection share of the variance** | **3.2%** | 1.6% | 0.9% | 0.6% | 0.3% | 0.2% |

**This corrects an overstatement in section 6.** Projection is the dominant
*measurement error* — larger than the statistical term from 10 to 100 kpc, and
the term that completes the error model at z = 0.4. It is **not** a large part
of the signal: at most 3.2% of the variance at fixed halo mass, falling to 0.2%
at 150 kpc. So it matters enormously for a per-galaxy likelihood and hardly at
all as an excuse for population-level model failure. Both statements are true
and they are about different quantities; anything quoting one should quote the
other.
