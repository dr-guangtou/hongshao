# The 1-D profile data: what is in it, what it means, and how to use it

*Written 2026-08-30 on branch `exp68-profile-data`. Code:
`hongshao/profile_data.py`. Figure: `figures/profile_data_qa.png`
(`scripts/profile_data_qa.py`). Product:
`data/processed/tng300_072_profiles_v1.npz` (gitignored, rebuildable in
about four minutes).*

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
