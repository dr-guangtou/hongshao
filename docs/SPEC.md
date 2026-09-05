# HongShao architecture specification

This file records stable library interfaces. Scientific experiment decisions
remain in the corresponding `experiments/expNN_slug/README.md`.

## The fitting sample

**Every model fit uses all galaxies selected at z = 0.4 whose halo history and
stellar-mass history are sane, at every epoch. Halo-mass completeness is not a
fitting criterion.** The mh-complete subset — the progenitors above the
per-epoch completeness cut — is an after-fit reporting check, scored with the
fitted parameters frozen. This is the user's rule of 2026-08-30 and it applies
to every experiment.

The rule is implemented once, in
`experiments/exp54_unpinned_amplitude/selection.py::fitting_sample_mask`, and
must not be re-implemented per experiment:

- finite, positive curves of growth and a finite DiffMAH halo mass at all five
  epochs;
- the backward-growth rule (`sane_history_mask`): no epoch whose measured
  `M*(<100 kpc)` is more than `MAX_BACKWARD_DEX` below the galaxy's own z = 0.4
  value;
- no stellar-history outlier at any epoch (`stellar_history_flags`): more than
  `SHMR_TOL_DEX` from the population's running-median `M*(<100 kpc)`--`Mh`
  relation at that epoch, an adjacent-epoch rise of `M*(<100 kpc)` larger than
  `JUMP_DEX`, or an adjacent-epoch fall of `M*(<30 kpc)` larger than
  `DROP_DEX`.

One flagged epoch removes the galaxy at every epoch: a history is sane or it is
not. A fitting script must print the sample it used and the count removed by
each criterion. Where an older stage passed a per-epoch mh-complete mask into a
fit, that is the superseded path and must be corrected rather than reproduced.

## The measured profile data and its error model

`hongshao/profile_data.py` is the single reader for the raw TNG300 profile drop.
It consolidates both per-galaxy directories into
`data/processed/tng300_072_profiles_v2.npz` and is the only place a new
experiment should get isophote profiles, surface mass densities, or profile
uncertainties. Reference: `doc/tng300_profile_data.md`, section 10 for the
current data model.

Interface guarantees:

- `load_profiles()` returns the v2 dict; `load_profiles(version=1)` returns the
  earlier build, kept as a reference point. v2 is a strict superset of v1.
  Arrays are indexed `(galaxy, epoch, radius)`, galaxy 0..3387 matching every
  other product in the repo, epoch 0..4 for z = 0.4, 0.7, 1.0, 1.5, 2.0.
- `cog_provided` is the fitting target. `cog_from_density` and `annulus_mass`
  are other views of the same galaxies and must not silently replace it.
- **The two densities are different measurements.** `sigma_shared` (the
  fiducial one) is the sigma-clipped isophote density and excludes satellites
  and intracluster light; `annulus_mass` is the CoG derivative and counts
  everything in the aperture. They differ by up to 0.64 dex at z = 2 in the
  outskirts. A result fitted to one is not comparable to a result fitted to the
  other, and any experiment using either must say which.
- On the shared grid a density of exactly zero means NOT MEASURED. Every
  consumer must gate on `sigma_measured`. Beyond `r_outer_valid` the
  reconstructed curve of growth is flat by construction.
- The error model has three separately named terms and they must not be summed
  silently: `sigma_iso_dex` (statistical; covariance via `cog_covariance`, or
  use `annulus_sigma_abs`, which is diagonal), `sigma_shape_dex` (coherent in
  radius), and `sigma_proj_dex` (projection; measured at z = 0.4 and outside
  10 kpc ONLY — any other use is an assumption the experiment must state).
- A curve of growth's radius-to-radius covariance is fixed by its variance,
  `Cov(M_i, M_j) = Var(M_min(i,j))`. Any objective treating the radii as
  independent must say so explicitly and justify it.
- Profile-quality flags (`sigma_resolved`, `has_shape_cols`, `iso_shape_ok`,
  `ellip_const_agree`, `annulus_negative`) are separate from, and never a
  substitute for, the fitting-sample rule above; the two compose.
- Radii are SEMI-MAJOR axes. `r_circ` carries the circularized equivalent; the
  two differ by 19% at z = 0.4 and 14% at z = 2, a 6.1% epoch differential, so
  the convention must never be mixed within one analysis.

## Analytic projected profiles

Reusable analytic profile families live in `hongshao/`. A promoted family must
provide the cumulative projected mass, projected surface density, and enclosed
radii from the same parameterization. If no elementary Fourier transform or
PSF-convolved profile exists, the library must expose a documented numerical
route rather than silently substitute an approximation.

### Cubic-logit profile

The Exp62 five-parameter candidate is named the **cubic-logit profile**. Its
Python prefix is `cubic_logit`; do not retain experiment-only names such as
`logit_cubic5` in the public interface.

The intrinsic parameters are total projected mass, total half-mass radius,
positive derivative margin, real asymmetry, and positive cubic curvature. The
shape constraints must be enforced by validation, and the half-mass radius
must remain exactly the radius enclosing half of the infinite-aperture model.
An aperture-normalized helper must be supplied because HongShao generally uses
the mass inside the last measured aperture rather than an extrapolated total.

The module must document the model's radial domain. In particular, a
mathematically finite cumulative profile is not automatically a physically
adequate extrapolation to an unresolved galaxy center.

The public implementation is `hongshao/cubic_logit.py`; its mathematical and
usage reference is `doc/cubic_logit_profile.md`. The family is approved for
projected, finite-resolution CoG interpolation and aperture-normalized mass
predictions. It is not approved as a physical central density law, a spherical
three-dimensional profile, or an unconstrained infinite-total estimator.

## The halo-history input (exp74, 2026-09-05)

**What the model may know.** The halo's full assembly history from the
simulation — every snapshot's mass, the DiffMAH parameters, the peak and
final halo mass — is legitimate input at every epoch (the user, 2026-09-04:
the application is simulation haloes with full merger trees). What the model
must never see is the galaxy's own stellar mass at any epoch: the amplitude,
the stellar-mass–halo-mass relation and their evolution are outputs.

**What must nevertheless match the truth.** At fixed halo mass at an epoch,
the model's stellar mass must depend on the halo's FUTURE growth no more than
TNG300's does (+0.003 dex per dex at z = 2, +0.03–0.05 at z = 1–1.5). This is
a standing QA gate (`selection.partial_growth` on the residual; exp74
`stage1_eval.py` section 4). The official DiffMAH curve fails it by 4–40×
because its single whole-history fit anchored at z = 0 lets the final mass
rewrite the early history (C19).

**Three representations, one interface** (`exp74/measured.py::build_input`):
`"official"` (the DiffMAH curve, `engine.HaloCurve`), `"pre-epoch"` (a
DiffMAH curve per epoch fitted to the history before it, `HaloCurve.logt0`
at the epoch), `"measured"` (the running-peak M200c interpolated, PCHIP in
log t, power law before 1.18 Gyr, `measured.MeasuredCurve`). The engine reads
a curve through its own `log_mah_table` / `dm_dlnt_table` when it carries
them and through the DiffMAH form otherwise, so a model fits and predicts on
any of the three without change. **The model keeps both the DiffMAH and the
measured-MAH paths** (the user, 2026-09-05). Recommended default for the
application: `"measured"`; the two honest inputs give the same model to a
point or two (exp74 README).
