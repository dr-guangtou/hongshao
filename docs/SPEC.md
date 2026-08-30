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
`data/processed/tng300_072_profiles_v1.npz` and is the only place a new
experiment should get isophote profiles, surface mass densities, or profile
uncertainties. Reference: `doc/tng300_profile_data.md`.

Interface guarantees:

- `load_profiles()` returns a dict of arrays indexed `(galaxy, epoch, radius)`
  with galaxy 0..3387 matching every other product in the repo and epoch 0..4
  for z = 0.4, 0.7, 1.0, 1.5, 2.0.
- `cog_provided` remains the fitting target. The two reconstructions
  (`cog_const_shape`, `cog_var_shape`) exist to explain the stored curve and to
  carry its error, and a fit must not switch target to one of them without an
  explicit recorded decision.
- The error model has three separately named terms and they must not be summed
  silently: `sigma_iso_dex` (statistical, with the covariance built by
  `cog_covariance`), `sigma_shape_dex` (coherent in radius), and
  `sigma_proj_dex` (projection; measured at z = 0.4 and outside 10 kpc only).
- A curve of growth's radius-to-radius covariance is fixed by its variance,
  `Cov(M_i, M_j) = Var(M_min(i,j))`. Any objective treating the radii as
  independent must say so explicitly and justify it.
- `iso_shape_ok` is the profile-quality cut. It is separate from, and never a
  substitute for, the fitting-sample rule above; the two compose.

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
