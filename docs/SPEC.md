# HongShao architecture specification

This file records stable library interfaces. Scientific experiment decisions
remain in the corresponding `experiments/expNN_slug/README.md`.

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
