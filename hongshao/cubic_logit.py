"""The cubic-logit projected profile.

The profile was developed in Exp62 as a compact analytic representation of
measured stellar curves of growth. Its infinite-aperture enclosed fraction is

    F(R) = sigmoid(P(x)),  x = ln(R / R50),
    P(x) = a x + b x**2 + c x**3,
    a = m + b**2 / (3 c),  m > 0, c > 0.

The derivative is ``P'(x) = m + 3 c (x + b / (3 c))**2`` and is therefore
strictly positive. Consequently F rises continuously from zero to one and
``R50`` is exactly the infinite-aperture projected half-mass radius.

This module uses physical shape parameters rather than optimizer coordinates:
``derivative_margin=m``, ``asymmetry=b``, and ``cubic_curvature=c``. Exp62's
optimizer stored ln(m) and ln(c); callers must exponentiate those two values.

Important domain warning
------------------------
For every c > 0 the implied projected surface density approaches zero at the
exact center as well as at infinity. It therefore has a central turnover when
extrapolated far enough inward. The family was validated on resolved 2--148
kpc curves of growth; do not assume that its R -> 0 extrapolation is a physical
galaxy core. ``cubic_logit_aperture_cog`` supports HongShao's usual convention
of normalizing at the last measured aperture without relabeling that aperture
mass as the infinite total.

The circular Fourier transform has no known elementary closed form. The
numerical implementation evaluates its order-zero Hankel transform with fixed
Gauss--Legendre quadrature in log radius. Spatial frequency ``k`` is the
angular wavenumber used in ``J0(k R)``, in inverse units of ``R``.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import expit, gammaln, j0, kve, logit, roots_legendre


def _validate_shape_parameters(
    half_mass_radius: float,
    derivative_margin: float,
    asymmetry: float,
    cubic_curvature: float,
) -> None:
    values = np.asarray(
        [half_mass_radius, derivative_margin, asymmetry, cubic_curvature], float
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("cubic-logit shape parameters must be finite")
    if half_mass_radius <= 0.0:
        raise ValueError("half_mass_radius must be positive")
    if derivative_margin <= 0.0:
        raise ValueError("derivative_margin must be positive")
    if cubic_curvature <= 0.0:
        raise ValueError("cubic_curvature must be positive")


def cubic_logit_linear_coefficient(
    derivative_margin: float,
    asymmetry: float,
    cubic_curvature: float,
) -> float:
    """Return the constrained linear coefficient ``a`` of the cubic warp."""
    _validate_shape_parameters(1.0, derivative_margin, asymmetry, cubic_curvature)
    return float(derivative_margin + asymmetry**2 / (3.0 * cubic_curvature))


def cubic_logit_warp(
    log_radius_ratio: np.ndarray | float,
    derivative_margin: float,
    asymmetry: float,
    cubic_curvature: float,
) -> np.ndarray:
    """Evaluate ``P(ln(R/R50))`` for scalar physical shape parameters."""
    linear_coefficient = cubic_logit_linear_coefficient(
        derivative_margin, asymmetry, cubic_curvature
    )
    coordinate = np.asarray(log_radius_ratio, float)
    return (
        linear_coefficient * coordinate
        + asymmetry * coordinate**2
        + cubic_curvature * coordinate**3
    )


def cubic_logit_warp_derivative(
    log_radius_ratio: np.ndarray | float,
    derivative_margin: float,
    asymmetry: float,
    cubic_curvature: float,
) -> np.ndarray:
    """Evaluate the positive derivative ``dP/dln(R/R50)``."""
    _validate_shape_parameters(1.0, derivative_margin, asymmetry, cubic_curvature)
    coordinate = np.asarray(log_radius_ratio, float)
    return (
        derivative_margin
        + 3.0
        * cubic_curvature
        * (coordinate + asymmetry / (3.0 * cubic_curvature)) ** 2
    )


def cubic_logit_cumulative_fraction(
    radius: np.ndarray | float,
    half_mass_radius: float,
    derivative_margin: float,
    asymmetry: float,
    cubic_curvature: float,
) -> np.ndarray:
    """Infinite-aperture projected mass fraction inside ``radius``.

    ``radius`` may be scalar or array-like and must be non-negative. Shape
    parameters are scalar. The value at exactly zero is defined by its limit,
    F(0)=0.
    """
    _validate_shape_parameters(
        half_mass_radius, derivative_margin, asymmetry, cubic_curvature
    )
    radius_array = np.asarray(radius, float)
    if not np.all(np.isfinite(radius_array)) or np.any(radius_array < 0.0):
        raise ValueError("radius must be finite and non-negative")
    output = np.zeros_like(radius_array)
    positive = radius_array > 0.0
    coordinate = np.log(radius_array[positive] / half_mass_radius)
    output[positive] = expit(
        cubic_logit_warp(coordinate, derivative_margin, asymmetry, cubic_curvature)
    )
    return output


def cubic_logit_enclosed_mass(
    radius: np.ndarray | float,
    total_mass: float,
    half_mass_radius: float,
    derivative_margin: float,
    asymmetry: float,
    cubic_curvature: float,
) -> np.ndarray:
    """Projected mass enclosed by ``radius`` for an infinite total mass."""
    if not np.isfinite(total_mass) or total_mass < 0.0:
        raise ValueError("total_mass must be finite and non-negative")
    return total_mass * cubic_logit_cumulative_fraction(
        radius,
        half_mass_radius,
        derivative_margin,
        asymmetry,
        cubic_curvature,
    )


def cubic_logit_aperture_cog(
    radius: np.ndarray | float,
    aperture_mass: float,
    aperture_radius: float,
    half_mass_radius: float,
    derivative_margin: float,
    asymmetry: float,
    cubic_curvature: float,
) -> np.ndarray:
    """CoG normalized to a measured mass inside ``aperture_radius``.

    This is the appropriate interface when ``aperture_mass`` is HongShao's
    measured ``Mstar(<148 kpc)``. The implied infinite total is
    ``aperture_mass / F(aperture_radius)``.
    """
    if not np.isfinite(aperture_mass) or aperture_mass < 0.0:
        raise ValueError("aperture_mass must be finite and non-negative")
    aperture_fraction = float(
        cubic_logit_cumulative_fraction(
            aperture_radius,
            half_mass_radius,
            derivative_margin,
            asymmetry,
            cubic_curvature,
        )
    )
    if aperture_fraction <= 0.0:
        raise ValueError("aperture_radius must enclose a positive model fraction")
    return (
        aperture_mass
        * cubic_logit_cumulative_fraction(
            radius,
            half_mass_radius,
            derivative_margin,
            asymmetry,
            cubic_curvature,
        )
        / aperture_fraction
    )


def cubic_logit_enclosed_radius(
    fraction: np.ndarray | float,
    half_mass_radius: float,
    derivative_margin: float,
    asymmetry: float,
    cubic_curvature: float,
) -> np.ndarray:
    """Radius enclosing an infinite-total fraction, using the real cubic root.

    The monotonic cubic reduces to ``y**3 + p*y + q = 0`` with ``p=m/c>0``.
    Its unique real solution is evaluated with a stable hyperbolic-sine form,
    so R20, R50, R80, and arbitrary enclosed radii require no root finder.
    Fractions zero and one return radii zero and infinity.
    """
    _validate_shape_parameters(
        half_mass_radius, derivative_margin, asymmetry, cubic_curvature
    )
    fraction_array = np.asarray(fraction, float)
    if (
        not np.all(np.isfinite(fraction_array))
        or np.any(fraction_array < 0.0)
        or np.any(fraction_array > 1.0)
    ):
        raise ValueError("fraction must be finite and between zero and one")

    output = np.empty_like(fraction_array)
    output[fraction_array == 0.0] = 0.0
    output[fraction_array == 1.0] = np.inf
    interior = (fraction_array > 0.0) & (fraction_array < 1.0)
    target_logit = logit(fraction_array[interior])
    depressed_linear = derivative_margin / cubic_curvature
    depressed_constant = (
        -(asymmetry**3) / (27.0 * cubic_curvature**3)
        - asymmetry * derivative_margin / (3.0 * cubic_curvature**2)
        - target_logit / cubic_curvature
    )
    argument = (
        3.0
        * depressed_constant
        / (2.0 * depressed_linear)
        * np.sqrt(3.0 / depressed_linear)
    )
    depressed_root = (
        -2.0 * np.sqrt(depressed_linear / 3.0) * np.sinh(np.arcsinh(argument) / 3.0)
    )
    log_radius_ratio = depressed_root - asymmetry / (3.0 * cubic_curvature)
    output[interior] = half_mass_radius * np.exp(log_radius_ratio)
    return output


def cubic_logit_surface_density(
    radius: np.ndarray | float,
    total_mass: float,
    half_mass_radius: float,
    derivative_margin: float,
    asymmetry: float,
    cubic_curvature: float,
) -> np.ndarray:
    """Circular projected surface density ``dM/(2 pi R dR)``.

    The output has mass per squared radius unit. Its limit at R=0 is zero for
    every strictly positive cubic curvature.
    """
    if not np.isfinite(total_mass) or total_mass < 0.0:
        raise ValueError("total_mass must be finite and non-negative")
    _validate_shape_parameters(
        half_mass_radius, derivative_margin, asymmetry, cubic_curvature
    )
    radius_array = np.asarray(radius, float)
    if not np.all(np.isfinite(radius_array)) or np.any(radius_array < 0.0):
        raise ValueError("radius must be finite and non-negative")

    output = np.zeros_like(radius_array)
    positive = radius_array > 0.0
    positive_radius = radius_array[positive]
    coordinate = np.log(positive_radius / half_mass_radius)
    warp = cubic_logit_warp(coordinate, derivative_margin, asymmetry, cubic_curvature)
    warp_derivative = cubic_logit_warp_derivative(
        coordinate, derivative_margin, asymmetry, cubic_curvature
    )
    logistic_derivative = expit(warp) * expit(-warp)
    output[positive] = (
        total_mass
        * logistic_derivative
        * warp_derivative
        / (2.0 * np.pi * positive_radius**2)
    )
    return output


def cubic_logit_surface_density_log_slope(
    radius: np.ndarray | float,
    half_mass_radius: float,
    derivative_margin: float,
    asymmetry: float,
    cubic_curvature: float,
) -> np.ndarray:
    """Return ``d ln(Sigma) / d ln(R)`` at positive radii.

    A zero is a stationary point of the projected density. The expression is

    ``-2 + P''/P' + (1 - 2F) P'``.
    """
    _validate_shape_parameters(
        half_mass_radius, derivative_margin, asymmetry, cubic_curvature
    )
    radius_array = np.asarray(radius, float)
    if not np.all(np.isfinite(radius_array)) or np.any(radius_array <= 0.0):
        raise ValueError("radius must be finite and positive")
    coordinate = np.log(radius_array / half_mass_radius)
    return _cubic_logit_density_log_slope_from_coordinate(
        coordinate, derivative_margin, asymmetry, cubic_curvature
    )


def _cubic_logit_density_log_slope_from_coordinate(
    coordinate: np.ndarray | float,
    derivative_margin: float,
    asymmetry: float,
    cubic_curvature: float,
) -> np.ndarray:
    warp = cubic_logit_warp(coordinate, derivative_margin, asymmetry, cubic_curvature)
    first_derivative = cubic_logit_warp_derivative(
        coordinate, derivative_margin, asymmetry, cubic_curvature
    )
    second_derivative = 2.0 * asymmetry + 6.0 * cubic_curvature * coordinate
    fraction = expit(warp)
    return (
        -2.0
        + second_derivative / first_derivative
        + (1.0 - 2.0 * fraction) * first_derivative
    )


def cubic_logit_density_stationary_radii(
    half_mass_radius: float,
    derivative_margin: float,
    asymmetry: float,
    cubic_curvature: float,
    *,
    grid_size: int = 1024,
    tail_probability: float = 1.0e-12,
) -> np.ndarray:
    """All resolved stationary radii of the projected surface density.

    The logarithmic-slope equation is transcendental, so these radii are found
    numerically between the declared inner and outer cumulative-probability
    tails. The returned array is sorted. Increase ``grid_size`` when auditing
    unusually extreme shape parameters.
    """
    if grid_size < 32:
        raise ValueError("grid_size must be at least 32")
    if not 0.0 < tail_probability < 0.5:
        raise ValueError("tail_probability must be between zero and one half")
    boundary_radius = cubic_logit_enclosed_radius(
        np.asarray([tail_probability, 1.0 - tail_probability]),
        half_mass_radius,
        derivative_margin,
        asymmetry,
        cubic_curvature,
    )
    coordinate_boundary = np.log(boundary_radius / half_mass_radius)

    def slope(coordinate_value: float) -> float:
        return float(
            _cubic_logit_density_log_slope_from_coordinate(
                coordinate_value,
                derivative_margin,
                asymmetry,
                cubic_curvature,
            )
        )

    lower_coordinate, upper_coordinate = coordinate_boundary
    while slope(lower_coordinate) <= 0.0:
        lower_coordinate *= 2.0
        if lower_coordinate < -512.0:
            raise RuntimeError("failed to bracket the inner density slope")
    while slope(upper_coordinate) >= 0.0:
        upper_coordinate *= 2.0
        if upper_coordinate > 512.0:
            raise RuntimeError("failed to bracket the outer density slope")
    coordinate = np.linspace(lower_coordinate, upper_coordinate, grid_size)

    sampled_slope = _cubic_logit_density_log_slope_from_coordinate(
        coordinate,
        derivative_margin,
        asymmetry,
        cubic_curvature,
    )
    roots = []
    for lower, upper, lower_slope, upper_slope in zip(
        coordinate[:-1],
        coordinate[1:],
        sampled_slope[:-1],
        sampled_slope[1:],
    ):
        if lower_slope == 0.0:
            roots.append(lower)
        elif lower_slope * upper_slope < 0.0:
            roots.append(brentq(slope, lower, upper))
    if sampled_slope[-1] == 0.0:
        roots.append(coordinate[-1])
    if not roots:
        return np.empty(0)
    return half_mass_radius * np.exp(np.unique(np.round(roots, decimals=13)))


def cubic_logit_density_peak_radius(
    half_mass_radius: float,
    derivative_margin: float,
    asymmetry: float,
    cubic_curvature: float,
    *,
    grid_size: int = 1024,
    tail_probability: float = 1.0e-12,
) -> float:
    """Radius of the global maximum of the projected surface density."""
    stationary = cubic_logit_density_stationary_radii(
        half_mass_radius,
        derivative_margin,
        asymmetry,
        cubic_curvature,
        grid_size=grid_size,
        tail_probability=tail_probability,
    )
    if len(stationary) == 0:
        raise RuntimeError("no density maximum found on the audited radial support")
    coordinate = np.log(stationary / half_mass_radius)
    warp = cubic_logit_warp(coordinate, derivative_margin, asymmetry, cubic_curvature)
    warp_derivative = cubic_logit_warp_derivative(
        coordinate, derivative_margin, asymmetry, cubic_curvature
    )
    log_density_without_constant = (
        -np.logaddexp(0.0, -warp)
        - np.logaddexp(0.0, warp)
        + np.log(warp_derivative)
        - 2.0 * coordinate
    )
    return float(stationary[np.argmax(log_density_without_constant)])


def render_cubic_logit_image(
    x_coordinate: np.ndarray,
    y_coordinate: np.ndarray,
    total_mass: float,
    half_mass_radius: float,
    derivative_margin: float,
    asymmetry: float,
    cubic_curvature: float,
    *,
    axis_ratio: float = 1.0,
    position_angle: float = 0.0,
) -> np.ndarray:
    """Evaluate a circular or elliptical projected surface-density image.

    Coordinates use the same length unit as ``half_mass_radius``. The position
    angle is in radians, measured counterclockwise. Elliptical radius is
    ``sqrt(x_major**2 + (y_minor/axis_ratio)**2)`` and the density is divided
    by ``axis_ratio`` so every elliptical aperture encloses the same fraction
    as its circular counterpart.

    This evaluates the density at pixel coordinates; it is not a finite-pixel
    integral. Subpixel sampling is advisable when the density turnover is not
    resolved.
    """
    if not np.isfinite(axis_ratio) or not 0.0 < axis_ratio <= 1.0:
        raise ValueError("axis_ratio must be finite and in (0, 1]")
    if not np.isfinite(position_angle):
        raise ValueError("position_angle must be finite")
    x_array, y_array = np.broadcast_arrays(
        np.asarray(x_coordinate, float), np.asarray(y_coordinate, float)
    )
    cosine = np.cos(position_angle)
    sine = np.sin(position_angle)
    major_coordinate = cosine * x_array + sine * y_array
    minor_coordinate = -sine * x_array + cosine * y_array
    elliptical_radius = np.sqrt(
        major_coordinate**2 + (minor_coordinate / axis_ratio) ** 2
    )
    return (
        cubic_logit_surface_density(
            elliptical_radius,
            total_mass,
            half_mass_radius,
            derivative_margin,
            asymmetry,
            cubic_curvature,
        )
        / axis_ratio
    )


def _log_radius_quadrature(
    half_mass_radius: float,
    derivative_margin: float,
    asymmetry: float,
    cubic_curvature: float,
    quadrature_order: int,
    tail_probability: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if quadrature_order < 32:
        raise ValueError("quadrature_order must be at least 32")
    if not 0.0 < tail_probability < 0.5:
        raise ValueError("tail_probability must be between zero and one half")
    boundary_radius = cubic_logit_enclosed_radius(
        np.asarray([tail_probability, 1.0 - tail_probability]),
        half_mass_radius,
        derivative_margin,
        asymmetry,
        cubic_curvature,
    )
    boundary = np.log(boundary_radius / half_mass_radius)
    nodes, weights = roots_legendre(quadrature_order)
    log_radius_ratio = 0.5 * (boundary[1] - boundary[0]) * nodes + 0.5 * np.sum(
        boundary
    )
    weights = 0.5 * (boundary[1] - boundary[0]) * weights
    warp = cubic_logit_warp(
        log_radius_ratio, derivative_margin, asymmetry, cubic_curvature
    )
    probability_density = (
        expit(warp)
        * expit(-warp)
        * cubic_logit_warp_derivative(
            log_radius_ratio, derivative_margin, asymmetry, cubic_curvature
        )
    )
    normalized_weights = weights * probability_density
    normalized_weights /= np.sum(normalized_weights)
    radius = half_mass_radius * np.exp(log_radius_ratio)
    return log_radius_ratio, radius, normalized_weights


def cubic_logit_radial_moment(
    power: float,
    half_mass_radius: float,
    derivative_margin: float,
    asymmetry: float,
    cubic_curvature: float,
    *,
    quadrature_order: int = 512,
    tail_probability: float = 1.0e-10,
) -> float:
    """Numerical infinite-total radial moment ``<R**power>``.

    All real moments exist because both radial tails decline faster than every
    power. Accuracy is controlled by ``quadrature_order`` and the omitted
    probability in each tail.
    """
    if not np.isfinite(power):
        raise ValueError("power must be finite")
    _, radius, probability_weight = _log_radius_quadrature(
        half_mass_radius,
        derivative_margin,
        asymmetry,
        cubic_curvature,
        quadrature_order,
        tail_probability,
    )
    return float(np.sum(probability_weight * radius**power))


def cubic_logit_fourier_transform(
    spatial_frequency: np.ndarray | float,
    half_mass_radius: float,
    derivative_margin: float,
    asymmetry: float,
    cubic_curvature: float,
    *,
    quadrature_order: int = 1024,
    tail_probability: float = 1.0e-10,
) -> np.ndarray:
    """Normalized circular 2-D Fourier transform of the surface density.

    The convention is ``T(k)=2 pi/M integral Sigma(R) J0(kR) R dR``. Thus
    T(0)=1 and a circular convolution multiplies this result by the normalized
    PSF transform. The transform is numerical; increase ``quadrature_order``
    and check convergence when ``k * half_mass_radius`` is large.
    """
    frequency = np.asarray(spatial_frequency, float)
    if not np.all(np.isfinite(frequency)) or np.any(frequency < 0.0):
        raise ValueError("spatial_frequency must be finite and non-negative")
    _, radius, probability_weight = _log_radius_quadrature(
        half_mass_radius,
        derivative_margin,
        asymmetry,
        cubic_curvature,
        quadrature_order,
        tail_probability,
    )
    flattened = frequency.reshape(-1)
    transform = np.sum(
        j0(flattened[:, None] * radius[None, :]) * probability_weight[None, :],
        axis=1,
    ).reshape(frequency.shape)
    transform = np.where(frequency == 0.0, 1.0, transform)
    return transform


def gaussian_psf_fourier_transform(
    spatial_frequency: np.ndarray | float, sigma: float
) -> np.ndarray:
    """Normalized 2-D Fourier transform of a circular Gaussian PSF."""
    frequency = np.asarray(spatial_frequency, float)
    if not np.all(np.isfinite(frequency)) or np.any(frequency < 0.0):
        raise ValueError("spatial_frequency must be finite and non-negative")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    return np.exp(-0.5 * (sigma * frequency) ** 2)


def moffat_psf_fourier_transform(
    spatial_frequency: np.ndarray | float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Normalized 2-D Fourier transform of a circular Moffat PSF.

    The real-space convention is

    ``P(R)=(beta-1)/(pi alpha**2) * (1 + R**2/alpha**2)**(-beta)``

    and requires ``beta>1`` for finite total flux. The transform is expressed
    with the modified Bessel function K of order ``beta-1``.
    """
    frequency = np.asarray(spatial_frequency, float)
    if not np.all(np.isfinite(frequency)) or np.any(frequency < 0.0):
        raise ValueError("spatial_frequency must be finite and non-negative")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive")
    if not np.isfinite(beta) or beta <= 1.0:
        raise ValueError("beta must be finite and greater than one")

    argument = alpha * frequency
    output = np.ones_like(argument)
    positive = argument > 0.0
    order = beta - 1.0
    positive_argument = argument[positive]
    log_transform = (
        (2.0 - beta) * np.log(2.0)
        - gammaln(order)
        + order * np.log(positive_argument)
        + np.log(kve(order, positive_argument))
        - positive_argument
    )
    output[positive] = np.exp(log_transform)
    unstable = positive & ~np.isfinite(output)
    for index in np.flatnonzero(unstable.reshape(-1)):
        transformed_argument = float(argument.reshape(-1)[index])
        output.reshape(-1)[index] = quad(
            lambda scaled_radius: (
                2.0
                * (beta - 1.0)
                * scaled_radius
                * np.exp(-beta * np.log1p(scaled_radius**2))
                * j0(transformed_argument * scaled_radius)
            ),
            0.0,
            np.inf,
            epsabs=1.0e-11,
            epsrel=1.0e-10,
            limit=300,
        )[0]
    return output


def _self_check() -> None:
    shape = (12.0, 0.8, 0.4, 0.08)
    fractions = np.asarray([0.2, 0.5, 0.8])
    radii = cubic_logit_enclosed_radius(fractions, *shape)
    recovered = cubic_logit_cumulative_fraction(radii, *shape)
    assert np.allclose(recovered, fractions, atol=1.0e-10)
    assert cubic_logit_fourier_transform(0.0, *shape) == 1.0
    print("cubic-logit self-check OK")


if __name__ == "__main__":
    _self_check()
