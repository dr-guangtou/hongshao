"""Focused checks for the promoted cubic-logit projected profile."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.signal import fftconvolve
from scipy.special import j0

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hongshao.cubic_logit import (  # noqa: E402
    cubic_logit_aperture_cog,
    cubic_logit_cumulative_fraction,
    cubic_logit_density_peak_radius,
    cubic_logit_density_stationary_radii,
    cubic_logit_enclosed_radius,
    cubic_logit_fourier_transform,
    cubic_logit_radial_moment,
    cubic_logit_surface_density,
    cubic_logit_surface_density_log_slope,
    gaussian_psf_fourier_transform,
    moffat_psf_fourier_transform,
    render_cubic_logit_image,
)

HALF_MASS_RADIUS = 12.0
DERIVATIVE_MARGIN = 0.8
ASYMMETRY = 0.4
CUBIC_CURVATURE = 0.08


def _shape_parameters() -> tuple[float, float, float, float]:
    return (
        HALF_MASS_RADIUS,
        DERIVATIVE_MARGIN,
        ASYMMETRY,
        CUBIC_CURVATURE,
    )


def test_cubic_logit_is_monotonic_and_has_exact_enclosed_radii() -> None:
    radius = np.geomspace(1.0e-5, 1.0e6, 4000)
    fraction = cubic_logit_cumulative_fraction(radius, *_shape_parameters())
    assert np.all(np.diff(fraction) >= 0.0)
    assert np.isclose(
        cubic_logit_cumulative_fraction(HALF_MASS_RADIUS, *_shape_parameters()),
        0.5,
        atol=1.0e-15,
    )

    targets = np.asarray([0.01, 0.2, 0.5, 0.8, 0.99])
    enclosed = cubic_logit_enclosed_radius(targets, *_shape_parameters())
    recovered = cubic_logit_cumulative_fraction(enclosed, *_shape_parameters())
    assert np.allclose(recovered, targets, atol=1.0e-10, rtol=0.0)
    assert enclosed[0] < enclosed[1] < enclosed[2] < enclosed[3] < enclosed[4]


def test_surface_density_integrates_to_the_cumulative_fraction() -> None:
    total_mass = 3.2e11
    shape = _shape_parameters()
    lower_fraction, upper_fraction = 1.0e-10, 1.0 - 1.0e-10
    lower_radius, upper_radius = cubic_logit_enclosed_radius(
        np.asarray([lower_fraction, upper_fraction]), *shape
    )

    def mass_per_log_radius(log_radius: float) -> float:
        radius = np.exp(log_radius)
        density = cubic_logit_surface_density(radius, total_mass, *shape)
        return float(2.0 * np.pi * radius**2 * density)

    integrated = quad(
        mass_per_log_radius,
        np.log(lower_radius),
        np.log(upper_radius),
        epsabs=1.0e-4,
        epsrel=2.0e-10,
    )[0]
    expected = total_mass * (upper_fraction - lower_fraction)
    assert np.isclose(integrated, expected, rtol=2.0e-9)
    assert cubic_logit_surface_density(0.0, total_mass, *shape) == 0.0


def test_density_peak_is_a_stationary_global_maximum() -> None:
    shape = _shape_parameters()
    stationary = cubic_logit_density_stationary_radii(*shape)
    peak_radius = cubic_logit_density_peak_radius(*shape)
    assert len(stationary) >= 1
    assert peak_radius in stationary
    assert abs(cubic_logit_surface_density_log_slope(peak_radius, *shape)) < 1.0e-9
    nearby = peak_radius * np.asarray([0.98, 1.0, 1.02])
    density = cubic_logit_surface_density(nearby, 1.0, *shape)
    assert density[1] > density[0]
    assert density[1] > density[2]


def test_aperture_normalization_is_exact() -> None:
    aperture_radius = 148.0
    aperture_mass = 2.4e11
    radius = np.asarray([2.0, HALF_MASS_RADIUS, aperture_radius])
    cog = cubic_logit_aperture_cog(
        radius,
        aperture_mass,
        aperture_radius,
        *_shape_parameters(),
    )
    assert np.isclose(cog[-1], aperture_mass, rtol=2.0e-15)
    assert np.all(np.diff(cog) > 0.0)


def test_circular_and_elliptical_images_converge_to_the_total_mass() -> None:
    total_mass = 1.0
    extent = 18.0 * HALF_MASS_RADIUS

    def image_sum(npixel: int, axis_ratio: float) -> float:
        pixel_scale = 2.0 * extent / npixel
        coordinate = (np.arange(npixel) + 0.5) * pixel_scale - extent
        x_coordinate, y_coordinate = np.meshgrid(coordinate, coordinate)
        density = render_cubic_logit_image(
            x_coordinate,
            y_coordinate,
            total_mass,
            *_shape_parameters(),
            axis_ratio=axis_ratio,
            position_angle=0.37,
        )
        return float(np.sum(density) * pixel_scale**2)

    for axis_ratio in (1.0, 0.55):
        coarse_error = abs(image_sum(240, axis_ratio) - total_mass)
        fine_error = abs(image_sum(720, axis_ratio) - total_mass)
        assert fine_error < coarse_error
        assert fine_error < 0.01


def test_fourier_transform_has_correct_origin_and_second_moment() -> None:
    shape = _shape_parameters()
    spatial_frequency = np.asarray([0.0, 1.0e-4 / HALF_MASS_RADIUS])
    transform = cubic_logit_fourier_transform(
        spatial_frequency, *shape, quadrature_order=768
    )
    second_moment = cubic_logit_radial_moment(2.0, *shape, quadrature_order=768)
    expected = 1.0 - spatial_frequency[1] ** 2 * second_moment / 4.0
    assert transform[0] == 1.0
    assert np.isclose(transform[1], expected, atol=2.0e-12, rtol=0.0)


def test_gaussian_and_moffat_psf_transforms_match_direct_hankel_integrals() -> None:
    spatial_frequency = np.asarray([0.0, 0.07, 0.2])
    sigma = 1.3
    gaussian = gaussian_psf_fourier_transform(spatial_frequency, sigma)
    assert gaussian[0] == 1.0

    alpha = 2.1
    beta = 3.7
    moffat = moffat_psf_fourier_transform(spatial_frequency, alpha, beta)
    assert moffat[0] == 1.0

    def moffat_profile(radius: float) -> float:
        return (
            (beta - 1.0) / (np.pi * alpha**2) * (1.0 + (radius / alpha) ** 2) ** (-beta)
        )

    for frequency, analytic in zip(spatial_frequency[1:], moffat[1:]):
        direct = quad(
            lambda radius: (
                2.0 * np.pi * radius * moffat_profile(radius) * j0(frequency * radius)
            ),
            0.0,
            np.inf,
            epsabs=2.0e-10,
            epsrel=2.0e-9,
            limit=400,
        )[0]
        assert np.isclose(analytic, direct, atol=2.0e-8, rtol=2.0e-8)

    large_beta = 200.0
    matched_alpha = sigma * np.sqrt(2.0 * large_beta)
    limiting_moffat = moffat_psf_fourier_transform(
        spatial_frequency, matched_alpha, large_beta
    )
    assert np.allclose(limiting_moffat, gaussian, atol=2.0e-3, rtol=2.0e-3)


def test_full_grid_psf_convolution_conserves_rendered_flux() -> None:
    npixel = 240
    pixel_scale = 0.25
    coordinate = (np.arange(npixel) + 0.5 - npixel / 2.0) * pixel_scale
    x_coordinate, y_coordinate = np.meshgrid(coordinate, coordinate)
    radius = np.hypot(x_coordinate, y_coordinate)
    image = render_cubic_logit_image(
        x_coordinate,
        y_coordinate,
        1.0,
        *_shape_parameters(),
    )
    sigma = 0.9
    gaussian_psf = np.exp(-0.5 * (radius / sigma) ** 2)
    alpha = 1.2
    beta = 4.765
    moffat_psf = (1.0 + (radius / alpha) ** 2) ** (-beta)
    for psf in (gaussian_psf, moffat_psf):
        psf /= np.sum(psf)
        convolved = fftconvolve(image, psf, mode="full")
        assert np.isclose(np.sum(convolved), np.sum(image), rtol=2.0e-14)
