"""Focused checks for the Exp67 squared-slope density grammar."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from exp67_grammar import (  # noqa: E402
    RADII_KPC,
    analytic_property_record,
    cog_log,
    density_log_slope,
    fit_profile,
    integrated_density_rate,
    logarithmic_density_rate,
    round_a_specs,
    surface_density,
    synthetic_parameters,
)


def test_round_a_repeats_the_complete_relaxed_atom_library() -> None:
    specs = round_a_specs()
    assert len(specs) == 88
    kinds = {kind for spec in specs.values() for kind in spec.atom_kinds}
    assert {
        "sigmoid",
        "rational",
        "arctangent",
        "shoulder",
        "harmonic",
        "damped_harmonic",
        "wave_packet",
        "chirp",
        "log_harmonic",
    } <= kinds
    assert all(spec.n_parameter <= 6 for spec in specs.values())


def test_squared_slope_stays_physical_at_large_signed_contrast() -> None:
    coordinate = np.linspace(0.0, 18.0, 6001)
    radius = np.concatenate(([0.0], np.geomspace(1.0e-7, 1.0e6, 2000)))
    for name in (
        "harmonic_sine_w1",
        "log_harmonic_cosine_w4",
        "wave_packet_sine_w1_w2",
        "chirp_cosine_kn0p25_w2",
    ):
        spec = round_a_specs()[name]
        parameters = synthetic_parameters(spec)
        parameters[3] = 12.0
        rate = logarithmic_density_rate(spec, parameters, coordinate)
        density = surface_density(spec, parameters, radius, grid_size=2048)
        slope = density_log_slope(spec, parameters, radius)
        assert np.all(rate >= 0.0), name
        assert np.all(np.isfinite(density)), name
        assert np.all(density >= 0.0), name
        assert np.all(density[radius <= RADII_KPC[-1]] > 0.0), name
        assert np.all(np.diff(density) <= 5.0e-11 * density[0]), name
        assert np.all(slope <= 1.0e-13), name
        assert slope[0] == 0.0, name


def test_single_harmonic_squared_rate_uses_exact_integral() -> None:
    coordinate = np.linspace(0.0, 9.0, 30001)
    specs = [
        spec
        for spec in round_a_specs().values()
        if spec.integration_class == "elementary_squared_harmonic"
    ]
    assert specs
    for spec in specs:
        parameters = synthetic_parameters(spec)
        exact = integrated_density_rate(spec, parameters, coordinate)
        numeric = cumulative_trapezoid(
            logarithmic_density_rate(spec, parameters, coordinate),
            coordinate,
            initial=0.0,
        )
        assert np.max(np.abs(exact - numeric)) < 3.0e-7, spec.name


def test_aperture_normalization_and_doubled_resolution() -> None:
    for spec in round_a_specs().values():
        parameters = synthetic_parameters(spec)
        coarse = cog_log(spec, parameters, RADII_KPC, grid_size=512)
        fine = cog_log(spec, parameters, RADII_KPC, grid_size=2048)
        assert abs(fine[-1] - parameters[0]) < 1.0e-12, spec.name
        assert np.all(np.diff(fine) > 0.0), spec.name
        assert np.max(np.abs(coarse - fine)) < 4.0e-4, spec.name


def test_synthetic_recovery_for_every_atom_class() -> None:
    representatives = (
        "constant",
        "sigmoid_w0p7",
        "rational",
        "arctangent_w0p7",
        "shoulder_w0p7",
        "harmonic_sine_w1",
        "damped_cosine_l0p75_w2",
        "wave_packet_sine_w1_w2",
        "chirp_sine_k0p25_w2",
        "log_harmonic_cosine_w2",
        "free_frequency_phase",
    )
    for name in representatives:
        spec = round_a_specs()[name]
        parameters = synthetic_parameters(spec)
        truth = cog_log(spec, parameters, RADII_KPC, grid_size=1024)
        result = fit_profile(
            spec,
            RADII_KPC,
            truth,
            grid_size=1024,
            n_starts=2,
            max_nfev=500,
            synthetic_truth_parameters=parameters,
        )
        assert result["success"], name
        assert result["cog_rms_dex"] < 3.0e-5, name


def test_property_record_distinguishes_exact_and_numerical_density() -> None:
    specs = round_a_specs()
    harmonic = analytic_property_record(specs["harmonic_sine_w1"])
    log_harmonic = analytic_property_record(specs["log_harmonic_sine_w1"])
    assert harmonic["integrated_density_rate"] == "elementary"
    assert harmonic["projected_density"] == "elementary"
    assert log_harmonic["integrated_density_rate"] == "numerical_quadrature"
    assert harmonic["finite_positive_center"]
    assert harmonic["zero_central_derivative"]
    assert harmonic["finite_total_mass"]
    assert harmonic["nonnegative_abel_deprojection"]
