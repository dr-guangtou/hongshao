"""Focused checks for the Exp66 expanded symbolic-density grammar."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from exp66_grammar import (  # noqa: E402
    APERTURE_RADIUS_KPC,
    RADII_KPC,
    analytic_property_record,
    build_round_b_specs,
    cog_log,
    density_log_slope,
    integrated_density_rate,
    logarithmic_density_rate,
    normalized_expression_value,
    round_a_specs,
    surface_density,
    synthetic_parameters,
)


def test_round_a_contains_every_predeclared_atom_class() -> None:
    specs = round_a_specs()
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
    assert any(spec.free_frequency for spec in specs.values())
    assert any(spec.free_phase for spec in specs.values())
    assert any(spec.free_damping for spec in specs.values())
    assert all(spec.n_parameter <= 6 for spec in specs.values())


def test_every_round_a_expression_is_algebraically_bounded() -> None:
    coordinate = np.linspace(0.0, 18.0, 4001)
    for spec in round_a_specs().values():
        for displacement in (-0.8, 0.0, 0.8):
            parameters = synthetic_parameters(spec, displacement=displacement)
            value = normalized_expression_value(spec, parameters, coordinate)
            assert np.max(np.abs(value)) <= 1.0 + 2.0e-13, spec.name


def test_fixed_and_fitted_harmonic_integrals_are_exact() -> None:
    coordinate = np.linspace(0.0, 9.0, 20001)
    harmonic_specs = [
        spec
        for spec in round_a_specs().values()
        if spec.integration_class == "elementary_harmonic"
    ]
    assert harmonic_specs
    for spec in harmonic_specs:
        parameters = synthetic_parameters(spec)
        exact = integrated_density_rate(spec, parameters, coordinate)
        numeric = cumulative_trapezoid(
            logarithmic_density_rate(spec, parameters, coordinate),
            coordinate,
            initial=0.0,
        )
        assert np.max(np.abs(exact - numeric)) < 2.0e-7, spec.name


def test_every_round_a_density_is_physical_and_quadrature_converged() -> None:
    radius = np.concatenate(([0.0], np.geomspace(1.0e-7, 1.0e6, 1600)))
    for spec in round_a_specs().values():
        parameters = synthetic_parameters(spec)
        density = surface_density(spec, parameters, radius, grid_size=2048)
        slope = density_log_slope(spec, parameters, radius)
        coarse = cog_log(spec, parameters, RADII_KPC, grid_size=512)
        fine = cog_log(spec, parameters, RADII_KPC, grid_size=2048)
        assert np.all(np.isfinite(density)), spec.name
        assert np.all(density > 0.0), spec.name
        assert np.all(np.diff(density) <= 5.0e-11 * density[0]), spec.name
        assert np.all(slope <= 1.0e-13), spec.name
        assert abs(density[1] / density[0] - 1.0) < 1.0e-9, spec.name
        assert np.all(np.diff(fine) > 0.0), spec.name
        assert abs(fine[-1] - parameters[0]) < 1.0e-12, spec.name
        assert np.max(np.abs(coarse - fine)) < 3.0e-4, spec.name


def test_round_b_generation_is_deterministic_bounded_and_capped() -> None:
    specs = round_a_specs()
    transitions = [
        specs["sigmoid_w0p7"],
        specs["arctangent_w0p7"],
    ]
    trigonometric = [
        specs["harmonic_sine_w1"],
        specs["damped_cosine_l0p75_w2"],
        specs["wave_packet_sine_w0p5_w1"],
        specs["chirp_sine_k0p25_w2"],
    ]
    first = build_round_b_specs(transitions, trigonometric)
    second = build_round_b_specs(transitions, trigonometric)
    assert tuple(first) == tuple(second)
    assert {spec.operation for spec in first.values()} >= {
        "mean",
        "product",
        "cube",
        "centered_square",
    }
    coordinate = np.linspace(0.0, 15.0, 3001)
    for spec in first.values():
        parameters = synthetic_parameters(spec)
        value = normalized_expression_value(spec, parameters, coordinate)
        assert spec.n_parameter <= 7, spec.name
        assert np.max(np.abs(value)) <= 1.0 + 2.0e-13, spec.name


def test_property_audit_is_explicit_about_analytic_and_numerical_routes() -> None:
    specs = round_a_specs()
    fixed = analytic_property_record(specs["harmonic_sine_w1"])
    packet = analytic_property_record(specs["wave_packet_sine_w0p5_w1"])
    for record in (fixed, packet):
        assert record["finite_positive_center"]
        assert record["zero_central_derivative"]
        assert record["finite_total_mass"]
        assert record["nonnegative_abel_deprojection"]
        assert record["cumulative_mass"] == "numerical_positive_quadrature"
        assert record["enclosed_radii"] == "numerical_root"
        assert record["psf_convolution"] == "numerical_hankel_or_direct"
    assert fixed["integrated_density_rate"] == "elementary"
    assert fixed["projected_density"] == "elementary"
    assert packet["integrated_density_rate"] == "numerical_quadrature"
    assert packet["projected_density"] == "numerical_quadrature"


def test_aperture_radius_matches_the_measured_cog_endpoint() -> None:
    assert APERTURE_RADIUS_KPC == RADII_KPC[-1]
