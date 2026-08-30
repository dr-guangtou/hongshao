"""Focused family and mechanics checks for Exp70."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from exp70_family import (  # noqa: E402
    RADII_KPC,
    free_reference_spec,
    shared_damping_specs,
)
from exp69_family import (  # noqa: E402
    cog_log,
    fit_profile,
    physical_descriptors,
    synthetic_parameters,
)


def test_shared_damping_grid_is_exactly_the_predeclared_grid() -> None:
    specs = shared_damping_specs()
    assert tuple(specs) == (
        "fixed_lambda_0p75",
        "fixed_lambda_1",
        "fixed_lambda_1p25",
        "fixed_lambda_1p5",
    )
    assert [spec.fixed_damping for spec in specs.values()] == [0.75, 1.0, 1.25, 1.5]
    assert all(spec.n_parameter == 5 for spec in specs.values())
    assert free_reference_spec().n_parameter == 6


def test_fixed_and_free_coordinates_decode_to_the_same_profile() -> None:
    free = free_reference_spec()
    for case_index, fixed in enumerate(shared_damping_specs().values()):
        fixed_parameters = synthetic_parameters(fixed, case_index)
        free_parameters = np.concatenate(
            (fixed_parameters, [np.log(float(fixed.fixed_damping))])
        )
        fixed_cog = cog_log(fixed, fixed_parameters, RADII_KPC, grid_size=1024)
        free_cog = cog_log(free, free_parameters, RADII_KPC, grid_size=1024)
        assert np.max(np.abs(fixed_cog - free_cog)) < 2.0e-12


def test_every_fixed_value_recovers_noiseless_profiles() -> None:
    for case_index in range(4):
        for name, spec in {
            "free_reference": free_reference_spec(),
            **shared_damping_specs(),
        }.items():
            parameters = synthetic_parameters(spec, case_index)
            truth = cog_log(spec, parameters, RADII_KPC, grid_size=1024)
            result = fit_profile(
                spec,
                RADII_KPC,
                truth,
                n_starts=4,
                max_nfev=600,
                grid_size=512,
                synthetic_truth_parameters=parameters,
            )
            truth_descriptor = physical_descriptors(spec, parameters)
            fitted_descriptor = physical_descriptors(spec, result["parameters"])
            assert result["cog_rms_dex"] < 3.0e-5, (name, case_index)
            assert np.all(np.isfinite(fitted_descriptor - truth_descriptor))
