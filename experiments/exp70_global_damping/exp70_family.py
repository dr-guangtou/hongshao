"""Frozen population-wide damped-cosine family grid for Exp70."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP69_DIR = HERE.parent / "exp69_damped_cosine_reparameterization"
sys.path.insert(0, str(EXP69_DIR))

from exp69_family import (  # noqa: E402
    LOG_A_BOUNDS,
    LOG_FREQUENCY_BOUNDS,
    LOG_MASS_BOUNDS,
    LOG_RADIUS_BOUNDS,
    RADII_KPC,  # noqa: F401 - re-exported as the Exp70 radius grid
    SIGNED_CONTRAST_BOUNDS,
    CandidateSpec,
)

SHARED_DAMPING_VALUES = (0.75, 1.0, 1.25, 1.5)


def _original_layout() -> tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    names = (
        "log_mstar_148",
        "log_core_radius",
        "log_a",
        "signed_contrast",
        "log_frequency",
    )
    lower = (
        LOG_MASS_BOUNDS[0],
        LOG_RADIUS_BOUNDS[0],
        LOG_A_BOUNDS[0],
        SIGNED_CONTRAST_BOUNDS[0],
        LOG_FREQUENCY_BOUNDS[0],
    )
    upper = (
        LOG_MASS_BOUNDS[1],
        LOG_RADIUS_BOUNDS[1],
        LOG_A_BOUNDS[1],
        SIGNED_CONTRAST_BOUNDS[1],
        LOG_FREQUENCY_BOUNDS[1],
    )
    return names, lower, upper


def shared_damping_specs() -> dict[str, CandidateSpec]:
    """Return the exact ordered four-value global damping grid."""
    names, lower, upper = _original_layout()
    suffixes = ("0p75", "1", "1p25", "1p5")
    return {
        f"fixed_lambda_{suffix}": CandidateSpec(
            name=f"fixed_lambda_{suffix}",
            label=rf"Shared $\lambda={damping:.2f}$",
            parameter_names=names,
            lower=lower,
            upper=upper,
            coordinate_system="fixed",
            fixed_damping=damping,
        )
        for damping, suffix in zip(SHARED_DAMPING_VALUES, suffixes, strict=True)
    }


def free_reference_spec() -> CandidateSpec:
    """Return the original six-coordinate free-damping reference."""
    names, lower, upper = _original_layout()
    return CandidateSpec(
        name="free_reference",
        label="Free per-profile damping",
        parameter_names=names + ("log_damping",),
        lower=lower + (np.log(0.05),),
        upper=upper + (np.log(3.0),),
        coordinate_system="original",
    )


def all_specs() -> dict[str, CandidateSpec]:
    """Return the free reference followed by every fixed candidate."""
    return {"free_reference": free_reference_spec(), **shared_damping_specs()}
