"""Checks that Exp69 reads only the frozen Exp67 discovery artifacts."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from exp69_run import (  # noqa: E402
    discovery_artifacts_available,
    load_discovery_bundle,
    operational_gate_indices,
)


def test_discovery_loader_requires_the_explicit_preserved_root() -> None:
    configured = os.environ.get("HONGSHAO_EXP67_ROOT")
    assert configured is not None
    assert Path(configured).name == "hongshao_exp67_squared_slope_symbolic_density"


def test_preserved_archive_matches_discovery_and_gate_indices_exactly() -> None:
    if not discovery_artifacts_available():
        pytest.skip("preserved Exp67 discovery artifacts are not available")
    bundle = load_discovery_bundle()
    assert bundle["stage"] == "discovery"
    assert bundle["truth_log"].shape == (6000, 24)
    assert len(np.unique(bundle["rows"])) == 1200
    assert np.array_equal(np.bincount(bundle["epochs"], minlength=5), [1200] * 5)
    assert np.array_equal(bundle["archive_indices"], bundle["discovery_indices"])

    gate = operational_gate_indices(bundle)
    assert gate.shape == (125,)
    assert len(np.unique(bundle["rows"][gate])) == 25
    assert np.array_equal(np.bincount(bundle["epochs"][gate], minlength=5), [25] * 5)
