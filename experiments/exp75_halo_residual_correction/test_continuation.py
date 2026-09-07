"""Focused continuation checks, with synthetic inputs where possible."""

import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from continuation import (
    BASE,
    check_deadline,
    check_roles,
    feature_sample,
    scaled_nodes,
    write_record,
)


class ContinuationChecks(unittest.TestCase):
    def test_deadline_stops_science(self):
        with self.assertRaises(TimeoutError):
            check_deadline(datetime(2026, 9, 7, 22, 0, tzinfo=timezone.utc))
        check_deadline(datetime(2026, 9, 7, 21, 59, tzinfo=timezone.utc))

    def test_role_overlap_rejected(self):
        with self.assertRaises(ValueError):
            check_roles(np.arange(6), np.array([5, 6]), np.array([7, 8]), 9)
        check_roles(np.arange(5), np.array([5, 6]), np.array([7, 8]), 9)

    def test_node_scaling_does_not_change_domain(self):
        original = {"n_early": 6, "n_node": 16, "t_start": 0.1}
        scaled = scaled_nodes(original, 2)
        self.assertEqual(scaled["t_start"], original["t_start"])
        self.assertEqual(original["n_node"], 16)
        self.assertEqual(scaled["n_node"], 32)

    def test_checkpoint_never_overwrites(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            write_record(path, {"status": "complete"})
            with self.assertRaises(FileExistsError):
                write_record(path, {"status": "changed"})
            self.assertIn("complete", path.read_text())

    def test_measured_features_keep_inputs_unchanged(self):
        sample = {
            "core": np.arange(50).reshape(10, 5).astype(float),
            "history": np.arange(90).reshape(10, 9).astype(float),
            "halo": np.arange(40).reshape(10, 4).astype(float),
            "measured_epoch_mass": np.full((10, 5), 13.0),
        }
        original = {key: value.copy() for key, value in sample.items()}
        result = feature_sample(sample, "measured")
        self.assertEqual(result["core"].shape, (10, 6))
        self.assertEqual(result["history"].shape, (10, 10))
        np.testing.assert_array_equal(
            result["core"][:, :5], sample["measured_epoch_mass"]
        )
        for key in sample:
            np.testing.assert_array_equal(sample[key], original[key])

    def test_measured_feature_evaluation_label_poison(self):
        rng = np.random.default_rng(753)
        radii = np.geomspace(2, 148.22, 24)
        baseline = np.tile(1e11 * (1 - np.exp(-radii / 18)), (30, 5, 1))
        core = rng.normal(size=(30, 5))
        core[:, 0] += 13
        sample = {
            "radii": radii,
            "core": core,
            "history": np.column_stack((core, core[:, 1:])),
            "halo": core[:, :4],
            "measured_epoch_mass": 13 + rng.normal(0, 0.2, (30, 5)),
            "target": BASE.apply_correction(
                baseline, rng.normal(0, 0.1, (30, 5, 4)), radii
            ),
        }
        sample["measured_epoch_mass"][3, 2] = np.nan
        sample = feature_sample(sample, "measured")
        roles = BASE.split_roles(BASE.make_folds(core[:, 0]), 0)
        before, _ = BASE.compare(sample, baseline, *roles)
        sample["target"][roles[-1]] *= 100
        after, _ = BASE.compare(sample, baseline, *roles)
        for name in before:
            np.testing.assert_array_equal(before[name], after[name])

    def test_exact_annuli_conserve_total_and_reject_outside_grid(self):
        report = BASE.load_module(
            "exp75_annular_checks", Path(__file__).with_name("continuation_report.py")
        )
        radii = np.geomspace(2, 148.22, 24)
        curves = np.tile(1e11 * (1 - np.exp(-radii / 18)), (2, 5, 1))
        np.testing.assert_allclose(
            report.annuli(curves, radii).sum(-1), curves[..., -1], rtol=1e-14
        )
        with self.assertRaises(ValueError):
            report.aperture(curves, radii, 150)

    def test_future_growth_recovers_known_conditional_slope(self):
        report = BASE.load_module(
            "exp75_growth_checks", Path(__file__).with_name("continuation_report.py")
        )
        rng = np.random.default_rng(758)
        mass = 13 + rng.normal(0, 0.3, (30, 5))
        growth = mass[:, :1] - mass
        radii = np.geomspace(2, 148.22, 24)
        truth = 10 ** (11 + 0.2 * (mass - 13) + 0.3 * growth)[..., None] * (
            1 - np.exp(-radii / 18)
        )
        sample = {"target": truth, "radii": radii, "measured_epoch_mass": mass}
        result = report.future_growth(
            sample, {"test": truth * 10 ** (0.1 * growth[..., None])}
        )
        for entry in result.values():
            self.assertAlmostEqual(
                entry["models"]["data"]["partial_slope_dex_per_dex"], 0.3
            )
            self.assertAlmostEqual(
                entry["models"]["test"]["residual_slope_model_minus_data"], 0.1
            )


if __name__ == "__main__":
    unittest.main()
