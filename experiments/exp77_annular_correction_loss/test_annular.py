"""Known-answer and held-out-label checks before the new annular fits."""

import unittest

from annular import (
    annular_matrix,
    apply_correction,
    fit_labels,
    predict_from_calibration,
)
import numpy as np


class AnnularChecks(unittest.TestCase):
    def setUp(self):
        self.radii = np.geomspace(2, 148.22, 24)
        self.baseline = 1e11 * (1 - np.exp(-self.radii / 18))

    def test_annulus_conservation_and_zero(self):
        np.testing.assert_allclose(
            (self.baseline @ annular_matrix(self.radii)).sum(),
            self.baseline[-1],
            rtol=1e-14,
        )
        np.testing.assert_array_equal(
            apply_correction(self.baseline, np.zeros(4), self.radii), self.baseline
        )

    def test_synthetic_recovery_all_weights(self):
        coefficients = np.array([0.12, -0.3, 0.2, 0.1])
        truth = apply_correction(self.baseline, coefficients, self.radii)
        for weight in (0, 0.25, 1):
            fitted, _ = fit_labels(self.baseline, truth, self.radii, weight)
            np.testing.assert_allclose(fitted, coefficients, atol=1e-5)
            prediction = apply_correction(self.baseline, fitted, self.radii)
            self.assertTrue(np.all(np.diff(prediction) >= 0))

    def test_evaluation_labels_never_change_predictions(self):
        rng = np.random.default_rng(77)
        baseline = np.tile(self.baseline, (12, 5, 1))
        sample = {
            "radii": self.radii,
            "core": rng.normal(size=(12, 6)),
            "target": apply_correction(
                baseline, rng.normal(0, 0.05, (12, 5, 4)), self.radii
            ),
        }
        sample["core"][0, 2] = np.nan
        calibration, held = np.arange(8), np.arange(8, 12)
        choice = {"feature": "core", "degree": 1, "penalty": 1}
        before, selected_before = predict_from_calibration(
            sample, baseline, calibration, held, choice
        )
        sample["target"][held] *= 100
        after, selected_after = predict_from_calibration(
            sample, baseline, calibration, held, choice
        )
        self.assertEqual(selected_before, selected_after)
        for name in before:
            np.testing.assert_array_equal(before[name], after[name])


if __name__ == "__main__":
    unittest.main()
