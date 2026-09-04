"""Focused checks; no repository-wide collection or real target data."""

import unittest

import numpy as np

from correction import (
    apply_correction,
    fit_coefficients,
    fit_map,
    make_folds,
    split_roles,
)


class CorrectionChecks(unittest.TestCase):
    def setUp(self):
        self.radii = np.geomspace(2, 148.22, 24)
        self.baseline = 1e11 * (1 - np.exp(-self.radii / 18))

    def test_zero_is_exact(self):
        np.testing.assert_array_equal(
            apply_correction(self.baseline, np.zeros(4), self.radii), self.baseline
        )

    def test_mass_and_monotonicity(self):
        for coefficients in np.random.default_rng(1).normal(size=(30, 4)):
            prediction = apply_correction(self.baseline, coefficients, self.radii)
            self.assertTrue(np.all(np.diff(prediction) >= 0))
            self.assertAlmostEqual(
                np.log10(prediction[-1] / self.baseline[-1]), coefficients[0]
            )

    def test_synthetic_recovery(self):
        expected = np.array([0.12, 0.3, -0.2, 0.15])
        target = apply_correction(self.baseline, expected, self.radii)
        actual = fit_coefficients(self.baseline, target, self.radii)
        np.testing.assert_allclose(actual, expected, atol=1e-5)

    def test_no_stellar_input_to_map(self):
        rng = np.random.default_rng(3)
        features = rng.normal(size=(60, 5))
        targets = features[:, :4] / 8
        fitted = fit_map(features[:40], targets[:40], 1, 0.01)
        before = fitted.predict(features[40:])
        targets[40:] = np.nan
        np.testing.assert_array_equal(before, fitted.predict(features[40:]))

    def test_missing_features_use_training_imputation(self):
        features = np.arange(40, dtype=float).reshape(20, 2)
        features[2, 1] = np.nan
        fitted = fit_map(features, features[:, :1] / 10, 1, 0.1)
        self.assertTrue(np.isfinite(fitted.predict([[3, np.nan]])).all())

    def test_roles_disjoint_and_complete(self):
        folds = make_folds(np.arange(100, dtype=float))
        for rotation in range(5):
            roles = split_roles(folds, rotation)
            np.testing.assert_array_equal(
                np.sort(np.concatenate(roles)), np.arange(100)
            )
            self.assertEqual([len(x) for x in roles], [60, 20, 20])


if __name__ == "__main__":
    unittest.main()
