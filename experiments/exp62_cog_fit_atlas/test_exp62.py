"""Mechanical checks for Exp62's time and halo-association calculations."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from families import cog_log, synthetic_parameters  # noqa: E402
from hongshao import qa  # noqa: E402
from mah_diversity import half_mass_times, normalized_mahs  # noqa: E402
from run import RADII  # noqa: E402
from shape_diversity import normalized_cogs  # noqa: E402
from stage3 import _polynomial_prediction, decoded_observables  # noqa: E402
from stage4 import (  # noqa: E402
    cross_fitted_association,
    galaxy_folds,
    mass_conditioned_shuffle,
)
from stage5 import (  # noqa: E402
    _validation_mask,
    CANDIDATE_SPECS,
    candidate_cog_log,
    candidate_synthetic_parameters,
    fit_candidate,
)
from stage6 import (  # noqa: E402
    _cog_error_tail_by_epoch,
    _predictor_design_condition_numbers,
    align_cubic_parameters,
    cross_fitted_shape_relation,
    decode_cubic_shape_predictions,
)
from stage7 import (  # noqa: E402
    coarse_annular_fractions,
    cross_fitted_profile_relation,
    diversity_r2,
)
from visual_record import _mass_residual  # noqa: E402


def test_quadratic_time_prediction_recovers_an_omitted_epoch() -> None:
    times = np.asarray([3.3, 4.3, 5.8, 7.3, 9.3])
    coefficients = np.asarray(
        [
            [[1.0, -0.3], [0.2, 0.5], [-0.01, 0.04]],
            [[0.5, 0.7], [-0.1, 0.2], [0.03, -0.02]],
        ]
    )
    design = np.vander(times, 3, increasing=True)
    values = np.einsum("ed,gdt->get", design, coefficients)
    for held in (1, 2, 3):
        actual = _polynomial_prediction(values, times, held, degree=2)
        assert np.allclose(actual, values[:, held], atol=1.0e-12)


def test_decoded_observables_are_finite_for_every_advanced_role() -> None:
    families = ("gompertz_log", "sersic1_moffat1p4", "double_sersic_free")
    profiles = np.asarray(
        [cog_log(family, synthetic_parameters(family), RADII) for family in families]
    )
    decoded = decoded_observables(profiles)
    assert decoded.shape == (len(families), 17)
    assert np.isfinite(decoded).all()
    assert np.all(decoded[:, 2] < decoded[:, 3])
    assert np.all(decoded[:, 3] < decoded[:, 4])
    assert np.all(decoded[:, 4] < decoded[:, 5])


def test_cross_fitted_association_recovers_mass_conditioned_signal() -> None:
    rng = np.random.default_rng(6104)
    n_galaxy = 600
    mass = rng.uniform(12.0, 15.0, n_galaxy)
    residual_features = rng.normal(size=(n_galaxy, 5))
    features = residual_features + 0.4 * (mass - 13.5)[:, None]
    target = (
        10.0
        + 0.6 * (mass - 13.5)
        + 0.15 * (mass - 13.5) ** 2
        + 0.7 * residual_features[:, 0]
        - 0.4 * residual_features[:, 2]
    )[:, None]
    result = cross_fitted_association(
        target,
        mass,
        features,
        galaxy_folds(mass[:, None]),
    )
    assert result["full_rmse"][0] < 0.05 * result["mass_rmse"][0]
    assert result["conditional_beta_standardized"][0, 0] > 0.5
    assert result["conditional_beta_standardized"][2, 0] < -0.3


def test_mass_conditioned_block_shuffle_preserves_internal_pairs() -> None:
    rng = np.random.default_rng(61)
    mass = np.linspace(12.0, 15.0, 100)
    first = np.arange(100, dtype=float)
    features = np.column_stack([first, 3.0 * first + 7.0, rng.normal(size=100)])
    shuffled = mass_conditioned_shuffle(features, mass, (0, 1), rng)
    assert np.allclose(shuffled[:, 1], 3.0 * shuffled[:, 0] + 7.0)
    assert np.array_equal(shuffled[:, 2], features[:, 2])
    assert not np.array_equal(shuffled[:, 0], features[:, 0])


def test_visual_scaling_residual_removes_quadratic_mass_trend() -> None:
    mass = np.linspace(12.0, 15.0, 200)
    values = np.column_stack(
        [
            2.0 + 0.5 * mass - 0.03 * mass**2,
            -1.0 + 0.2 * mass + 0.04 * mass**2,
        ]
    )
    residual = _mass_residual(values, mass)
    assert np.max(np.abs(residual)) < 1.0e-12


def test_cog_normalization_removes_mass_and_size() -> None:
    first_fraction = np.linspace(0.1, 1.0, len(RADII))
    second_fraction = np.interp(RADII / 2.0, RADII, first_fraction)
    cogs_log = np.log10(np.vstack([1.0e11 * first_fraction, 3.0e11 * second_fraction]))
    r50 = np.asarray(
        [
            qa.enclosed_radius(10.0 ** cogs_log[0], RADII, 0.5),
            qa.enclosed_radius(10.0 ** cogs_log[1], RADII, 0.5),
        ]
    )
    profiles = normalized_cogs(cogs_log, r50, np.asarray([0.8, 1.0, 1.2]))
    assert np.allclose(profiles[:, 1], np.log10(0.5))


def test_diffmah_normalization_removes_endpoint_mass_and_half_mass_time() -> None:
    parameters = np.asarray(
        [
            [13.5, np.log10(2.5), 3.0, 0.7],
            [14.2, np.log10(3.5), 2.2, 0.4],
        ]
    )
    endpoint_time_gyr = 9.38911025
    half_mass_time_gyr = half_mass_times(parameters, endpoint_time_gyr)
    histories = normalized_mahs(
        parameters,
        endpoint_time_gyr,
        half_mass_time_gyr,
        np.asarray([0.8, 1.0, 1.2]),
    )
    assert np.all(np.isfinite(half_mass_time_gyr))
    assert np.allclose(histories[:, 1], -np.log10(2.0), atol=1.0e-12)


def test_stage5_candidates_are_monotonic_normalized_and_recoverable() -> None:
    for family in CANDIDATE_SPECS:
        parameters = candidate_synthetic_parameters(family)
        profile = candidate_cog_log(family, parameters, RADII)
        assert np.isfinite(profile).all(), family
        assert np.all(np.diff(profile) > 0.0), family
        assert abs(profile[-1] - parameters[0]) < 1.0e-12, family
        result = fit_candidate(family, RADII, profile)
        assert result["cog_rms_dex"] < 2.0e-5, family


def test_stage5_candidates_have_declared_complexity() -> None:
    assert CANDIDATE_SPECS["burr4"].n_parameter == 4
    assert CANDIDATE_SPECS["beta_prime5"].n_parameter == 5
    assert CANDIDATE_SPECS["logit_warp4"].n_parameter == 4
    assert CANDIDATE_SPECS["logit_warp5"].n_parameter == 5
    assert CANDIDATE_SPECS["logit_cubic4"].n_parameter == 4
    assert CANDIDATE_SPECS["logit_cubic5"].n_parameter == 5
    assert CANDIDATE_SPECS["logit_cubic5_absolute"].n_parameter == 5
    assert CANDIDATE_SPECS["logit_cubic5_hybrid0p25"].n_parameter == 5


def test_stage5_validation_excludes_only_development_profile_pairs() -> None:
    rows = np.asarray([10, 10, 11, 12])
    epochs = np.asarray([0, 1, 0, 0])
    mask = _validation_mask(
        rows,
        epochs,
        development_rows=np.asarray([10, 12]),
        development_epochs=np.asarray([1, 0]),
    )
    assert np.array_equal(mask, np.asarray([True, False, True, False]))


def test_stage6_cog_error_tail_is_computed_per_galaxy_and_epoch() -> None:
    truth = np.zeros((4, 2, 3))
    prediction = np.zeros_like(truth)
    prediction[:, 0] = np.asarray([0.0, 0.1, 0.2, 0.3])[:, None]
    prediction[:, 1] = np.asarray([0.0, 0.0, 0.0, 0.4])[:, None]
    result = _cog_error_tail_by_epoch(prediction, truth)
    assert np.allclose(result["maximum_cog_rms_dex"], [0.3, 0.4])
    assert np.allclose(result["fraction_above_0p2_dex"], [0.25, 0.25])


def test_stage6_predictor_conditioning_is_finite_for_independent_features() -> None:
    rng = np.random.default_rng(6206)
    mass = rng.normal(size=(100, 2))
    features = rng.normal(size=(100, 2, 3))
    folds = np.arange(100) % 5
    condition = _predictor_design_condition_numbers(mass, features, folds)
    assert condition.shape == (2, 5)
    assert np.all(np.isfinite(condition))
    assert np.all(condition < 2.0)


def test_stage6_parameter_alignment_preserves_galaxy_epoch_pairs() -> None:
    stage_rows = np.asarray([4, 4, 8, 8])
    stage_epochs = np.asarray([0, 1, 0, 1])
    stage_parameters = np.arange(20, dtype=float).reshape(4, 5)
    aligned = align_cubic_parameters(
        stage_rows,
        stage_epochs,
        stage_parameters,
        galaxy_ids=np.asarray([8, 4]),
        n_epoch=2,
    )
    assert np.array_equal(aligned[0, 0], stage_parameters[2])
    assert np.array_equal(aligned[0, 1], stage_parameters[3])
    assert np.array_equal(aligned[1, 0], stage_parameters[0])
    assert np.array_equal(aligned[1, 1], stage_parameters[1])


def test_stage6_additive_quadratic_relation_recovers_nonlinear_signal() -> None:
    rng = np.random.default_rng(6206)
    n_galaxy = 600
    mass = rng.uniform(12.0, 15.0, n_galaxy)
    features = rng.normal(size=(n_galaxy, 2))
    targets = (
        0.4 * (mass - 13.5) + 0.7 * features[:, 0] - 0.9 * (features[:, 1] ** 2 - 1.0)
    )[:, None]
    folds = np.arange(n_galaxy) % 5
    linear = cross_fitted_shape_relation(targets, mass, features, folds, degree=1)
    quadratic = cross_fitted_shape_relation(targets, mass, features, folds, degree=2)
    assert quadratic["full_rmse"][0] < 0.05 * linear["full_rmse"][0]
    assert quadratic["residual_r2"][0] > 0.99


def test_stage6_decoder_holds_mass_and_monotonicity() -> None:
    shape_parameters = np.asarray([candidate_synthetic_parameters("logit_cubic5")[1:]])
    profiles, clipped = decode_cubic_shape_predictions(
        np.asarray([11.4]), shape_parameters
    )
    assert not clipped[0]
    assert np.isclose(profiles[0, -1], 11.4)
    assert np.all(np.diff(profiles[0]) > 0.0)


def test_stage7_diversity_r2_weights_inner_and_outer_equally() -> None:
    normalized_radius = np.asarray([0.7, 0.85, 1.0, 1.5, 3.0])
    truth = np.zeros((8, len(normalized_radius)))
    baseline = np.tile(np.asarray([0.2, 0.2, 0.0, 0.1, 0.1]), (8, 1))
    complete = 0.5 * baseline
    result = diversity_r2(truth, baseline, complete, normalized_radius)
    assert np.allclose(result, [0.75, 0.75, 0.75])


def test_stage7_direct_and_fold_local_pca_recover_rank_two_signal() -> None:
    rng = np.random.default_rng(6207)
    n_galaxy = 500
    mass = rng.uniform(12.0, 15.0, n_galaxy)
    features = rng.normal(size=(n_galaxy, 2))
    radial_modes = np.asarray(
        [
            [-0.5, -0.2, 0.0, 0.1, 0.2],
            [0.2, -0.1, 0.0, 0.3, -0.2],
        ]
    )
    targets = (
        0.03 * (mass - 13.5)[:, None] * radial_modes[:1]
        + features[:, :1] * radial_modes[:1]
        + features[:, 1:] * radial_modes[1:]
    )
    folds = np.arange(n_galaxy) % 5
    direct = cross_fitted_profile_relation(
        targets, mass, None, features, folds, degree=1
    )
    pca = cross_fitted_profile_relation(
        targets, mass, None, features, folds, degree=1, n_pca=2
    )
    direct_rms = np.sqrt(np.mean((direct["full_prediction"] - targets) ** 2))
    pca_rms = np.sqrt(np.mean((pca["full_prediction"] - targets) ** 2))
    assert direct_rms < 1.0e-10
    assert pca_rms < 1.0e-10


def test_stage7_coarse_annuli_are_independent_cumulative_differences() -> None:
    normalized_radius = np.asarray([0.7, 0.85, 1.0, 1.5, 3.0])
    cumulative_fraction = np.asarray([[0.2, 0.35, 0.5, 0.7, 0.9]])
    annuli = coarse_annular_fractions(
        np.log10(cumulative_fraction), normalized_radius, normalized_radius
    )
    assert np.allclose(annuli, [[0.15, 0.15, 0.2, 0.2]])
