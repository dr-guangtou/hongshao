"""Nested-fold symbolic regression for the exp50 quantile-coordinate map.

PySR sees only residuals of the complete linear DiffMAH+c200c mean. The full
linear core is retained, so the comparison measures the incremental value of
the discovered nonlinear terms rather than PySR's variable pruning.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from hongshao.emulator import _chol_psd, _fit_logvar  # noqa: E402
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402
from model import compress_cogs, reconstruct_cogs, target_names  # noqa: E402
from run import (  # noqa: E402
    FEATURE_NAMES,
    N_FOLD,
    RADII,
    cv_quantile,
    folds_of,
    load_sample,
    summarize_model,
)

set_style()

OUTDIR = HERE / "outputs"
SYMBOLIC_OUTDIR = OUTDIR / "symbolic"
FIGDIR = HERE / "figures"
N_MAX = int(os.environ.get("EXP50_SR_NMAX", 0))
N_ITERATION = int(os.environ.get("EXP50_SR_NITER", 50))
MAX_SIZE = int(os.environ.get("EXP50_SR_MAXSIZE", 15))
N_DRAW = int(os.environ.get("EXP50_SR_NDRAW", 32))
VARIABLES = sp.symbols("x0:5")
VARIABLE_LOCALS = {f"x{index}": variable for index, variable in enumerate(VARIABLES)}
LINEAR_MONOMIALS = [
    tuple(int(index == column) for index in range(5)) for column in range(5)
]


def new_symbolic_regressor(seed: int):
    from pysr import PySRRegressor

    return PySRRegressor(
        niterations=N_ITERATION,
        maxsize=MAX_SIZE,
        binary_operators=["+", "-", "*"],
        unary_operators=["square"],
        nested_constraints={"square": {"square": 0}},
        model_selection="best",
        elementwise_loss="L2DistLoss()",
        progress=False,
        verbosity=0,
        deterministic=True,
        parallelism="serial",
        random_state=seed,
        output_directory=tempfile.mkdtemp(prefix="pysr_exp50_"),
    )


def nonlinear_monomials(expression) -> list[tuple[int, ...]]:
    polynomial = sp.Poly(sp.expand(expression), *VARIABLES)
    return sorted(
        monomial
        for monomial in polynomial.monoms()
        if sum(monomial) >= 2 and sum(monomial) <= 3
    )


def monomial_design(
    standardized: np.ndarray, monomials: list[tuple[int, ...]]
) -> np.ndarray:
    columns = []
    for monomial in monomials:
        value = np.ones(len(standardized))
        for feature_index, exponent in enumerate(monomial):
            value *= standardized[:, feature_index] ** exponent
        columns.append(value)
    return np.column_stack(columns) if columns else np.empty((len(standardized), 0))


def monomial_label(monomial: tuple[int, ...]) -> str:
    parts = []
    for feature, exponent in zip(FEATURE_NAMES, monomial):
        if exponent == 1:
            parts.append(feature)
        elif exponent > 1:
            parts.append(f"{feature}^{exponent}")
    return "*".join(parts)


def standardized_monomial_label(monomial: tuple[int, ...]) -> str:
    parts = []
    for feature, exponent in zip(FEATURE_NAMES, monomial):
        if exponent == 1:
            parts.append(f"z_{feature}")
        elif exponent > 1:
            parts.append(f"z_{feature}^{exponent}")
    return "*".join(parts)


def term_tex(label: str) -> str:
    """Math-safe display form for a monomial label."""
    parts = []
    for part in label.replace(" (diagnostic)", "").split("*"):
        base, _, exponent = part.partition("^")
        base = {"c200c": "c_{200c}"}.get(base, base.replace("_", r"\_"))
        parts.append(base if not exponent else f"{base}^{{{exponent}}}")
    suffix = r"\ \mathrm{(diagnostic)}" if "diagnostic" in label else ""
    return "$" + r"\,".join(parts) + suffix + "$"


def nested_symbolic(
    features: np.ndarray, targets: np.ndarray
) -> tuple[dict, list[dict], dict[str, Counter]]:
    """Discover nonlinear terms within each training fold and predict held-out data."""
    n_galaxy, n_target = targets.shape
    mean_parameters = np.empty_like(targets)
    sigma_parameters = np.empty_like(targets)
    mean_cogs = np.empty((n_galaxy, len(RADII)))
    draw_cogs = np.empty((N_DRAW, n_galaxy, len(RADII)))
    fold_id = np.empty(n_galaxy, dtype=int)
    mean_dropped = np.empty(n_galaxy, dtype=int)
    draw_dropped = np.empty((N_DRAW, n_galaxy), dtype=int)
    records: list[dict] = []
    occurrence = {name: Counter() for name in target_names("quantile3")}

    for fold_number, fold in enumerate(folds_of(n_galaxy)):
        train = np.setdiff1d(np.arange(n_galaxy), fold)
        feature_mean = features[train].mean(axis=0)
        feature_std = features[train].std(axis=0)
        train_standardized = (features[train] - feature_mean) / feature_std
        test_standardized = (features[fold] - feature_mean) / feature_std
        train_linear = np.column_stack([np.ones(len(train)), train_standardized])
        test_linear = np.column_stack([np.ones(len(fold)), test_standardized])
        train_residuals = np.empty((len(train), n_target))
        test_sigmas = np.empty((len(fold), n_target))
        selected_by_target = []

        for target_index, name in enumerate(target_names("quantile3")):
            linear_beta, *_ = np.linalg.lstsq(
                train_linear, targets[train, target_index], rcond=None
            )
            linear_residual = targets[train, target_index] - train_linear @ linear_beta
            # Hold the search seed fixed so recurrence differences reflect the
            # training galaxies, not different stochastic searches.
            regressor = new_symbolic_regressor(seed=0)
            regressor.fit(
                train_standardized,
                linear_residual,
                variable_names=[f"x{index}" for index in range(5)],
            )
            expression = sp.sympify(str(regressor.sympy()), locals=VARIABLE_LOCALS)
            monomials = nonlinear_monomials(expression)
            selected_by_target.append(monomials)
            for monomial in monomials:
                occurrence[name][monomial_label(monomial)] += 1

            train_design = np.column_stack(
                [train_linear, monomial_design(train_standardized, monomials)]
            )
            test_design = np.column_stack(
                [test_linear, monomial_design(test_standardized, monomials)]
            )
            augmented_beta, *_ = np.linalg.lstsq(
                train_design, targets[train, target_index], rcond=None
            )
            mean_parameters[fold, target_index] = test_design @ augmented_beta
            train_residuals[:, target_index] = (
                targets[train, target_index] - train_design @ augmented_beta
            )
            gamma = _fit_logvar(
                train_residuals[:, target_index], train_standardized, ridge=2.0
            )
            variance_design = np.column_stack([np.ones(len(fold)), test_standardized])
            test_sigmas[:, target_index] = np.exp(0.5 * variance_design @ gamma)
            sigma_parameters[fold, target_index] = test_sigmas[:, target_index]
            records.append(
                {
                    "fold": fold_number,
                    "target": name,
                    "expression": str(sp.expand(expression)),
                    "terms": ";".join(monomial_label(value) for value in monomials)
                    or "none",
                }
            )

        train_sigmas = np.empty_like(train_residuals)
        for target_index in range(n_target):
            gamma = _fit_logvar(
                train_residuals[:, target_index], train_standardized, ridge=2.0
            )
            variance_design = np.column_stack([np.ones(len(train)), train_standardized])
            train_sigmas[:, target_index] = np.exp(0.5 * variance_design @ gamma)
        correlation = np.corrcoef((train_residuals / train_sigmas).T)
        normal = np.random.default_rng(3000 + fold_number).standard_normal(
            (N_DRAW, len(fold), n_target)
        )
        parameter_draws = mean_parameters[fold][None] + test_sigmas[None] * (
            normal @ _chol_psd(correlation).T
        )
        mean_cogs[fold], mean_dropped[fold] = reconstruct_cogs(
            mean_parameters[fold], RADII, "quantile3", True
        )
        decoded, dropped = reconstruct_cogs(
            parameter_draws.reshape(-1, n_target), RADII, "quantile3", True
        )
        draw_cogs[:, fold] = decoded.reshape(N_DRAW, len(fold), -1)
        draw_dropped[:, fold] = dropped.reshape(N_DRAW, len(fold))
        fold_id[fold] = fold_number
        print(f"  completed symbolic fold {fold_number + 1}/{N_FOLD}")

    result = {
        "mean_parameters": mean_parameters,
        "sigma_parameters": sigma_parameters,
        "mean_cogs": mean_cogs,
        "draw_cogs": draw_cogs,
        "fold_id": fold_id,
        "mean_dropped": mean_dropped,
        "draw_dropped": draw_dropped,
    }
    return result, records, occurrence


def stable_equations(
    features: np.ndarray, targets: np.ndarray, occurrence: dict[str, Counter]
) -> list[dict]:
    """Refit the terms recurring in at least four folds on the full sample."""
    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0)
    standardized = (features - feature_mean) / feature_std
    linear = np.column_stack([np.ones(len(features)), standardized])
    equations = []
    for target_index, name in enumerate(target_names("quantile3")):
        stable_labels = sorted(
            label for label, count in occurrence[name].items() if count >= 4
        )
        label_to_monomial = {}
        for powers in np.ndindex(*(4,) * 5):
            if 2 <= sum(powers) <= 3:
                label_to_monomial[monomial_label(powers)] = powers
        stable_monomials = [label_to_monomial[label] for label in stable_labels]
        design = np.column_stack(
            [linear, monomial_design(standardized, stable_monomials)]
        )
        beta, *_ = np.linalg.lstsq(design, targets[:, target_index], rcond=None)
        terms = [f"{beta[0]:+.6g}"]
        terms.extend(
            f"{coefficient:+.6g}*z_{feature}"
            for coefficient, feature in zip(beta[1:6], FEATURE_NAMES)
        )
        terms.extend(
            f"{coefficient:+.6g}*{standardized_monomial_label(monomial)}"
            for coefficient, monomial in zip(beta[6:], stable_monomials)
        )
        equations.append(
            {
                "target": name,
                "stable_terms": ";".join(stable_labels) or "none",
                "equation_standardized_features": " ".join(terms),
            }
        )
    return equations


def symbolic_figure(summaries: dict[str, dict], occurrence: dict[str, Counter]) -> None:
    model_names = ["linear", "symbolic", "poly2"]
    colors = [OKABE_ITO[7], OKABE_ITO[2], OKABE_ITO[4]]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9))
    axes[0].bar(
        model_names,
        [summaries[name]["profile_crps_dex"] for name in model_names],
        color=colors,
    )
    axes[0].set(ylabel="Profile CRPS (dex)", title="End-to-end predictive distribution")
    for name, color in zip(model_names, colors):
        axes[1].plot(
            RADII, summaries[name]["profile_crps_by_radius"], color=color, label=name
        )
    axes[1].set(
        xscale="log",
        xlabel="Radius (kpc)",
        ylabel="CRPS (dex)",
        title="Radial predictive score",
    )
    axes[1].legend()

    all_terms = sorted({term for counter in occurrence.values() for term in counter})
    matrix = np.array(
        [
            [occurrence[target][term] for term in all_terms]
            for target in target_names("quantile3")
        ]
    )
    image = axes[2].imshow(matrix, cmap="cividis", vmin=0, vmax=N_FOLD, aspect="auto")
    axes[2].set_xticks(
        np.arange(len(all_terms)),
        [term_tex(term) for term in all_terms],
        rotation=45,
        ha="right",
    )
    axes[2].set_yticks(
        np.arange(len(target_names("quantile3"))), target_names("quantile3")
    )
    axes[2].set_title("Term recurrence across five folds")
    fig.colorbar(image, ax=axes[2], label="Number of folds")
    for label, axis in zip("ABC", axes):
        axis.text(
            -0.16, 1.05, label, transform=axis.transAxes, fontsize=11, fontweight="bold"
        )
    fig.suptitle("exp50 — symbolic corrections to the complete linear halo map")
    fig.tight_layout()
    save_fig(fig, FIGDIR / "exp50_symbolic_regression")
    plt.close(fig)


def residual_figure(
    features: np.ndarray, targets: np.ndarray, occurrence: dict[str, Counter]
) -> None:
    standardized = (features - features.mean(axis=0)) / features.std(axis=0)
    linear = np.column_stack([np.ones(len(features)), standardized])
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.0))
    for target_index, (axis, name) in enumerate(
        zip(axes.ravel(), target_names("quantile3"))
    ):
        beta, *_ = np.linalg.lstsq(linear, targets[:, target_index], rcond=None)
        residual = targets[:, target_index] - linear @ beta
        stable = [
            (term, count) for term, count in occurrence[name].items() if count >= 4
        ]
        if stable:
            term = max(stable, key=lambda item: item[1])[0]
            powers = next(
                powers
                for powers in np.ndindex(*(4,) * 5)
                if 2 <= sum(powers) <= 3 and monomial_label(powers) == term
            )
            x_value = monomial_design(standardized, [powers])[:, 0]
        else:
            term = "late^2 (diagnostic)"
            x_value = standardized[:, 3] ** 2
        edges = np.quantile(x_value, np.linspace(0.0, 1.0, 13))
        centers, medians, errors = [], [], []
        for lower, upper in zip(edges[:-1], edges[1:]):
            selected = (x_value >= lower) & (x_value <= upper)
            centers.append(np.median(x_value[selected]))
            medians.append(np.median(residual[selected]))
            errors.append(np.std(residual[selected]) / np.sqrt(selected.sum()))
        axis.axhline(0.0, color="0.6", linestyle=":")
        axis.errorbar(centers, medians, yerr=errors, fmt="o-", color=OKABE_ITO[2])
        axis.set(xlabel=term_tex(term), ylabel="Linear-model residual", title=name)
    axes.ravel()[-1].axis("off")
    for label, axis in zip("ABCDEF", axes.ravel()):
        axis.text(
            -0.14, 1.04, label, transform=axis.transAxes, fontsize=11, fontweight="bold"
        )
    fig.suptitle("exp50 — visual check of recurrent symbolic structure")
    fig.tight_layout()
    save_fig(fig, FIGDIR / "exp50_symbolic_residuals")
    plt.close(fig)


def run() -> None:
    features, truth_log, _ = load_sample(N_MAX)
    targets = compress_cogs(truth_log, RADII, "quantile3")
    print(f"exp50 symbolic regression: n={len(features)}, niteration={N_ITERATION}")
    symbolic_result, records, occurrence = nested_symbolic(features, targets)
    linear_result = cv_quantile(
        features, targets, "quantile3", mean="linear", n_draw=N_DRAW
    )
    poly2_result = cv_quantile(
        features, targets, "quantile3", mean="poly2", n_draw=N_DRAW
    )
    results = {
        "linear": linear_result,
        "symbolic": symbolic_result,
        "poly2": poly2_result,
    }
    summaries = {
        name: summarize_model(name, result, truth_log)
        for name, result in results.items()
    }
    equations = stable_equations(features, targets, occurrence)

    print("\n[end-to-end symbolic comparison]")
    for name in ("linear", "symbolic", "poly2"):
        print(
            f"  {name:8s}: profile CRPS {summaries[name]['profile_crps_dex']:.5f} dex, "
            f"median CoG RMS {summaries[name]['median_profile_rms_dex']:.5f} dex"
        )
    print("\n[terms recurring in at least four of five folds]")
    for equation in equations:
        print(f"  {equation['target']:20s}: {equation['stable_terms']}")

    SYMBOLIC_OUTDIR.mkdir(parents=True, exist_ok=True)
    with (SYMBOLIC_OUTDIR / "fold_terms.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["fold", "target", "expression", "terms"]
        )
        writer.writeheader()
        writer.writerows(records)
    with (SYMBOLIC_OUTDIR / "equations.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["target", "stable_terms", "equation_standardized_features"],
        )
        writer.writeheader()
        writer.writerows(equations)
    payload = {
        "n_galaxy": len(features),
        "n_iteration": N_ITERATION,
        "models": summaries,
        "term_occurrence": {
            target: dict(counter) for target, counter in occurrence.items()
        },
        "stable_equations": equations,
        "feature_standardization": {
            "names": FEATURE_NAMES,
            "mean": features.mean(axis=0).tolist(),
            "standard_deviation": features.std(axis=0).tolist(),
        },
    }
    (SYMBOLIC_OUTDIR / "results.json").write_text(json.dumps(payload, indent=2))
    write_manifest(
        SYMBOLIC_OUTDIR,
        {
            "symbolic_n_galaxy": len(features),
            "symbolic_n_iteration": N_ITERATION,
            "symbolic_profile_crps_dex": summaries["symbolic"]["profile_crps_dex"],
        },
    )
    symbolic_figure(summaries, occurrence)
    residual_figure(features, targets, occurrence)
    print(f"\nwrote symbolic figures to {FIGDIR}")
    print(f"wrote symbolic outputs to {SYMBOLIC_OUTDIR}")


if __name__ == "__main__":
    run()
