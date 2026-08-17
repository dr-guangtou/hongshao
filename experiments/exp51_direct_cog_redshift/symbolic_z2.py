"""exp51 stage 5 — portable z=2 symbolic corrections.

PySR sees only residuals of the complete linear relation using DiffMAH
parameters fitted to t <= t(z=2) and c200c measured at z=2. The full linear
core is retained in every candidate.
"""

from __future__ import annotations

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
from epoch_local import fit_epoch_local_features  # noqa: E402
from run import (  # noqa: E402
    EXP51_MODEL,
    RADII,
    compress_cogs,
    cv_quantile,
    folds_of,
    reconstruct_cogs,
    summarize_model,
)

target_names = EXP51_MODEL.target_names

set_style()

FIGDIR = HERE / "figures"
OUTDIR = HERE / "outputs" / "symbolic_z2"
FEATURE_NAMES = ["logmp", "logtc", "early", "late", "c200c"]
VARIABLES = sp.symbols("x0:5")
VARIABLE_LOCALS = {f"x{index}": variable for index, variable in enumerate(VARIABLES)}
N_MAX = int(os.environ.get("EXP51_SYMBOLIC_NMAX", 0))
N_ITERATION = int(os.environ.get("EXP51_SYMBOLIC_NITER", 50))
MAX_SIZE = int(os.environ.get("EXP51_SYMBOLIC_MAXSIZE", 15))
N_DRAW = int(os.environ.get("EXP51_SYMBOLIC_NDRAW", 32))
EXP50_TERMS = {
    0: [(0, 0, 0, 0, 2)],
    1: [(0, 1, 1, 0, 0)],
    2: [],
    3: [(1, 0, 1, 0, 0)],
    4: [],
}


def load_z2_sample():
    data = fit_epoch_local_features()
    features = np.column_stack(
        [np.asarray(data["parameters"])[:, 4], np.asarray(data["concentration"])[:, 4]]
    )
    cogs_log = np.asarray(data["cogs_log"])[:, 4]
    indices = np.asarray(data["indices"])
    if N_MAX and N_MAX < len(features):
        order = np.argsort(features[:, 0])
        positions = np.linspace(0, len(features) - 1, N_MAX).round().astype(int)
        selected = order[positions]
        features, cogs_log, indices = (
            features[selected],
            cogs_log[selected],
            indices[selected],
        )
    return features, cogs_log, indices


def new_symbolic_regressor():
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
        random_state=0,
        output_directory=tempfile.mkdtemp(prefix="pysr_exp51_z2_"),
    )


def nonlinear_monomials(expression) -> list[tuple[int, ...]]:
    polynomial = sp.Poly(sp.expand(expression), *VARIABLES)
    return sorted(
        monomial for monomial in polynomial.monoms() if 2 <= sum(monomial) <= 3
    )


def monomial_design(
    standardized: np.ndarray, monomials: list[tuple[int, ...]]
) -> np.ndarray:
    columns = []
    for monomial in monomials:
        value = np.ones(len(standardized))
        for feature, exponent in enumerate(monomial):
            value *= standardized[:, feature] ** exponent
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


def term_tex(label: str) -> str:
    parts = []
    for part in label.split("*"):
        base, _, exponent = part.partition("^")
        base = {"c200c": "c_{200c}"}.get(base, base)
        parts.append(base if not exponent else f"{base}^{{{exponent}}}")
    return "$" + r"\,".join(parts) + "$"


def cv_augmented(
    features: np.ndarray,
    targets: np.ndarray,
    discovery: bool,
) -> tuple[dict, dict[str, Counter], list[dict]]:
    n_galaxy, n_target = targets.shape
    mean_parameters = np.empty_like(targets)
    sigma_parameters = np.empty_like(targets)
    mean_cogs = np.empty((n_galaxy, len(RADII)))
    draw_cogs = np.empty((N_DRAW, n_galaxy, len(RADII)))
    mean_dropped = np.empty(n_galaxy, int)
    draw_dropped = np.empty((N_DRAW, n_galaxy), int)
    fold_id = np.empty(n_galaxy, int)
    occurrence = {name: Counter() for name in target_names("quantile3")}
    records = []

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

        for target, name in enumerate(target_names("quantile3")):
            if discovery:
                linear_beta, *_ = np.linalg.lstsq(
                    train_linear, targets[train, target], rcond=None
                )
                linear_residual = targets[train, target] - train_linear @ linear_beta
                regressor = new_symbolic_regressor()
                regressor.fit(
                    train_standardized,
                    linear_residual,
                    variable_names=[f"x{index}" for index in range(5)],
                )
                expression = sp.sympify(str(regressor.sympy()), locals=VARIABLE_LOCALS)
                monomials = nonlinear_monomials(expression)
            else:
                expression = "predefined"
                monomials = EXP50_TERMS[target]
            for monomial in monomials:
                occurrence[name][monomial_label(monomial)] += 1
            train_design = np.column_stack(
                [train_linear, monomial_design(train_standardized, monomials)]
            )
            test_design = np.column_stack(
                [test_linear, monomial_design(test_standardized, monomials)]
            )
            beta, *_ = np.linalg.lstsq(train_design, targets[train, target], rcond=None)
            mean_parameters[fold, target] = test_design @ beta
            train_residuals[:, target] = targets[train, target] - train_design @ beta
            gamma = _fit_logvar(
                train_residuals[:, target], train_standardized, ridge=2.0
            )
            variance_design = np.column_stack([np.ones(len(fold)), test_standardized])
            test_sigmas[:, target] = np.exp(0.5 * variance_design @ gamma)
            sigma_parameters[fold, target] = test_sigmas[:, target]
            records.append(
                {
                    "fold": fold_number,
                    "target": name,
                    "expression": str(expression),
                    "terms": ";".join(monomial_label(value) for value in monomials)
                    or "none",
                }
            )

        train_sigmas = np.empty_like(train_residuals)
        for target in range(n_target):
            gamma = _fit_logvar(
                train_residuals[:, target], train_standardized, ridge=2.0
            )
            variance_design = np.column_stack([np.ones(len(train)), train_standardized])
            train_sigmas[:, target] = np.exp(0.5 * variance_design @ gamma)
        correlation = np.corrcoef((train_residuals / train_sigmas).T)
        normal = np.random.default_rng(10000 + fold_number).standard_normal(
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
        if discovery:
            print(f"  completed symbolic fold {fold_number + 1}/5")
    result = {
        "mean_parameters": mean_parameters,
        "sigma_parameters": sigma_parameters,
        "mean_cogs": mean_cogs,
        "draw_cogs": draw_cogs,
        "mean_dropped": mean_dropped,
        "draw_dropped": draw_dropped,
        "fold_id": fold_id,
    }
    return result, occurrence, records


def fit_experiment() -> None:
    features, cogs_log, indices = load_z2_sample()
    targets = compress_cogs(cogs_log, RADII, "quantile3")
    print(
        f"exp51 portable z=2 symbolic search: n={len(features)}, "
        f"iterations={N_ITERATION}, draws={N_DRAW}"
    )
    linear = cv_quantile(features, targets, "quantile3", mean="linear", n_draw=N_DRAW)
    poly2 = cv_quantile(features, targets, "quantile3", mean="poly2", n_draw=N_DRAW)
    fixed, fixed_occurrence, _ = cv_augmented(features, targets, discovery=False)
    symbolic, occurrence, records = cv_augmented(features, targets, discovery=True)
    results = {
        "linear": summarize_model("linear", linear, cogs_log),
        "exp50_fixed_terms": summarize_model("exp50_fixed_terms", fixed, cogs_log),
        "symbolic": summarize_model("symbolic", symbolic, cogs_log),
        "poly2": summarize_model("poly2", poly2, cogs_log),
    }
    print("\n[portable z=2 symbolic comparison]")
    for name, summary in results.items():
        print(
            f"  {name:18s}: profile CRPS {summary['profile_crps_dex']:.5f} dex; "
            f"median profile RMS {summary['median_profile_rms_dex']:.5f} dex"
        )
    print("\n[terms recurring in at least four of five folds]")
    stable = {}
    for target, counter in occurrence.items():
        stable[target] = sorted(term for term, count in counter.items() if count >= 4)
        print(f"  {target:28s}: {stable[target] or ['none']}")
    figure(results, occurrence, len(features))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_galaxy": len(features),
        "redshift": 2.0,
        "n_iteration": N_ITERATION,
        "n_draw": N_DRAW,
        "input_scope": "DiffMAH fitted only to t<=t(z=2), plus c200c(z=2)",
        "models": results,
        "stable_terms": stable,
        "term_occurrence": {
            target: dict(counter) for target, counter in occurrence.items()
        },
        "exp50_fixed_term_occurrence": {
            target: dict(counter) for target, counter in fixed_occurrence.items()
        },
        "fold_records": records,
    }
    (OUTDIR / "results.json").write_text(json.dumps(payload, indent=2))
    np.savez_compressed(
        OUTDIR / "predictions.npz",
        indices=indices,
        features=features,
        truth_cogs_log=cogs_log,
        symbolic_mean_cogs_log=symbolic["mean_cogs"],
        symbolic_draw_cogs_log=symbolic["draw_cogs"],
    )
    write_manifest(
        OUTDIR,
        {
            "n_galaxy": len(features),
            "redshift": 2.0,
            "n_iteration": N_ITERATION,
            "n_draw": N_DRAW,
        },
    )
    print(f"wrote symbolic outputs to {OUTDIR}")


def figure(results: dict, occurrence: dict[str, Counter], n_galaxy: int) -> None:
    names = ["linear", "exp50_fixed_terms", "symbolic", "poly2"]
    colors = [OKABE_ITO[7], OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[4]]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9))
    axes[0].bar(
        ["linear", "exp50 terms", "new symbolic", "dense degree-2"],
        [results[name]["profile_crps_dex"] for name in names],
        color=colors,
    )
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set(
        ylabel="Held-out profile CRPS (dex)", title="Portable z=2 mean relation"
    )
    for name, color in zip(names, colors):
        axes[1].plot(
            RADII,
            results[name]["profile_crps_by_radius"],
            color=color,
            label=name.replace("_", " "),
        )
    axes[1].set(
        xscale="log",
        xlabel="Radius (kpc)",
        ylabel="Profile CRPS (dex)",
        title="Radial predictive score",
    )
    axes[1].legend(fontsize=7)
    all_terms = sorted({term for counter in occurrence.values() for term in counter})
    matrix = np.array(
        [
            [occurrence[target][term] for term in all_terms]
            for target in target_names("quantile3")
        ]
    )
    image = axes[2].imshow(matrix, vmin=0, vmax=5, cmap="cividis", aspect="auto")
    axes[2].set_xticks(
        range(len(all_terms)),
        [term_tex(term) for term in all_terms],
        rotation=45,
        ha="right",
    )
    axes[2].set_yticks(range(5), target_names("quantile3"))
    axes[2].set(title="Nonlinear-term recurrence across folds")
    fig.colorbar(image, ax=axes[2], label="Folds out of five")
    for label, axis in zip("ABC", axes):
        axis.text(
            -0.15,
            1.05,
            label,
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
        )
    fig.suptitle(
        f"exp51 — symbolic corrections with past-only z=2 inputs (n={n_galaxy})"
    )
    fig.tight_layout()
    save_fig(fig, FIGDIR / "exp51_symbolic_z2")
    plt.close(fig)


if __name__ == "__main__":
    fit_experiment()
