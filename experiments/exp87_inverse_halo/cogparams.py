"""exp87 Stage 8 — a curve of growth written as a FEW parameters.

Each family is fitted per galaxy to the 24 log cumulative masses (unweighted
least squares in log mass, 2-148 kpc). The parameters, not the 24 points, are
then the inputs of the symbolic search (`stage8_explore.py`).

  sersic    one Sersic profile, projected:  M(<R) = M_tot P(2n, b_n (R/R_e)^(1/n))
            parameters: log M_tot (to infinity), log R_e, n
  hill      a logistic in log R:            M(<R) = M_inf / (1 + (R_h/R)^a)
            parameters: log M_inf, log R_h (the half-mass radius), a (steepness)
  double    two exponential-type components with free Sersic index on the inner
            one: an inner Sersic + an outer exponential (n = 1) envelope
            parameters: log M_tot, the outer component's mass fraction,
            log R_inner, n_inner, log R_outer
  logpoly   a cubic in u = log10(R / 20 kpc): log M(<R) = a0 + a1 u + a2 u^2 + a3 u^3
            parameters: a0 (log M(<20 kpc)), a1 (the local slope), a2, a3
  sizes     not a fit: log M(<148 kpc) and the radii enclosing 20, 50 and 80 per
            cent of it (the non-parametric reference)

`fit_all` returns, per family, the parameter table, the column names and the
rms residual of the fit per galaxy [dex]; the result is cached by galaxy index.

Run (fits the parent, prints the quality of each family):
    PYTHONPATH=. uv run python -u experiments/exp87_inverse_halo/cogparams.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.special import gammainc, gammaincinv

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import config as C                                       # noqa: E402
import data as D                                         # noqa: E402

PARAMS_NPZ = C.OUTDIR / "stage8_cog_params.npz"
PIVOT_KPC = 20.0
N_BOUNDS = (0.3, 12.0)
LOG_RADIUS_BOUNDS = (-0.5, 3.5)
MASS_FLOOR = 1e-12


def sersic_fraction(radii, log_re, n):
    """The fraction of a projected Sersic profile's total mass inside each radius."""
    b = gammaincinv(2.0 * n, 0.5)
    return gammainc(2.0 * n, b * (radii / 10.0 ** log_re) ** (1.0 / n))


def model_sersic(theta, radii):
    return theta[0] + np.log10(np.clip(sersic_fraction(radii, theta[1], theta[2]), MASS_FLOOR, None))


def model_hill(theta, radii):
    return theta[0] - np.log10(1.0 + (10.0 ** theta[1] / radii) ** theta[2])


def model_double(theta, radii):
    """theta: log M_inner, log R_inner, n_inner, log M_outer, log R_outer (outer n = 1)."""
    inner = 10.0 ** theta[0] * sersic_fraction(radii, theta[1], theta[2])
    outer = 10.0 ** theta[3] * sersic_fraction(radii, theta[4], 1.0)
    return np.log10(np.clip(inner + outer, MASS_FLOOR, None))


def _fit(model, starts, lower, upper, radii, logm):
    best = None
    for start in starts:
        res = least_squares(lambda th: model(th, radii) - logm, np.clip(start, lower, upper), bounds=(lower, upper), x_scale="jac")
        if best is None or res.cost < best.cost:
            best = res
    return best.x, float(np.sqrt(2.0 * best.cost / len(radii)))


def fit_sersic(radii, logm):
    top = logm[-1]
    starts = [(top + 0.1, 1.2, 4.0), (top + 0.3, 1.6, 6.0), (top, 0.9, 2.0)]
    return _fit(model_sersic, starts, (top - 0.5, LOG_RADIUS_BOUNDS[0], N_BOUNDS[0]), (top + 2.5, LOG_RADIUS_BOUNDS[1], N_BOUNDS[1]), radii, logm)


def fit_hill(radii, logm):
    top = logm[-1]
    starts = [(top + 0.1, 1.2, 1.0), (top + 0.4, 1.7, 0.7), (top, 0.9, 1.5)]
    return _fit(model_hill, starts, (top - 0.5, LOG_RADIUS_BOUNDS[0], 0.1), (top + 3.0, LOG_RADIUS_BOUNDS[1], 6.0), radii, logm)


def fit_double(radii, logm):
    top = logm[-1]
    starts = [(top - 0.3, 0.6, 2.0, top - 0.3, 1.6), (top - 0.15, 0.9, 3.0, top - 0.6, 1.9), (top - 0.5, 0.4, 1.5, top - 0.1, 1.4)]
    lower = (top - 3.0, LOG_RADIUS_BOUNDS[0], N_BOUNDS[0], top - 3.0, 0.5)
    upper = (top + 1.0, 2.0, 8.0, top + 1.5, 3.0)
    theta, rms = _fit(model_double, starts, lower, upper, radii, logm)
    total = np.log10(10.0 ** theta[0] + 10.0 ** theta[3])
    return np.array([total, 10.0 ** (theta[3] - total), theta[1], theta[2], theta[4]]), rms


FAMILIES = {
    "sersic": (fit_sersic, ["logM_tot", "logR_e", "n"]),
    "hill": (fit_hill, ["logM_inf", "logR_h", "a"]),
    "double": (fit_double, ["logM_tot", "f_outer", "logR_in", "n_in", "logR_out"]),
}


def fit_logpoly(radii, logcog):
    u = np.log10(radii / PIVOT_KPC)
    design = np.vander(u, 4, increasing=True)
    coef, *_ = np.linalg.lstsq(design, logcog.T, rcond=None)
    rms = np.sqrt(np.mean((design @ coef - logcog.T) ** 2, axis=0))
    return coef.T, rms


def fit_all(force=False, verbose=True):
    """Fit every family to the parent's curves of growth (the curated sample's
    z = 0.4 curves are the same arrays). Returns dict(index, <family> (n, p),
    <family>_rms (n,), <family>_names)."""
    if PARAMS_NPZ.exists() and not force:
        return dict(np.load(PARAMS_NPZ, allow_pickle=True))
    parent = D.load_parent(verbose=False)
    radii, logcog = parent.radii, parent.logcog
    out = dict(index=parent.index)
    for name, (fitter, labels) in FAMILIES.items():
        t0 = time.time()
        fits = [fitter(radii, row) for row in logcog]
        out[name] = np.array([f[0] for f in fits])
        out[f"{name}_rms"] = np.array([f[1] for f in fits])
        out[f"{name}_names"] = np.array(labels)
        if verbose:
            print(f"    {name:<8} fitted to {len(logcog)} curves of growth in {time.time() - t0:.0f} s", flush=True)
    out["logpoly"], out["logpoly_rms"] = fit_logpoly(radii, logcog)
    out["logpoly_names"] = np.array(["a0", "a1", "a2", "a3"])
    out["sizes"] = np.column_stack([logcog[:, -1], D.sizes(logcog, radii)])
    out["sizes_rms"] = np.zeros(len(logcog))
    out["sizes_names"] = np.array(["logM148", "logR20", "logR50", "logR80"])
    C.OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez(PARAMS_NPZ, **out)
    return out


def table_for(index, family, params=None):
    """(n, p) parameters of `family` for the galaxies `index`, and the names."""
    params = params or fit_all()
    pos = {int(g): i for i, g in enumerate(params["index"])}
    rows = np.array([pos[int(g)] for g in index])
    return np.asarray(params[family], float)[rows], [str(s) for s in params[f"{family}_names"]]


def report(params):
    print(f"    {'family':<9}{'parameters':>11}{'median rms':>12}{'90th pct':>10}{'99th pct':>10}   parameter medians [16th, 84th]")
    for name in ("sersic", "hill", "double", "logpoly"):
        rms = params[f"{name}_rms"]
        table, labels = np.asarray(params[name]), [str(s) for s in params[f"{name}_names"]]
        spread = "; ".join(f"{lab} {np.median(col):.2f} [{np.percentile(col, 16):.2f}, {np.percentile(col, 84):.2f}]" for lab, col in zip(labels, table.T))
        print(f"    {name:<9}{table.shape[1]:>11}{np.median(rms):>12.4f}{np.percentile(rms, 90):>10.4f}{np.percentile(rms, 99):>10.4f}   {spread}")
    at_bound = {"sersic n": np.mean((params["sersic"][:, 2] <= N_BOUNDS[0] + 1e-3) | (params["sersic"][:, 2] >= N_BOUNDS[1] - 1e-3)),
                "double n_in": np.mean((params["double"][:, 3] <= N_BOUNDS[0] + 1e-3) | (params["double"][:, 3] >= 8.0 - 1e-3))}
    print("    fraction of fits at a bound: " + ", ".join(f"{k} {v:.3f}" for k, v in at_bound.items()))


if __name__ == "__main__":
    report(fit_all(force="--force" in sys.argv))
