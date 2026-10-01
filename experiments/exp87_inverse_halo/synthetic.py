"""exp87 Stage 1 — the MECHANICS GATE: the scoring, the truncated-likelihood
fits and the harness, exercised end to end on synthetic populations whose
truth is known. No real galaxy is scored before this passes.

The synthetic world mirrors the real one. A population of haloes has log mass
y drawn from a steep power-law mass function (log-slope -gamma); each carries a
4-coordinate "curve of growth" x = a + b y + noise; the SAMPLE is the part with
y >= 13.0. For this generator the population's p(y | x) is a normal with a
LINEAR mean and a constant scale (known in closed form), so the sample's
p(y | x) is exactly a truncated normal with those latent parameters: the oracle.

Checks (the plan's list; each must pass or Stage 1 STOPS):
  A  the oracle scores better than the ordinary normal fit, than the ordinary
     fit renormalised above the cut (truncation counted twice) and than a
     shifted oracle; the fitted truncated-linear model is within 1% of the oracle
  B  the truncated likelihood recovers the latent slopes and scale within 3%
  C  the fitted predictive's PIT is uniform out of fold
  D  on this LINEAR truth, flexible learners through the common head gain
     nothing; the 95th percentile of their best-of gain is the null level,
     and delta = max(1%, that) is the threshold every later stage uses
  E  an injected curvature of the latent mean worth 3% of the CRPS is detected
     (significant by the paired rule at delta) in at least 80% of replicates
  F  thinning the sample in a way that depends on x at fixed y (the curated
     sample's defect) biases the fitted relation in the known direction
  G  for the record: the range-restricted R^2 and the ordinary fit's scatter
     against the population's

Run (detached, ~15 min):
    PYTHONPATH=. uv run python -u experiments/exp87_inverse_halo/synthetic.py [--smoke]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import config as C                                       # noqa: E402
import harness as HN                                     # noqa: E402
import heads as H                                        # noqa: E402
import methods as M                                      # noqa: E402
import scoring as S                                      # noqa: E402

RULE = "=" * 100
GAMMA, CUT, Y_LO, Y_HI = 1.15, 13.0, 11.8, 15.6
N_SAMPLE = 3380
A_VEC = np.array([2.3, 0.6, 0.2, 0.0])
B_VEC = np.array([0.67, 0.30, 0.10, 0.0])
NOISE_SD = np.array([0.154, 0.20, 0.15, 0.10])
NOISE_CORR = np.array([[1.0, 0.5, 0.2, 0.0], [0.5, 1.0, 0.3, 0.0], [0.2, 0.3, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
SIGMA = NOISE_SD[:, None] * NOISE_CORR * NOISE_SD[None, :]
SIGMA_INV = np.linalg.inv(SIGMA)
S_LATENT = float(1.0 / np.sqrt(B_VEC @ SIGMA_INV @ B_VEC))
SLOPES_LATENT = S_LATENT ** 2 * (SIGMA_INV @ B_VEC)
SHIFT_PRIOR = GAMMA * np.log(10.0) * S_LATENT ** 2
TARGET_CURVATURE_LOSS = 0.03


def latent_mean(x):
    """The population's E[y | x] for the linear generator (prior shift included)."""
    return (x - A_VEC) @ SLOPES_LATENT - SHIFT_PRIOR


def draw_population(n, rng):
    u = rng.uniform(size=n)
    y = Y_LO - np.log10(1.0 - u * (1.0 - 10.0 ** (-GAMMA * (Y_HI - Y_LO)))) / GAMMA
    x = A_VEC + np.outer(y, B_VEC) + rng.multivariate_normal(np.zeros(4), SIGMA, size=n)
    return x, y


def draw_sample(rng, n=N_SAMPLE, kappa=0.0, thin=0.0):
    """A sample of `n` selected objects: (x, y, latent m, latent s, fold).
    kappa != 0: y is redrawn from a latent normal with a CURVED mean (the
    inverse generator; the sample's p(y | x) is then exactly truncated normal
    with that mean). thin > 0: drop this fraction of the objects whose first
    coordinate lies high at fixed y (selection depending on x at fixed y)."""
    xs, ys, ms = [], [], []
    while sum(len(v) for v in ys) < n:
        x, y = draw_population(40 * n, rng)
        m = latent_mean(x)
        if kappa != 0.0:
            u = (x[:, 0] - 11.1) / 0.3
            m = m + kappa * (u * u - 1.0)
            y = m + S_LATENT * rng.standard_normal(len(m))
        keep = y >= CUT
        if thin > 0.0:
            high = (x[:, 0] - A_VEC[0] - B_VEC[0] * y) > 0.5 * NOISE_SD[0]
            keep &= ~(high & (rng.uniform(size=len(y)) < thin))
        xs.append(x[keep])
        ys.append(y[keep])
        ms.append(m[keep])
    x, y, m = (np.concatenate(v)[:n] for v in (xs, ys, ms))
    group = rng.permutation(n) % 5
    fold = np.where(group == 0, -1, 0)
    dev = np.flatnonzero(group != 0)
    fold[dev] = rng.permutation(len(dev)) % C.N_FOLDS
    return x, y, m, np.full(n, S_LATENT), fold


def poly2_method():
    return M.DirectLinear("poly2", "L2", ridge_grid=M.RIDGE_GRID,
                          transform=lambda: make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False)))


def replicate_scores(x, y, fold, flexible=True):
    """Per-galaxy out-of-fold CRPS of the ladder's representatives on one replicate."""
    out = {}
    lin = M.DirectLinear("linear", "L1", heteroscedastic=False)
    out["linear"], _ = HN.oof_scores(lin, x, y, fold, CUT)
    if flexible:
        out["poly2"], _ = HN.oof_scores(poly2_method(), x, y, fold, CUT)
        out["gbm"], _ = HN.oof_scores(M.LearnerHead("gbm", "L3", M._gbm), x, y, fold, CUT)
        out["gbm-mills"], _ = HN.oof_scores(M.LearnerHead("gbm-mills", "L3", M._gbm, mills=True), x, y, fold, CUT)
        out["ridge+head"], _ = HN.oof_scores(M.LearnerHead("ridge+head", "L1", M._scaled(lambda: M.Ridge(alpha=1e-3))), x, y, fold, CUT)
    return out


def calibrate_kappa(rng):
    """The curvature whose neglect costs TARGET_CURVATURE_LOSS of the CRPS (bisection on a large sample)."""
    def loss(kappa):
        x, y, m, s, _ = draw_sample(np.random.default_rng(5), n=40000, kappa=kappa)
        oracle = S.TruncNormal(m, s, CUT).crps(y).mean()
        fit = H.TruncLinear(CUT).fit(x, y)
        return fit.predict(x).crps(y).mean() / oracle - 1.0
    lo, hi = 0.0, 0.3
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if loss(mid) < TARGET_CURVATURE_LOSS else (lo, mid)
    return 0.5 * (lo + hi), loss(0.5 * (lo + hi))


def main(smoke=False):
    t_start = time.time()
    n_rep_null, n_rep_curve = (3, 2) if smoke else (14, 10)
    rng = np.random.default_rng(C.SEED)
    gate = {"config": C.config_hash(), "smoke": smoke}
    print(f"{RULE}\nexp87 STAGE 1 — the mechanics gate on synthetic truth{' (SMOKE)' if smoke else ''}\n{RULE}\n")
    print(f"  the synthetic world: mass-function log-slope {GAMMA}, cut {CUT}; latent scale s = {S_LATENT:.4f} dex, latent slopes "
          + " / ".join(f"{v:+.3f}" for v in SLOPES_LATENT) + f"; the prior shifts the inverse mean by {-SHIFT_PRIOR:+.3f} dex")

    # ---- A, B, C, D, G on the linear truth ----------------------------------- #
    recs = []
    for r in range(n_rep_null):
        x, y, m, s, fold = draw_sample(rng)
        used = fold >= 0
        sc = replicate_scores(x, y, fold)
        oracle = S.TruncNormal(m, s, CUT)
        naive, _ = HN.oof_scores(M.DirectLinear("naive", "L1", heteroscedastic=False), x, y, fold, None)
        double = S.TruncNormal(naive["latent_m"][used], naive["latent_s"][used], CUT)
        full = H.TruncLinear(CUT).fit(x[used], y[used])
        _, slopes = full.raw_coefficients()
        ordinary = H.TruncLinear(None).fit(x[used], y[used])
        xp, yp = draw_population(200000, rng)
        r2_pop = 1.0 - np.var(yp - latent_mean(xp)) / np.var(yp)
        r2_sel = 1.0 - np.mean(sc["linear"]["err_mean"][used] ** 2) / np.var(y[used])
        recs.append(dict(
            oracle=oracle.crps(y)[used].mean(), linear=sc["linear"]["crps"][used].mean(), naive=naive["crps"][used].mean(),
            double=double.crps(y[used]).mean(), shifted=S.TruncNormal(m + 0.05, s, CUT).crps(y)[used].mean(),
            slope_ratio=float(np.linalg.norm(slopes[:3]) / np.linalg.norm(SLOPES_LATENT[:3])), slopes=slopes,
            scale_ratio=float(np.median(full.latent(x[used])[1]) / S_LATENT),
            naive_scale_ratio=float(np.median(ordinary.latent(x[used])[1]) / S_LATENT),
            naive_slope_ratio=float(np.linalg.norm(ordinary.raw_coefficients()[1][:3]) / np.linalg.norm(SLOPES_LATENT[:3])),
            pit=sc["linear"]["pit"][used], r2_pop=r2_pop, r2_sel=r2_sel,
            gains={k: 1.0 - sc[k]["crps"][used].mean() / sc["linear"]["crps"][used].mean() for k in sc if k != "linear"},
            fraction_safe=float(np.mean(m - 2.326 * s >= CUT))))
        print(f"    replicate {r + 1}/{n_rep_null}: oracle {recs[-1]['oracle']:.4f}, truncated-linear {recs[-1]['linear']:.4f}, ordinary {recs[-1]['naive']:.4f}, "
              f"double-counted {recs[-1]['double']:.4f}; gains over linear " + ", ".join(f"{k} {100 * v:+.2f}%" for k, v in recs[-1]["gains"].items())
              + f"  ({time.time() - t_start:.0f} s)", flush=True)
    mean = {k: float(np.mean([r[k] for r in recs])) for k in ("oracle", "linear", "naive", "double", "shifted", "slope_ratio",
                                                             "scale_ratio", "naive_scale_ratio", "naive_slope_ratio", "r2_pop", "r2_sel", "fraction_safe")}
    # the slope VECTOR is compared after averaging over replicates: a per-replicate direction cosine is
    # biased below one by the noise in the two near-zero slopes (seen in the smoke run)
    mean_slopes = np.mean([r["slopes"] for r in recs], axis=0)
    mean["slope_error"] = float(np.linalg.norm(mean_slopes - SLOPES_LATENT) / np.linalg.norm(SLOPES_LATENT))
    pit = np.concatenate([r["pit"] for r in recs])
    best_null = np.array([max(r["gains"][k] for k in ("poly2", "gbm", "gbm-mills")) for r in recs])
    delta = max(C.GAIN_FLOOR, float(np.quantile(best_null, 0.95)))
    route = float(np.mean([abs(r["gains"]["ridge+head"]) for r in recs]))
    check = {}
    check["A_oracle_beats_wrong_predictives"] = bool(mean["oracle"] < mean["naive"] and mean["oracle"] < mean["double"] and mean["oracle"] < mean["shifted"]
                                                     and mean["linear"] / mean["oracle"] - 1.0 < 0.01)
    check["B_slopes_and_scale_within_3pct"] = bool(mean["slope_error"] < 0.03 and abs(mean["scale_ratio"] - 1) < 0.03)
    check["C_pit_uniform"] = bool(abs(pit.mean() - 0.5) < 0.01 and abs(pit.var() - 1 / 12) < 0.004)
    check["D_no_gain_on_linear_truth"] = bool(np.median(best_null) < C.GAIN_FLOOR and route < 0.01)
    print(f"\n  A  mean CRPS over {n_rep_null} replicates [dex]: oracle {mean['oracle']:.4f} | truncated-linear fit {mean['linear']:.4f} "
          f"(+{100 * (mean['linear'] / mean['oracle'] - 1):.2f}%) | ordinary normal {mean['naive']:.4f} (+{100 * (mean['naive'] / mean['oracle'] - 1):.1f}%) | "
          f"ordinary renormalised above the cut {mean['double']:.4f} (+{100 * (mean['double'] / mean['oracle'] - 1):.1f}%) | "
          f"oracle shifted 0.05 dex {mean['shifted']:.4f}   {'PASS' if check['A_oracle_beats_wrong_predictives'] else 'FAIL'}")
    print("  B  truncated likelihood: replicate-mean slopes " + " / ".join(f"{v:+.3f}" for v in mean_slopes) + " (truth "
          + " / ".join(f"{v:+.3f}" for v in SLOPES_LATENT) + f"), relative error of the vector {100 * mean['slope_error']:.2f}%, scale ratio {mean['scale_ratio']:.3f} "
          f"(ordinary fit: slopes {mean['naive_slope_ratio']:.3f}, scale {mean['naive_scale_ratio']:.3f})   {'PASS' if check['B_slopes_and_scale_within_3pct'] else 'FAIL'}")
    print(f"  C  out-of-fold PIT of the truncated-linear predictive: mean {pit.mean():.4f}, variance {pit.var():.4f} (uniform 0.5000, 0.0833)   "
          f"{'PASS' if check['C_pit_uniform'] else 'FAIL'}")
    print(f"  D  best-of(poly2, gbm, gbm-mills) gain over linear on the LINEAR truth: median {100 * np.median(best_null):+.2f}%, 95th percentile "
          f"{100 * np.quantile(best_null, 0.95):+.2f}% -> delta = {100 * delta:.2f}%; the head route differs from the direct route by {100 * route:.2f}% "
          f"(ridge)   {'PASS' if check['D_no_gain_on_linear_truth'] else 'FAIL'}")
    print(f"  G  for the record: R^2 of the same relation {mean['r2_pop']:.3f} in the population, {mean['r2_sel']:.3f} in the selected sample (range "
          f"restriction); {100 * mean['fraction_safe']:.0f}% of the sample is truncation-safe (P(y < cut | x) < 1%)")

    # ---- E: the injected curvature ------------------------------------------- #
    kappa, loss = calibrate_kappa(rng)
    detected, gains = [], []
    for r in range(n_rep_curve):
        x, y, m, s, fold = draw_sample(rng, kappa=kappa)
        used = fold >= 0
        sc = replicate_scores(x, y, fold)
        best = min(("poly2", "gbm", "gbm-mills"), key=lambda k: sc[k]["crps"][used].mean())
        ok, boot = S.significant_gain(sc[best]["crps"][used], sc["linear"]["crps"][used], delta)
        detected.append(ok)
        gains.append(-boot["rel"])
        print(f"    curvature replicate {r + 1}/{n_rep_curve}: best {best}, gain {100 * -boot['rel']:.2f}% [{100 * -boot['rel_hi']:.2f}, {100 * -boot['rel_lo']:.2f}]"
              f"  {'detected' if ok else 'missed'}  ({time.time() - t_start:.0f} s)", flush=True)
    check["E_curvature_detected"] = bool(np.mean(detected) >= 0.8)
    print(f"  E  injected curvature kappa = {kappa:.4f} (neglecting it costs {100 * loss:.2f}% of the CRPS): detected in {int(np.sum(detected))} of "
          f"{n_rep_curve} replicates, median gain {100 * np.median(gains):.2f}%   {'PASS' if check['E_curvature_detected'] else 'FAIL'}")

    # ---- F: selection depending on x at fixed y ------------------------------ #
    shifts = []
    for r in range(4 if not smoke else 2):
        x, y, m, s, fold = draw_sample(rng, thin=0.6)
        fit = H.TruncLinear(CUT).fit(x, y)
        shifts.append(float(np.mean(fit.latent(x)[0] - m)))
    check["F_thinning_bias_shown"] = bool(np.mean(shifts) > 0.01 and np.mean(shifts) > 3 * np.std(shifts) / np.sqrt(len(shifts)))
    print(f"  F  dropping 60% of the objects whose first coordinate lies high at fixed y: the fitted latent mean is biased by "
          f"{np.mean(shifts):+.3f} +/- {np.std(shifts):.3f} dex (expected positive)   {'PASS' if check['F_thinning_bias_shown'] else 'FAIL'}")

    gate.update(checks=check, delta=delta, mean=mean, pit_mean=float(pit.mean()), pit_var=float(pit.var()),
                null_gains=best_null.tolist(), kappa=kappa, curvature_gains=gains, thinning_shift=float(np.mean(shifts)),
                s_latent=S_LATENT, passed=bool(all(check.values())), seconds=round(time.time() - t_start, 1))
    C.OUTDIR.mkdir(parents=True, exist_ok=True)
    out = C.OUTDIR / f"gate_mechanics{'_smoke' if smoke else ''}.json"
    out.write_text(json.dumps(gate, indent=1))
    print(f"\n  STAGE 1 {'PASSED' if gate['passed'] else 'FAILED — STOP'}; delta = {100 * delta:.2f}%; {gate['seconds']:.0f} s; wrote {out}")
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    sys.exit(main(smoke="--smoke" in sys.argv))
