"""exp30 — information-criterion (AIC/BIC) comparison of the multi-epoch models.

Fair parameter accounting:
  - NNLS models: k = outer params + ACTIVE (nonzero) masses. scipy nnls returns exact
    zeros for the inactive set; the active-set size is the standard effective-df
    estimate for sign-constrained least squares (LASSO-style).
  - exp29 parametric models: k = parametric count + 5 aperture pins (each epoch's
    amplitude is matched to the data at 148 kpc -> a data-fitted parameter).
  - Common data & residual: all n = 5x24 = 120 points, relative residual
    r = (model-data)/data; least-squares IC with sigma profiled out:
        AIC  = n ln(RSS/n) + 2k
        AICc = AIC + 2k(k+1)/(n-k-1)
        BIC  = n ln(RSS/n) + k ln(n)
  Caveats (printed): CoG points are correlated along radius and across epochs, so
  n=120 overstates independent information (sensitivity row with n_eff=40); residuals
  are misspecification-dominated (no noise model) -> ICs rank descriptive parsimony,
  not truth; 'alone' and 'loose-quad' are not consistent histories -- the scientific
  constraint ICs cannot see.

Run: PYTHONPATH=. uv run python experiments/exp30_transport_kernel/ic_compare.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, nnls

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP29 = ROOT / "experiments" / "exp29_outer_deposit"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP29))
import transport_floor as tf                                                          # noqa: E402
from run import ANCHOR_SNAP, ANCHOR_Z                                                # noqa: E402
from real_mah_test import real_mah                                                   # noqa: E402
from hongshao.plotting import set_style, save_fig                                    # noqa: E402

set_style()
FIGDIR = HERE / "figures"
R = tf.R
N_PTS = 5 * len(R)                                    # 120
# model -> (source, explicit params incl. pins)
PARAMETRIC = {"independent": 25 + 5, "loose-quad": 15 + 5, "puff-ratio": 6 + 5}
NNLS_OUTER = {"alone": 10, "additive": 2, "transport": 4, "envelope": 5}
ORDER = ["alone", "additive", "transport", "envelope", "independent", "loose-quad", "puff-ratio"]


def solve_with_count(theta, mah, data, mode):
    """Joint NNLS at fixed theta -> (cogs (5,24), n_active)."""
    ti, snap, t_obs = mah["t"], mah["snap"], mah["t_obs"]
    masks = [snap <= sa for sa in ANCHOR_SNAP]
    blocks = [tf.basis(theta, ti, t_obs, tf.AT[k], mode) * masks[k][None, :] for k in range(5)]
    A = np.vstack([b / data[k][:, None] for k, b in enumerate(blocks)])
    x, _ = nnls(A, np.ones(A.shape[0]), maxiter=10 * A.shape[1])
    return np.array([b @ x for b in blocks]), int((x > 0).sum())


def alone_with_count(mah, data):
    """Per-epoch free-mass NNLS (as tf.fit_alone) -> (cogs, total active masses)."""
    ti, snap, t_obs = mah["t"], mah["snap"], mah["t_obs"]
    cogs, n_act = [], 0
    for k in range(5):
        m = snap <= ANCHOR_SNAP[k]

        def loss(th):
            B = tf.basis(th, ti[m], t_obs, tf.AT[k], "additive")
            _, rn = nnls(B / data[k][:, None], np.ones(len(R)), maxiter=10 * int(m.sum()))
            return rn

        best = None
        for p0 in ([1.8, 1.0], [2.3, 1.7], [2.6, 1.7]):
            r = minimize(loss, p0, method="Nelder-Mead",
                         options=dict(maxiter=1500, xatol=1e-4, fatol=1e-10))
            if best is None or r.fun < best.fun:
                best = r
        B = tf.basis(best.x, ti[m], t_obs, tf.AT[k], "additive")
        x, _ = nnls(B / data[k][:, None], np.ones(len(R)), maxiter=10 * int(m.sum()))
        cogs.append(B @ x); n_act += int((x > 0).sum())
    return np.array(cogs), n_act


def ics(rss, k, n):
    gof = n * np.log(rss / n)
    aic = gof + 2 * k
    aicc = aic + (2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else np.inf)
    bic = gof + k * np.log(n)
    return aic, aicc, bic


def main():
    d = np.load(HERE / "outputs" / "transport_floor.npz")
    s = np.load(EXP29 / "outputs" / "final_scorecard.npz", allow_pickle=True)
    assert np.array_equal(d["index"], s["index"]), "sample mismatch between npz files"
    datas = d["data"]; ng = len(datas)
    snames = list(s["names"])

    cogs, keff = {}, {}
    for nm in PARAMETRIC:                              # exp29 parametric (cached CoGs)
        cogs[nm] = s["models"][:, snames.index(nm)]
        keff[nm] = np.full(ng, PARAMETRIC[nm], float)
    cogs["additive"] = d["additive"]; cogs["transport"] = d["transport"]
    cogs["envelope"] = d["envelope"]; cogs["alone"] = d["alone"]
    for nm in NNLS_OUTER:
        keff[nm] = np.zeros(ng)

    for i in range(ng):                                # active-mass counts
        mah = real_mah(int(d["index"][i]))
        data = [datas[i][k] for k in range(5)]
        _, th_add = tf.fit_joint(mah, data, "additive")
        for nm, th in (("additive", th_add), ("transport", d["params_transport"][i]),
                       ("envelope", d["params_envelope"][i])):
            _, na = solve_with_count(th, mah, data, nm)
            keff[nm][i] = NNLS_OUTER[nm] + na
        _, na = alone_with_count(mah, data)
        keff["alone"][i] = NNLS_OUTER["alone"] + na

    rss = {nm: np.array([(((cogs[nm][i] - datas[i]) / datas[i]) ** 2).sum() for i in range(ng)])
           for nm in ORDER}
    table = {}
    for nm in ORDER:
        vals = np.array([ics(rss[nm][i], keff[nm][i], N_PTS) for i in range(ng)])   # (n,3)
        table[nm] = vals
    base = {c: np.min([np.median(table[nm][:, c]) for nm in ORDER]) for c in range(3)}

    print(f"exp30 — IC comparison (n_gal={ng}, n_pts={N_PTS} relative residuals, "
          "k = explicit + active NNLS masses + pins)\n")
    print(f"  {'model':>12s} | {'k_eff':>6s} | {'relRMS':>7s} | {'dAIC':>8s} | {'dAICc':>8s} | "
          f"{'dBIC':>8s} | consistent history?")
    CONS = {"alone": "no (5 separate fits)", "additive": "yes", "transport": "yes",
            "envelope": "yes", "independent": "no (5 separate fits)",
            "loose-quad": "no (params drift with z_k)", "puff-ratio": "yes"}
    for nm in ORDER:
        med = [np.median(table[nm][:, c]) for c in range(3)]
        print(f"  {nm:>12s} | {np.median(keff[nm]):6.0f} | "
              f"{100*np.sqrt(np.median(rss[nm])/N_PTS):6.1f}% | {med[0]-base[0]:8.0f} | "
              f"{med[1]-base[1]:8.0f} | {med[2]-base[2]:8.0f} | {CONS[nm]}")

    neff = 40                                          # correlation sensitivity: ~8 radii x 5 epochs
    print(f"\n  BIC sensitivity with n_eff={neff} (correlated CoG points), dBIC vs best:")
    tb = {nm: np.median([ics(rss[nm][i] * neff / N_PTS, keff[nm][i], neff)[2] for i in range(ng)])
          for nm in ORDER}
    b0 = min(tb.values())
    print("    " + "   ".join(f"{nm} {tb[nm]-b0:+.0f}" for nm in ORDER))

    print("\n  caveats: CoG points are correlated (n overstated; see sensitivity row); residuals")
    print("  are model-misspecification, not noise -> ICs rank descriptive parsimony only; and")
    print("  ICs cannot see the CONSISTENCY requirement -- 'alone'/'independent'/'loose-quad' fit")
    print("  better precisely by abandoning a single history. Compare within the consistent class.")
    _figure(table, keff, rss, base)


def _figure(table, keff, rss, base):
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.0, 5.0))
    x = np.arange(len(ORDER))
    for off, c, lab in ((-0.2, 0, "dAIC"), (0.0, 1, "dAICc"), (0.2, 2, "dBIC")):
        vals = [np.median(table[nm][:, c]) - base[c] for nm in ORDER]
        a.bar(x + off, vals, width=0.2, label=lab)
    a.set_xticks(x); a.set_xticklabels(ORDER, rotation=30, fontsize=8, ha="right")
    a.set(ylabel="median $\\Delta$IC vs best", title="A. Information criteria (n=120 rel. residuals)")
    a.axhline(0, c="0.5", lw=0.8); a.legend(fontsize=8)
    med_k = [np.median(keff[nm]) for nm in ORDER]
    rel = [100 * np.sqrt(np.median(rss[nm]) / N_PTS) for nm in ORDER]
    b2 = b.twinx()
    b.bar(x - 0.15, med_k, width=0.3, color="0.6", label="k_eff")
    b2.bar(x + 0.15, rel, width=0.3, color="#0072B2", label="rel-RMS [%]")
    b.set_xticks(x); b.set_xticklabels(ORDER, rotation=30, fontsize=8, ha="right")
    b.set(ylabel="median effective parameters", title="B. Effective params vs fit quality")
    b2.set_ylabel("relative RMS [%]", color="#0072B2")
    fig.suptitle("exp30 — fair model comparison: effective parameters (active NNLS masses + pins) "
                 "and information criteria", fontsize=12)
    fig.tight_layout()
    print("\nwrote", save_fig(fig, FIGDIR / "exp30_ic_compare")[0])


if __name__ == "__main__":
    sys.exit(main())
