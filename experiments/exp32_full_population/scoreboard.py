"""exp32 step 3 — the honest scoreboard vs mass (full population, n~2397).

Models (all held-out, halo-only inputs, full (n,5,24) CoGs):
  transport         universal-theta 10-fold-CV CoGs (primary config) x LOO SHMR
  transport-bins    the mass-binned-theta variant (if its CV exists)
  transport-real    real-MAH config (config check down-mass)
  logmh-only        closed-form LOO regression log M*(<R,z_k) <- logMh(z_k)
  direct            + z=0.4 halo secondaries (c200c, t50, fz2) — exp31 pattern
  direct-epoch      EPOCH-MATCHED history features only: logMh(z_k),
                    t50(z_k)/t(z_k), Mh(t_k/2)/Mh(t_k) — settles whether the
                    MAH's high-z irrelevance was feature misalignment

LOO linear regressions use the exact hat-matrix identity (e_loo = e/(1-h)), so
the full sample costs nothing. QA = hongshao.qa tiers, reported per logM*
quartile.

Run: PYTHONPATH=. uv run python experiments/exp32_full_population/scoreboard.py [--dev]
Demo: ... scoreboard.py demo
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP29 = ROOT / "experiments" / "exp29_outer_deposit"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP29))
from run import ANCHOR_Z, ANCHOR_SNAP                                                # noqa: E402
from real_mah_test import real_mah                                                   # noqa: E402
from hongshao import qa                                                              # noqa: E402
from hongshao.tng_data import COG_RAD_KPC, load_cosmic_time                          # noqa: E402
from hongshao.plotting import set_style, save_fig                                    # noqa: E402

set_style()
FIGDIR, OUTDIR = HERE / "figures", HERE / "outputs"
POP_NPZ = OUTDIR / "population.npz"
R = COG_RAD_KPC
AT = load_cosmic_time()[ANCHOR_SNAP]
COLORS = {"transport": "#D55E00", "transport-bins": "#CC3377",
          "transport-real": "#999933", "logmh-only": "0.45",
          "direct": "#0072B2", "direct-epoch": "#009E73"}


def loo_linear(X, y):
    """Exact leave-one-out predictions of a linear fit (hat-matrix identity)."""
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ beta
    AtA_inv = np.linalg.pinv(A.T @ A)
    h = np.einsum("ij,jk,ik->i", A, AtA_inv, A)
    return y - (y - yhat) / np.clip(1.0 - h, 1e-9, None)


def regression_cogs(feats_by_epoch, data):
    """LOO per-radius/epoch regression CoGs (n,5,24)."""
    n = data.shape[0]
    cogs = np.empty((n, 5, 24))
    for k in range(5):
        for r in range(24):
            cogs[:, k, r] = 10.0 ** loo_linear(feats_by_epoch[k],
                                               np.log10(data[:, k, r]))
    return cogs


def epoch_features(pop):
    """Epoch-matched real-MAH history features: t50(z_k)/t(z_k), Mh(t_k/2)/Mh(t_k)."""
    n = len(pop["index"])
    t50_rel = np.empty((n, 5))
    f_half = np.empty((n, 5))
    for i in range(n):
        mah = real_mah(int(pop["index"][i]))
        Mh, tf = 10.0 ** mah["logMh_full"], mah["t_full"]
        for k in range(5):
            tk = AT[k]
            mk = float(np.interp(tk, tf, Mh))
            t50_rel[i, k] = float(np.interp(0.5 * mk, Mh, tf)) / tk
            f_half[i, k] = float(np.interp(0.5 * tk, tf, Mh)) / mk
    return t50_rel, f_half


def build_models(dev=False):
    pop = np.load(POP_NPZ)
    rows = pop["dev100"] if dev else np.arange(len(pop["index"]))
    data = pop["data"][rows]
    tag = "_dev" if dev else ""
    models = {}

    # transport variants: CV cogs (pinned) rescaled to the LOO SHMR amplitude
    logms_zk = np.log10(data[:, :, -1])
    for name, cvfile, key, mhkey in (
            ("transport", f"um_cv_diffmah{tag}.npz", "cogs_global", "logmh_zk_diffmah"),
            ("transport-bins", f"um_cv_diffmah{tag}.npz", "cogs_bins", "logmh_zk_diffmah"),
            ("transport-real", f"um_cv_real{tag}.npz", "cogs_global", "logmh_zk_real")):
        f = OUTDIR / cvfile
        if not f.exists():
            continue
        d = np.load(f)
        if key not in d:
            continue
        shmr = np.column_stack([loo_linear(pop[mhkey][rows][:, [k]], logms_zk[:, k])
                                for k in range(5)])
        scale = 10.0 ** shmr / data[:, :, -1]
        models[name] = d[key] * scale[:, :, None]

    # regressions (closed-form LOO)
    mh = pop["logmh_zk_real"][rows]
    sec = np.column_stack([pop["c200c"][rows], pop["t50"][rows], pop["fz2"][rows]])
    t50_rel, f_half = epoch_features(pop)
    t50_rel, f_half = t50_rel[rows], f_half[rows]
    models["logmh-only"] = regression_cogs([mh[:, [k]] for k in range(5)], data)
    models["direct"] = regression_cogs(
        [np.column_stack([mh[:, k], sec]) for k in range(5)], data)
    models["direct-epoch"] = regression_cogs(
        [np.column_stack([mh[:, k], t50_rel[:, k], f_half[:, k]]) for k in range(5)],
        data)
    return pop, rows, data, models


def main():
    dev = "--dev" in sys.argv
    pop, rows, data, models = build_models(dev)
    logms = pop["logms"][rows]
    edges = np.quantile(logms, np.linspace(0, 1, 5))
    qbin = [np.where((logms >= edges[q]) & (logms <= edges[q + 1] + 1e-9))[0]
            for q in range(4)]

    results = {m: qa.evaluate(models[m], data, R, ANCHOR_Z, name=m, figdir=FIGDIR,
                              verbose=False, figures=(m == "transport" and not dev))
               for m in models}

    is_aper = lambda k: "(<" in k
    print(f"\n=== exp32 scoreboard (held-out, n={len(rows)}) by logM* quartile "
          f"(edges {np.round(edges, 2)}) ===")
    for label, getter in (
            ("tier1 kpc apertures [dex]", lambda r, s: np.median(
                [qa.dex_scatter(r["model"][k][s], r["truth"][k][s])
                 for k in r["keys"] if k.startswith("kpc:") and is_aper(k)])),
            ("tier2 Re envelopes [dex]", lambda r, s: np.median(
                [qa.dex_scatter(r["model"][k][s], r["truth"][k][s])
                 for k in r["keys"] if k.startswith("Re:") and "(>" in k])),
            ("tier3 max|rel| R>5kpc [%]", lambda r, s: 100 * np.nanmean(
                np.nanmedian(r["mr_out"][s], axis=0)))):
        print(f"\n  {label}")
        for m, r in results.items():
            cells = [f"{getter(r, q):6.3f}" if "dex" in label else f"{getter(r, q):5.1f}%"
                     for q in qbin]
            allv = getter(r, np.arange(len(rows)))
            print(f"    {m:>16s} | " + " | ".join(cells)
                  + (f" | {allv:6.3f}" if "dex" in label else f" | {allv:5.1f}%"))
    print("\n  tier2b plane fidelity |model-truth| scatter, M(<30) vs M(50-100), "
          "epoch-avg [dex]:")
    for m, r in results.items():
        st = r["planes"][("kpc:M(<30)", "kpc:M(50-100)")]
        print(f"    {m:>16s} : "
              f"{np.nanmean([abs(mo['scatter'] - t['scatter']) for t, mo in st]):.3f}")
    _figure(results, qbin, edges, len(rows))


def _figure(results, qbin, edges, n):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    x = np.arange(4)
    labels = [f"{edges[q]:.2f}-{edges[q+1]:.2f}" for q in range(4)]
    for m, r in results.items():
        a = [np.median([qa.dex_scatter(r["model"][k][s], r["truth"][k][s])
                        for k in r["keys"] if k.startswith("kpc:") and "(<" in k])
             for s in qbin]
        axes[0].plot(x, a, "o-", c=COLORS[m], label=m)
        e = [np.median([qa.dex_scatter(r["model"][k][s], r["truth"][k][s])
                        for k in r["keys"] if k.startswith("Re:") and "(>" in k])
             for s in qbin]
        axes[1].plot(x, e, "o-", c=COLORS[m], label=m)
        p = [100 * np.nanmean(np.nanmedian(r["mr_out"][s], axis=0)) for s in qbin]
        axes[2].plot(x, p, "o-", c=COLORS[m], label=m)
    for ax, t, yl in ((axes[0], "A. kpc apertures", "held-out dex scatter"),
                      (axes[1], "B. Re envelopes", "held-out dex scatter"),
                      (axes[2], "C. profile max|rel| (R>5 kpc)", "median [%]")):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set(xlabel="logM* quartile", ylabel=yl, title=t)
        ax.grid(alpha=0.25, axis="y")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"exp32 scoreboard — held-out error vs stellar mass (n={n})",
                 fontsize=12)
    fig.tight_layout()
    print("\nwrote", save_fig(fig, FIGDIR / "exp32_scoreboard")[0])


def demo():
    """Self-check: hat-matrix LOO == brute-force LOO on a small noisy problem."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 2))
    y = 1.0 + X @ [0.5, -1.2] + 0.1 * rng.normal(size=30)
    fast = loo_linear(X, y)
    slow = np.empty(30)
    for i in range(30):
        m = np.arange(30) != i
        b = np.linalg.lstsq(np.column_stack([np.ones(29), X[m]]), y[m], rcond=None)[0]
        slow[i] = np.concatenate([[1.0], X[i]]) @ b
    assert np.allclose(fast, slow, atol=1e-10), "hat-matrix LOO must equal brute force"
    print("scoreboard.demo OK: closed-form LOO exact")


if __name__ == "__main__":
    if sys.argv[1:2] == ["demo"]:
        demo()
    else:
        sys.exit(main())
