"""exp31 — the honest forward scoreboard: transport emulator vs regression references.

Every model predicts full 5-epoch CoGs (n,5,24) for UNSEEN galaxies (leave-galaxy-out,
halo-only inputs) and is scored by the one standardized harness (hongshao.qa):

  transport-real     phase-4 universal-theta e2e (real MAH; SHMR amplitude)
  transport-diffmah  same, DiffMAH-parameter input (the differentiable config)
  logmh-only         per-radius/epoch LOGO regression log M*(<R,z_k) <- logMh(z_k):
                     the classic SHMR generalized to every aperture — the halo-mass-
                     only null (no assembly history, no secondaries)
  direct             exp08-pattern: same + halo-only secondaries (c200c, t50, fz2)
                     — the tier-1/2 statistical ceiling; no consistency, no physics

The regression models predict each radius independently, so they compete on ALL
tiers (their CoGs need not be monotonic — they are statistical predictors).

Run: PYTHONPATH=. uv run python experiments/exp31_forward_scoreboard/run.py [--refit]
Demo: ... run.py demo
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP30 = ROOT / "experiments" / "exp30_transport_kernel"
EXP29 = ROOT / "experiments" / "exp29_outer_deposit"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP29))
sys.path.insert(0, str(EXP30))
import pop_forward as pf                                                             # noqa: E402
from run import ANCHOR_Z                                                             # noqa: E402
from hongshao import qa                                                              # noqa: E402
from hongshao.tng_data import COG_RAD_KPC                                            # noqa: E402
from hongshao.plotting import set_style, save_fig                                    # noqa: E402

set_style()
FIGDIR, OUTDIR = HERE / "figures", HERE / "outputs"
OUT_NPZ = OUTDIR / "scoreboard.npz"
R = COG_RAD_KPC
MODELS = ["transport-real", "transport-diffmah", "logmh-only", "direct"]
COLORS = {"transport-real": "#CC3377", "transport-diffmah": "#D55E00",
          "logmh-only": "0.45", "direct": "#0072B2"}


def logo_regress(X, y):
    """Leave-one-out linear-regression predictions (n,) for y from features X (n,f)."""
    n = len(y)
    pred = np.empty(n)
    for i in range(n):
        m = np.arange(n) != i
        A = np.column_stack([np.ones(int(m.sum())), X[m]])
        coef, *_ = np.linalg.lstsq(A, y[m], rcond=None)
        pred[i] = np.concatenate([[1.0], X[i]]) @ coef
    return pred


def regression_cogs(features_by_epoch, data):
    """Per-radius/epoch LOGO regression CoGs. features_by_epoch: list of 5 (n,f)."""
    n = data.shape[0]
    cogs = np.empty((n, 5, len(R)))
    for k in range(5):
        for r in range(len(R)):
            cogs[:, k, r] = 10.0 ** logo_regress(features_by_epoch[k],
                                                 np.log10(data[:, k, r]))
    return cogs


def transport_e2e(d4, config, data):
    """Phase-4 LOGO universal-theta CoGs rescaled from data-pinned to SHMR amplitude."""
    scale = 10.0 ** d4[config + "_shmr_pred"] / data[:, :, -1]
    return d4[config + "_cogs_univ"] * scale[:, :, None]


def compute():
    d4 = np.load(EXP30 / "outputs" / "pop_forward.npz")
    dp = np.load(EXP30 / "outputs" / "param_emulator.npz")
    assert list(d4["real_index"]) == list(dp["index"]) == list(d4["diffmah_index"])
    data = dp["data"]
    gals = pf.load_gals("real", len(dp["index"]))
    logmh_zk = np.array([g["logmh_zk"] for g in gals])
    sec = np.array([[g["c200c"], g["t50"], g["fz2"]] for g in gals])
    feats_shmr = [logmh_zk[:, [k]] for k in range(5)]
    feats_direct = [np.column_stack([logmh_zk[:, k], sec]) for k in range(5)]
    cogs = {"transport-real": transport_e2e(d4, "real", data),
            "transport-diffmah": transport_e2e(d4, "diffmah", data),
            "logmh-only": regression_cogs(feats_shmr, data),
            "direct": regression_cogs(feats_direct, data)}
    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, index=dp["index"], data=data,
             **{m.replace("-", "_"): cogs[m] for m in MODELS})
    print(f"wrote {OUT_NPZ}  (n={len(data)})")
    return np.load(OUT_NPZ)


def group_dex(res, prefix, kind):
    """Median-over-quantities dex scatter per epoch for one bin group (5,)."""
    sel = [k for k in res["keys"] if k.startswith(prefix) and kind(k)]
    return np.median([qa.dex_scatter(res["model"][k], res["truth"][k])
                      for k in sel], axis=0)


def main():
    refit = "--refit" in sys.argv
    d = compute() if (refit or not OUT_NPZ.exists()) else np.load(OUT_NPZ)
    data = d["data"]

    results = {}
    for m in MODELS:
        # full standard QA; figures only for the two headline models
        figs = m in ("transport-diffmah", "direct")
        results[m] = qa.evaluate(d[m.replace("-", "_")], data, R, ANCHOR_Z, name=m,
                                 figdir=FIGDIR, verbose=True, figures=figs)

    is_aper = lambda k: "(<" in k
    is_ann = lambda k: "-" in k.split(":")[1]
    is_env = lambda k: "(>" in k
    print("\n=== SCOREBOARD (LOGO, n=%d) — median dex scatter per epoch ===" % len(data))
    for prefix, unit in (("kpc:", "kpc"), ("Re:", "Re")):
        print(f"\n  [{unit} bins]")
        print(f"    {'model':>18s} {'group':>10s} | "
              + " | ".join(f"z={z}".rjust(6) for z in ANCHOR_Z))
        for m in MODELS:
            for gname, kind in (("apertures", is_aper), ("annuli", is_ann),
                                ("envelopes", is_env)):
                g = group_dex(results[m], prefix, kind)
                print(f"    {m:>18s} {gname:>10s} | "
                      + " | ".join(f"{v:6.3f}" for v in g))
    print("\n    profile max|rel| median, epoch-avg (all R | R>5 kpc):")
    for m in MODELS:
        ma = 100 * np.nanmean(np.nanmedian(results[m]["mr_all"], axis=0))
        mo = 100 * np.nanmean(np.nanmedian(results[m]["mr_out"], axis=0))
        print(f"    {m:>18s} : {ma:5.1f}% | {mo:5.1f}%")
    print("\n    plane fidelity, M(<30) vs M(50-100): |model-truth| relation scatter,"
          " epoch-avg dex:")
    for m in MODELS:
        st = results[m]["planes"][("kpc:M(<30)", "kpc:M(50-100)")]
        dsc = np.nanmean([abs(mo["scatter"] - t["scatter"]) for t, mo in st])
        print(f"    {m:>18s} : {dsc:.3f}")
    _figure(results)


def _figure(results):
    """Scoreboard: dex scatter per quantity, one line per model, panel per epoch;
    top row = fixed-kpc bins, bottom row = R_half-relative bins."""
    fig, axes = plt.subplots(2, 5, figsize=(19, 9.0), sharey="row")
    for row, prefix in enumerate(("kpc:", "Re:")):
        keys = [k for k in results[MODELS[0]]["keys"] if k.startswith(prefix)]
        x = np.arange(len(keys))
        for j in range(5):
            ax = axes[row, j]
            for m in MODELS:
                sc = [qa.dex_scatter(results[m]["model"][k][:, j],
                                     results[m]["truth"][k][:, j]) for k in keys]
                ax.plot(x, sc, "o-", c=COLORS[m], lw=1.6, ms=4, label=m)
            ax.set_xticks(x)
            ax.set_xticklabels([k.split(":")[1] for k in keys], rotation=60, fontsize=7)
            if row == 0:
                ax.set(title=f"z={ANCHOR_Z[j]}")
            ax.grid(alpha=0.25, axis="y")
        axes[row, 0].set(ylabel=f"LOGO dex scatter ({prefix[:-1]} bins)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("exp31 scoreboard — halo-only LOGO dex scatter per mass quantity "
                 "(top: fixed kpc; bottom: $R_{\\rm half}$ units)", fontsize=12)
    fig.tight_layout()
    print("\nwrote", save_fig(fig, FIGDIR / "exp31_scoreboard")[0])


def demo():
    """Self-check: LOGO regression exact on a noiseless linear relation; e2e rescale
    pins the transport CoG total to the SHMR-predicted mass."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 2))
    y = 1.5 + 2.0 * X[:, 0] - 0.7 * X[:, 1]
    assert np.allclose(logo_regress(X, y), y, atol=1e-9), "LOGO must be exact"
    d4 = np.load(EXP30 / "outputs" / "pop_forward.npz")
    dp = np.load(EXP30 / "outputs" / "param_emulator.npz")
    cogs = transport_e2e(d4, "real", dp["data"])
    want = 10.0 ** d4["real_shmr_pred"]
    ok = np.isfinite(cogs[:, :, -1])
    assert np.allclose(cogs[:, :, -1][ok], want[ok], rtol=1e-9), \
        "e2e total must equal the SHMR prediction"
    print("run.demo OK: exact LOGO regression; e2e totals = SHMR prediction")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        sys.exit(main())
