"""exp44 — "calibrate low, predict high": per-object low-z anchoring.

Per galaxy, ONE coordinate of the adopted 1ch-mof theta is refit
(bounded scalar inside the physical box — the exp41 stage-0 protocol)
against ANCHOR epochs only; everything else stays at the official
population theta. The model is then evaluated at ALL five epochs, and
the science lives entirely at the epochs that were NOT anchored.

Controls: the population theta (no anchoring), an ORACLE anchored on
all five epochs (the ceiling of one per-galaxy number), and a SHUFFLE
null (each galaxy given another galaxy's fitted deviation — identical
marginal distribution, no galaxy-specific pairing). Anchoring only
means something if it beats SHUFFLE at the prediction epochs.

Run: PYTHONPATH=. uv run python experiments/exp44_lowz_anchoring/\
anchor.py {demo|fit|report|physics|figures} [--dev]
"""
import importlib.util
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
E38_DIR = ROOT / "experiments/exp38_deposit_rethink"
E40_DIR = ROOT / "experiments/exp40_epoch_objective"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
KS_ALL = [0, 1, 2, 3, 4]
ANCHORS = {"z04": [0], "z07": [0, 1], "all": [0, 1, 2, 3, 4]}
AXES = {"sig": 4, "rc": 0}                       # index into the base6
NAMES = ["log_rc", "g", "q", "mu", "sig", "gamma"]
LO6 = np.array([1.0, 0.0, 0.0, 0.0, 0.05, 1.06])
HI6 = np.array([3.0, 6.0, 3.0, 3.0, 2.0, 6.0])
APER_KPC = (10.0, 30.0, 50.0, 100.0)             # what a survey delivers
# arm -> (axis, anchor scope, observable); "oracle" is the all-epoch ceiling
ARMS = {
    "sig_z04_prof": ("sig", "z04", "prof"),
    "sig_z07_prof": ("sig", "z07", "prof"),
    "sig_z04_aper": ("sig", "z04", "aper"),
    "sig_z07_aper": ("sig", "z07", "aper"),
    "rc_z07_prof": ("rc", "z07", "prof"),
    "sig_oracle_prof": ("sig", "all", "prof"),
}
_W = {}


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _w_init(rows):
    s2 = _load_by_path("exp38_stage2", E38_DIR / "stage2_multiepoch.py")
    s2._w_init(rows)
    _W.update(s2._W)
    _W["s2"] = s2
    _W["th0"] = np.load(E40_DIR / "outputs/latestart.npz")["theta_z15"]
    e = _W["e"]
    _W["i_aper"] = [int(np.searchsorted(e.R, a)) for a in APER_KPC]


# --------------------------------------------------------------------------- #
# the two anchoring observables                                                #
# --------------------------------------------------------------------------- #
def gal_loss_aper(th, g, ks):
    """Relative-RMS error on the kpc APERTURE masses only — the
    observationally honest anchor (the inner profile shape is not
    trusted; the aperture masses are). Model CoGs are M(<500)-pinned,
    so these are absolute predictions, exactly what a survey compares."""
    s2 = _W["s2"]
    idx = _W["i_aper"]
    cogs = s2.model_cogs(th, g, ks, "1ch-mof")
    if cogs is None:
        return 4.0
    out = []
    for c, k in zip(cogs, ks):
        d = g["data"][k]
        rel = np.array([c[i] / d[i] - 1.0 for i in idx])
        out.append(np.sqrt(np.mean(rel ** 2)))
    return float(np.mean(out))


def anchor_loss(th, g, ks, observable):
    if observable == "aper":
        return gal_loss_aper(th, g, ks)
    return _W["s2"].gal_loss(th, g, ks, "1ch-mof")


# --------------------------------------------------------------------------- #
# the per-galaxy anchoring fit                                                 #
# --------------------------------------------------------------------------- #
def _fit_one(args):
    """One galaxy, one arm: bounded scalar refit of the arm's axis
    against the arm's anchor epochs under the arm's observable."""
    gi, arm = args
    axis, scope, observable = ARMS[arm]
    j = AXES[axis]
    ks = ANCHORS[scope]
    g = _W["gals"][gi]
    th0 = _W["th0"]
    l_base = anchor_loss(th0, g, ks, observable)

    def f(x):
        th = th0.copy()
        th[j] = x
        return anchor_loss(th, g, ks, observable)

    r = minimize_scalar(f, bounds=(LO6[j], HI6[j]), method="bounded",
                        options=dict(maxiter=80, xatol=1e-4))
    val, lo = (r.x, r.fun) if r.fun < l_base else (th0[j], l_base)
    return gi, g["row"], val, lo, l_base


def cmd_fit(dev=False):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    tag = "_dev" if dev else ""
    n = len(_W["gals"])
    workers = max(os.cpu_count() - 2, 2)
    OUTDIR.mkdir(exist_ok=True)
    out = {}
    print(f"exp44 anchoring fits (n={n}{', DEV' if dev else ''}; one "
          "per-galaxy coordinate, official theta elsewhere):", flush=True)
    with Pool(workers, initializer=_w_init, initargs=(rows,)) as pool:
        for arm in ARMS:
            t0 = time.time()
            res = pool.map(_fit_one, [(i, arm) for i in range(n)])
            gi, row, val, lo, l_base = map(np.array, zip(*res))
            order = np.argsort(gi)
            out[f"val_{arm}"] = val[order]
            out[f"row_{arm}"] = row[order]
            out[f"gain_{arm}"] = (l_base - lo)[order] / np.maximum(
                l_base[order], 1e-12)
            axis, scope, observable = ARMS[arm]
            j = AXES[axis]
            d = val[order] - _W["th0"][j]
            print(f"  [{arm}] anchor-epoch gain {100*np.median(out[f'gain_{arm}']):.1f}"
                  f"%  delta {NAMES[j]} 16/50/84: "
                  f"{np.percentile(d, 16):+.3f}/{np.median(d):+.3f}/"
                  f"{np.percentile(d, 84):+.3f}  "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
    np.savez(OUTDIR / f"anchored{tag}.npz", th0=_W["th0"], **out)
    print(f"  wrote {OUTDIR / f'anchored{tag}.npz'}")


# --------------------------------------------------------------------------- #
# evaluation at ALL epochs                                                     #
# --------------------------------------------------------------------------- #
def _eval_one(args):
    """One galaxy under a given per-galaxy value: per-epoch
    [pinned shape R>5, M(<5) bias, M(<10) bias] at all five epochs."""
    gi, j, val = args
    s2 = _W["s2"]
    e = _W["e"]
    g = _W["gals"][gi]
    th = _W["th0"].copy()
    if j is not None:
        th[j] = val
    i5 = int(np.searchsorted(e.R, 5.0))
    i10 = int(np.searchsorted(e.R, 10.0))
    m = e.R > 5.0
    cogs = s2.model_cogs(th, g, KS_ALL, "1ch-mof")
    if cogs is None:
        return gi, np.full((5, 3), np.nan)
    met = []
    for c, k in zip(cogs, KS_ALL):
        d = g["data"][k]
        cs = c * (d[-1] / c[-1])
        met.append([np.abs(cs[m] / d[m] - 1).max(),
                    c[i5] / d[i5] - 1, c[i10] / d[i10] - 1])
    return gi, np.array(met)


def _evaluate(pool, n, j, vals):
    """Median per-epoch metrics over the sample for one configuration."""
    jobs = [(i, j, None if vals is None else vals[i]) for i in range(n)]
    res = pool.map(_eval_one, jobs)
    arr = np.full((n, 5, 3), np.nan)
    for gi, met in res:
        arr[gi] = met
    return np.nanmedian(arr, axis=0)


def _print_row(label, arr, anchor_ks, e):
    cells = []
    for k in range(5):
        star = "" if k in anchor_ks else "*"
        cells.append(f"z{e.ANCHOR_Z[k]}{star}: {100*arr[k, 0]:.1f}/"
                     f"{100*arr[k, 1]:+.1f}")
    print(f"    {label:>16s}  " + "  ".join(cells))


def cmd_report(dev=False):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    tag = "_dev" if dev else ""
    d = np.load(OUTDIR / f"anchored{tag}.npz")
    e = _W["e"]
    n = len(_W["gals"])
    workers = max(os.cpu_count() - 2, 2)
    rng = np.random.default_rng(44)
    store = {}
    print(f"exp44 report (n={n}{', DEV' if dev else ''}) — per epoch: "
          "pinned shape R>5 [%] / M(<5) bias [%]; * = NOT anchored "
          "(the science lives there):", flush=True)
    with Pool(workers, initializer=_w_init, initargs=(rows,)) as pool:
        base = _evaluate(pool, n, None, None)
        store["met_pop"] = base
        print("\n  == baseline ==")
        _print_row("pop (no anchor)", base, KS_ALL, e)
        for arm, (axis, scope, observable) in ARMS.items():
            j = AXES[axis]
            vals = d[f"val_{arm}"]
            anchor_ks = ANCHORS[scope]
            arr = _evaluate(pool, n, j, vals)
            shuf = _evaluate(pool, n, j, vals[rng.permutation(n)])
            store[f"met_{arm}"] = arr
            store[f"met_shuffle_{arm}"] = shuf
            print(f"\n  == {arm} (axis {NAMES[j]}, anchor "
                  f"{[e.ANCHOR_Z[k] for k in anchor_ks]}, {observable}) ==")
            _print_row("anchored", arr, anchor_ks, e)
            _print_row("SHUFFLE null", shuf, anchor_ks, e)
            gain = 100 * (base[:, 0] - arr[:, 0])
            gain_s = 100 * (base[:, 0] - shuf[:, 0])
            pred = [k for k in range(5) if k not in anchor_ks]
            print(f"    shape gain vs pop at PREDICTION epochs: "
                  + "  ".join(f"z{e.ANCHOR_Z[k]}: {gain[k]:+.1f} "
                              f"(shuffle {gain_s[k]:+.1f})" for k in pred))
    np.savez(OUTDIR / f"report{tag}.npz", **store)
    print(f"\n  wrote {OUTDIR / f'report{tag}.npz'}")


def cmd_physics(dev=False, arm="sig_z07_aper"):
    """Differential + outskirt tests on an anchored population — per-object
    freedom must not break what objectives broke in exp39."""
    from hongshao.profile_emulator import density_from_cog
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    tag = "_dev" if dev else ""
    d = np.load(OUTDIR / f"anchored{tag}.npz")
    s2, e, gals = _W["s2"], _W["e"], _W["gals"]
    j = AXES[ARMS[arm][0]]
    vals = d[f"val_{arm}"]
    n = len(gals)
    cogs = np.full((n, 5, len(e.R)), np.nan)
    for i, g in enumerate(gals):
        th = _W["th0"].copy()
        th[j] = vals[i]
        out = s2.model_cogs(th, g, KS_ALL, "1ch-mof")
        if out is not None:
            cogs[i] = out
    data = np.stack([g["data"] for g in gals])
    logms = np.array([g["logms"] for g in gals])
    print(f"exp44 physics on the anchored population [{arm}] (marks: data "
          "0.37/0.11; population theta 0.40/0.13, outskirt T1 "
          "+0.025/+0.028):")
    _, rows_d = e.differential(cogs, data, logms)
    print("    differential massive tercile: " + "  ".join(
        f"z{e.ANCHOR_Z[k+1]}->z{e.ANCHOR_Z[k]}: "
        f"{rows_d[('data', 2, k)][0]:.2f}/{rows_d[('data', 2, k)][1]:.2f} -> "
        f"{rows_d[('model', 2, k)][0]:.2f}/{rows_d[('model', 2, k)][1]:.2f}"
        for k in range(4)))
    c0 = cogs[:, 0]
    ok = np.isfinite(c0).all(1) & (c0 > 0).all(1)
    ls_d, mid = density_from_cog(np.log10(data[:, 0]), e.R)
    ls_m = np.full_like(ls_d, np.nan)
    ls_m[ok] = density_from_cog(np.log10(c0[ok]), e.R)[0]
    dl = ls_m - ls_d
    bands = ((mid >= 30.0) & (mid < 60.0), (mid >= 60.0) & (mid <= 148.0))
    edges = np.quantile(logms, [0, 1 / 3, 2 / 3, 1])
    cells = []
    for b in range(3):
        sel = ok & (logms >= edges[b]) & (logms <= edges[b + 1] + 1e-9)
        cells.append(" ".join(f"{np.nanmedian(dl[np.ix_(sel, bm)]):+.3f}"
                              for bm in bands))
    print("    overshoot terciles (z=0.4, [30-60 / 60-148 kpc]): "
          + "  |  ".join(f"T{b+1} {c}" for b, c in enumerate(cells)))


def cmd_figures(dev=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hongshao.plotting import save_fig, set_style
    set_style()
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    tag = "_dev" if dev else ""
    r = np.load(OUTDIR / f"report{tag}.npz")
    e = _W["e"]
    zs = np.array([e.ANCHOR_Z[k] for k in KS_ALL])
    show = [("sig_z04_aper", "#CC79A7"), ("sig_z07_aper", "#D55E00"),
            ("sig_z07_prof", "#E69F00"), ("sig_oracle_prof", "#009E73")]
    FIGDIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    for ax, col, ylab in ((axes[0], 0, r"pinned shape max$|$rel$|$, "
                           r"$R>5$ kpc [$\%$]"),
                          (axes[1], 1, r"$M_\star(<5\,{\rm kpc})$ bias "
                           r"[$\%$]")):
        ax.plot(zs, 100 * r["met_pop"][:, col], "-k", lw=2.2, ms=6,
                marker="s", label="population theta (no anchor)")
        for arm, color in show:
            if f"met_{arm}" not in r.files:
                continue
            anchor_ks = ANCHORS[ARMS[arm][1]]
            y = 100 * r[f"met_{arm}"][:, col]
            in_a = np.array([k in anchor_ks for k in KS_ALL])
            ax.plot(zs, y, "-", color=color, lw=1.6, label=arm)
            ax.plot(zs[in_a], y[in_a], "o", color=color, ms=6)
            ax.plot(zs[~in_a], y[~in_a], "o", mfc="none", mec=color, ms=6)
            ys = 100 * r[f"met_shuffle_{arm}"][:, col]
            ax.plot(zs, ys, ":", color=color, lw=1.1, alpha=0.8)
        ax.set(xlabel=r"evaluation epoch $z$", ylabel=ylab)
    axes[1].axhline(0.0, color="0.85", lw=0.8, zorder=0)
    axes[0].legend(fontsize=7.5)
    fig.suptitle("calibrate low, predict high — filled = anchored epochs, "
                 "open = predicted, dotted = shuffle null", fontsize=11)
    fig.tight_layout()
    print("wrote", save_fig(fig, FIGDIR / f"exp44_transfer{tag}")[0])
    plt.close(fig)


def demo():
    rows = np.load(POP_NPZ)["dev100"][:10]
    _w_init(rows)
    s2, gals, th0 = _W["s2"], _W["gals"], _W["th0"]
    g = gals[0]
    # (1) both observables are finite on the official theta, and the
    # aperture loss really only sees the four aperture radii
    for ks in ([0], [0, 1]):
        assert np.isfinite(anchor_loss(th0, g, ks, "prof"))
        assert np.isfinite(anchor_loss(th0, g, ks, "aper"))
    la = gal_loss_aper(th0, g, [0])
    cogs = s2.model_cogs(th0, g, [0], "1ch-mof")[0]
    manual = np.sqrt(np.mean([(cogs[i] / g["data"][0][i] - 1.0) ** 2
                              for i in _W["i_aper"]]))
    assert abs(la - manual) < 1e-12, (la, manual)
    # (2) the anchoring fit never makes the anchor loss worse and stays
    # inside the physical box
    for arm in ARMS:
        axis, scope, observable = ARMS[arm]
        j = AXES[axis]
        _, _, val, lo, l_base = _fit_one((0, arm))
        assert lo <= l_base + 1e-12, (arm, lo, l_base)
        assert LO6[j] - 1e-9 <= val <= HI6[j] + 1e-9, (arm, val)
    # (3) evaluation with j=None reproduces the population model exactly
    _, met_pop = _eval_one((0, None, None))
    _, met_same = _eval_one((0, AXES["sig"], th0[AXES["sig"]]))
    assert np.allclose(met_pop, met_same, atol=1e-12)
    # (4) anchoring improves the ANCHOR LOSS at its own epoch — but that
    # does NOT guarantee the EVALUATION metric improves there: the
    # anchor losses are relative RMS (all radii / the four apertures),
    # the readout is the 148-pinned worst-radius shape over R>5 kpc.
    # Two different objectives (the exp29 "a floor in one metric is not a
    # floor in another" lesson); which way the readout moves is a
    # MEASUREMENT this experiment reports, not an invariant to assert.
    _, _, val04, lo04, base04 = _fit_one((0, "sig_z04_prof"))
    assert lo04 <= base04 + 1e-12
    assert not np.isclose(val04, th0[AXES["sig"]]), \
        "the scalar refit did not move — anchoring would be a no-op"
    th_a = th0.copy()
    th_a[AXES["sig"]] = val04
    assert anchor_loss(th_a, g, [0], "prof") <= \
        anchor_loss(th0, g, [0], "prof") + 1e-12
    print("exp44 demo OK: both anchoring observables finite and the "
          "aperture loss uses exactly the four survey apertures; every "
          "arm's scalar refit is non-worsening and in-box; the "
          "unanchored evaluation reproduces the population model "
          "exactly; anchoring improves its own ANCHOR LOSS (the readout "
          "metric is a different objective — measured, not assumed)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    dev = "--dev" in sys.argv
    if cmd == "demo":
        demo()
    elif cmd == "fit":
        cmd_fit(dev)
    elif cmd == "report":
        cmd_report(dev)
    elif cmd == "physics":
        cmd_physics(dev, sys.argv[sys.argv.index("--arm") + 1]
                    if "--arm" in sys.argv else "sig_z07_aper")
    elif cmd == "figures":
        cmd_figures(dev)
    else:
        sys.exit(f"unknown subcommand {cmd!r}")
