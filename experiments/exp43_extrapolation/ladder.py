"""exp43 — the extrapolation ladder: fit low-z scopes, judge everything
the fit never saw, for BOTH kernels (1ch-mof and 2ch-exp).

Fits are the exp40 protocol (joint plain loss, two warm starts,
parent-side penalties) on the exp38 stage-2 structures unchanged; the
z04/z10/z15/z20 rungs reuse recorded thetas where they exist (README
table). The report evaluates every (variant, scope) at ALL five epochs —
extrapolated epochs starred — plus the differential-deposition pairs,
and draws the extrapolation figure.

Run: PYTHONPATH=. uv run python experiments/exp43_extrapolation/ladder.py \
    {demo|fit|report} [--dev]
"""
import importlib.util
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
E38_DIR = ROOT / "experiments/exp38_deposit_rethink"
E40_DIR = ROOT / "experiments/exp40_epoch_objective"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
KS_ALL = [0, 1, 2, 3, 4]
SCOPES = {"z04": [0], "z07": [0, 1], "z10": [0, 1, 2],
          "z15": [0, 1, 2, 3], "z20": [0, 1, 2, 3, 4]}
VARIANTS = ("1ch-mof", "2ch-exp")
NEW_FITS = [("1ch-mof", "z07"), ("2ch-exp", "z07"),
            ("2ch-exp", "z10"), ("2ch-exp", "z15")]
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


def _ladder_npz(tag=""):
    return OUTDIR / f"ladder{tag}.npz"


def recorded_thetas(tag=""):
    """All (variant, scope) -> theta available: reused records + this
    experiment's fits (missing entries absent)."""
    out = {}
    single = np.load(E38_DIR / "outputs/stage3_single.npz")
    multi = np.load(E38_DIR / "outputs/stage2_multiepoch.npz")
    late = np.load(E40_DIR / "outputs/latestart.npz")
    out[("1ch-mof", "z04")] = single["theta_1ch-mof_k0"]
    out[("2ch-exp", "z04")] = single["theta_2ch-exp_k0"]
    out[("1ch-mof", "z10")] = late["theta_z10"]
    out[("1ch-mof", "z15")] = late["theta_z15"]
    out[("1ch-mof", "z20")] = multi["theta_1ch-mof"]
    out[("2ch-exp", "z20")] = multi["theta_2ch-exp"]
    if _ladder_npz(tag).exists():
        d = np.load(_ladder_npz(tag))
        for variant, scope in NEW_FITS:
            key = f"theta_{variant}_{scope}"
            if key in d.files:
                out[(variant, scope)] = d[key]
    return out


# --------------------------------------------------------------------------- #
# variant-aware evaluation helpers (the exp40 report pattern)                  #
# --------------------------------------------------------------------------- #
def bias_table(th, variant, label, fitted_ks):
    """Median per-epoch [M<5 bias, M<10 bias, pinned shape R>5] at ALL
    epochs; extrapolated epochs starred in the printout."""
    e = _W["e"]
    s2 = _W["s2"]
    gals = _W["gals"]
    i5 = int(np.searchsorted(e.R, 5.0))
    i10 = int(np.searchsorted(e.R, 10.0))
    m = e.R > 5.0
    rows = []
    for g in gals:
        cogs = s2.model_cogs(th, g, KS_ALL, variant)
        if cogs is None:
            continue
        per = []
        for c, k in zip(cogs, KS_ALL):
            d = g["data"][k]
            cs = c * (d[-1] / c[-1])
            per.append([c[i5] / d[i5] - 1, c[i10] / d[i10] - 1,
                        np.abs(cs[m] / d[m] - 1).max()])
        rows.append(per)
    arr = np.median(np.array(rows), axis=0)
    print(f"    [{label}] per epoch (M<5 % | M<10 % | pinned shape R>5 %; "
          "* = extrapolated):")
    for j, k in enumerate(KS_ALL):
        star = " " if k in fitted_ks else "*"
        print(f"      z={e.ANCHOR_Z[k]}{star}: {100*arr[j, 0]:+.1f} | "
              f"{100*arr[j, 1]:+.1f} | {100*arr[j, 2]:.1f}")
    return arr


def differential_row(th, variant, label, fitted_ks):
    """The massive-tercile differential pairs; * = pair includes an
    unfitted epoch. Returns the four (f>50, f>100) model pairs."""
    e = _W["e"]
    s2 = _W["s2"]
    gals = _W["gals"]
    data = np.stack([g["data"] for g in gals])
    logms = np.array([g["logms"] for g in gals])
    cogs = np.full((len(gals), 5, len(e.R)), np.nan)
    for i, g in enumerate(gals):
        out = s2.model_cogs(th, g, KS_ALL, variant)
        if out is not None:
            cogs[i] = out
    _, rows_d = e.differential(cogs, data, logms)
    cells, vals = [], []
    for k in range(4):
        star = "" if (k in fitted_ks and (k + 1) in fitted_ks) else "*"
        cells.append(f"z{e.ANCHOR_Z[k+1]}->z{e.ANCHOR_Z[k]}{star}: "
                     f"{rows_d[('data', 2, k)][0]:.2f}/"
                     f"{rows_d[('data', 2, k)][1]:.2f} -> "
                     f"{rows_d[('model', 2, k)][0]:.2f}/"
                     f"{rows_d[('model', 2, k)][1]:.2f}")
        vals.append([rows_d[("model", 2, k)][0], rows_d[("model", 2, k)][1]])
    print(f"    [{label}] differential massive tercile "
          "(data -> model; * = pair extrapolated):")
    print("      " + "  ".join(cells))
    return np.array(vals)


# --------------------------------------------------------------------------- #
# the missing fits                                                             #
# --------------------------------------------------------------------------- #
def _chunk(args):
    p, variant, ks, lo, hi = args
    s2 = _W["s2"]
    return sum(s2.gal_loss(p, g, ks, variant) for g in _W["gals"][lo:hi])


def cmd_fit(dev=False):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    s2 = _W["s2"]
    tag = "_dev" if dev else ""
    scale = 0.15 if dev else 1.0
    workers = max(os.cpu_count() - 2, 2)
    n = len(_W["gals"])
    edges = np.linspace(0, n, workers + 1).astype(int)
    rec = recorded_thetas(tag)
    fitted = dict(np.load(_ladder_npz(tag))) if _ladder_npz(tag).exists() \
        else {}
    OUTDIR.mkdir(exist_ok=True)
    print(f"exp43 ladder fits (n={n}{', DEV' if dev else ''}; joint plain "
          "loss, exp40 protocol):", flush=True)
    with Pool(workers, initializer=_w_init, initargs=(rows,)) as pool:
        for variant, scope in NEW_FITS:
            key = f"theta_{variant}_{scope}"
            if key in fitted:
                print(f"  [{variant} {scope}] already fitted — skip")
                continue
            ks = SCOPES[scope]
            # warm from the widest recorded fit of the same variant + a
            # weaker-envelope nudge (the exp40 start pattern)
            warm = rec[(variant, "z20")]
            nudge = warm.copy()
            nudge[2] = max(warm[2] - 0.5, 0.05)
            maxiter = max(int(5000 * scale), 80)

            def loss(p):
                parts = pool.map(_chunk, [(p, variant, ks, edges[i],
                                           edges[i + 1])
                                          for i in range(workers)])
                return sum(parts) / n + s2.penalty(p, variant)

            best = None
            for p0 in (warm, nudge):
                t0 = time.time()
                r = minimize(loss, p0, method="Nelder-Mead",
                             options=dict(maxiter=maxiter, xatol=3e-4,
                                          fatol=1e-8))
                print(f"  [{variant} {scope}] start: loss {r.fun:.4f} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
                if best is None or r.fun < best.fun:
                    best = r
            th = best.x
            ab = s2._at_bound(th, variant)
            npar = 6 if variant == "1ch-mof" else 5
            print(f"  [{variant} {scope}] params: {np.round(th[:npar], 2)}; "
                  f"at bound: {', '.join(ab) if ab else 'NONE'}")
            bias_table(th, variant, f"{variant} {scope}", ks)
            fitted[key] = th
            fitted[f"loss_{variant}_{scope}"] = best.fun
            np.savez(_ladder_npz(tag), **fitted)
            print(f"  wrote {_ladder_npz(tag)} (+{variant} {scope})",
                  flush=True)


# --------------------------------------------------------------------------- #
# the report: tables + the extrapolation figure                                #
# --------------------------------------------------------------------------- #
def cmd_report(dev=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hongshao.plotting import save_fig, set_style
    set_style()
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    e = _W["e"]
    tag = "_dev" if dev else ""
    rec = recorded_thetas(tag)
    scope_colors = {"z04": "#CC79A7", "z07": "#D55E00", "z10": "#E69F00",
                    "z15": "#009E73", "z20": "#0072B2"}
    met = {}
    diff = {}
    print(f"exp43 ladder report (n={len(_W['gals'])}"
          f"{', DEV' if dev else ''}):")
    for variant in VARIANTS:
        for scope, ks in SCOPES.items():
            if (variant, scope) not in rec:
                print(f"  [{variant} {scope}] MISSING — run `fit` first")
                continue
            th = rec[(variant, scope)]
            print(f"\n  == {variant}, fitted {scope} "
                  f"(epochs {[e.ANCHOR_Z[k] for k in ks]}) ==")
            npar = 6 if variant == "1ch-mof" else 5
            print(f"    params: {np.round(th[:npar], 2)}")
            met[(variant, scope)] = bias_table(th, variant,
                                               f"{variant} {scope}", ks)
            diff[(variant, scope)] = differential_row(
                th, variant, f"{variant} {scope}", ks)
    out = {}
    for (variant, scope), arr in met.items():
        out[f"met_{variant}_{scope}"] = arr
        out[f"diff_{variant}_{scope}"] = diff[(variant, scope)]
    np.savez(OUTDIR / f"ladder_report{tag}.npz", **out)

    # the extrapolation figure: metric vs evaluation epoch, one line per
    # fit scope; open markers = extrapolated epochs
    FIGDIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.6), sharex=True)
    zs = np.array([e.ANCHOR_Z[k] for k in KS_ALL])
    for col, variant in enumerate(VARIANTS):
        for scope, ks in SCOPES.items():
            if (variant, scope) not in met:
                continue
            arr = met[(variant, scope)]
            color = scope_colors[scope]
            in_fit = np.array([k in ks for k in KS_ALL])
            for row, j, flip in ((0, 2, 1.0), (1, 0, 1.0)):
                ax = axes[row, col]
                y = 100.0 * arr[:, j] * flip
                ax.plot(zs, y, "-", color=color, lw=1.5,
                        label=f"fit {scope}" if row == 0 else None)
                ax.plot(zs[in_fit], y[in_fit], "o", color=color, ms=6)
                ax.plot(zs[~in_fit], y[~in_fit], "o", mfc="none",
                        mec=color, ms=6)
        axes[0, col].set_title(f"{variant}"
                               + ("  (adopted)" if col == 0 else
                                  "  (alternative)"))
    for ax in axes[0]:
        ax.set_ylabel(r"pinned shape max$|$rel$|$, $R>5$ kpc [$\%$]")
    for ax in axes[1]:
        ax.set_ylabel(r"$M_\star(<5\,{\rm kpc})$ bias [$\%$]")
        ax.set_xlabel(r"evaluation epoch $z$")
        ax.axhline(0.0, color="0.85", lw=0.8, zorder=0)
    axes[0, 0].legend(fontsize=8, title="fit scope",
                      title_fontsize=8, ncol=2)
    fig.suptitle("the extrapolation ladder — filled markers = fitted "
                 "epochs, open = extrapolated", fontsize=11)
    fig.tight_layout()
    print("\nwrote", save_fig(fig, FIGDIR / f"exp43_ladder{tag}")[0])
    plt.close(fig)


SCOPE_COLORS = {"z04": "#CC79A7", "z07": "#D55E00", "z10": "#E69F00",
                "z15": "#009E73", "z20": "#0072B2"}


def cmd_figures(dev=False):
    """The demonstration figures for the ladder conclusions:
    exp43_differential — the measured differential-deposition curve vs
    every (variant, scope), extrapolated pairs open;
    exp43_residual_profiles — median pinned-shape residual vs radius at
    the two extrapolated target epochs (z=1.5, 2.0) per scope;
    plus the STANDARD QA set for the star rung, the 1ch-mof z07 fit
    evaluated at all five epochs (qa_*_exp43_1ch-mof_z07)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hongshao import qa
    from hongshao.plotting import save_fig, set_style
    set_style()
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    e = _W["e"]
    s2 = _W["s2"]
    gals = _W["gals"]
    tag = "_dev" if dev else ""
    rec = recorded_thetas(tag)
    rep = np.load(OUTDIR / f"ladder_report{tag}.npz")
    data = np.stack([g["data"] for g in gals])
    logms = np.array([g["logms"] for g in gals])
    logmh = np.array([g["logmh"] for g in gals])
    FIGDIR.mkdir(exist_ok=True)

    # (1) the differential-extrapolation figure
    _, rows_dd = e.differential(data, data, logms)
    d50 = [rows_dd[("data", 2, k)][0] for k in range(4)]
    d100 = [rows_dd[("data", 2, k)][1] for k in range(4)]
    x = np.arange(4)
    pair_labels = [rf"z{e.ANCHOR_Z[k+1]:g}$\to$z{e.ANCHOR_Z[k]:g}"
                   for k in range(4)]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2), sharex=True)
    for col, variant in enumerate(VARIANTS):
        for scope, ks in SCOPES.items():
            key = f"diff_{variant}_{scope}"
            if key not in rep.files:
                continue
            vals = rep[key]
            fitted_pair = np.array([(k in ks and (k + 1) in ks)
                                    for k in range(4)])
            for row, j in ((0, 0), (1, 1)):
                ax = axes[row, col]
                color = SCOPE_COLORS[scope]
                ax.plot(x, vals[:, j], "-", color=color, lw=1.4,
                        label=f"fit {scope}" if row == 0 else None)
                ax.plot(x[fitted_pair], vals[fitted_pair, j], "o",
                        color=color, ms=6)
                ax.plot(x[~fitted_pair], vals[~fitted_pair, j], "o",
                        mfc="none", mec=color, ms=6)
        for row, dd, lab in ((0, d50, r"beyond 50 kpc"),
                             (1, d100, r"beyond 100 kpc")):
            axes[row, col].plot(x, dd, "-s", color="0.1", lw=2.2, ms=6,
                                zorder=5,
                                label="measured" if row == 0 else None)
            axes[row, col].set_ylabel("fraction of massive-tercile\n"
                                      rf"growth {lab}")
        axes[0, col].set_title(f"{variant}")
        axes[1, col].set_xticks(x, pair_labels)
        axes[1, col].set_xlabel("epoch pair")
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("differential deposition: measured vs every fit scope "
                 "(open markers = the pair was NOT in the fit)",
                 fontsize=11)
    fig.tight_layout()
    print("wrote", save_fig(fig, FIGDIR / f"exp43_differential{tag}")[0])
    plt.close(fig)

    # (2) median pinned-shape residual vs radius at the extrapolated
    # target epochs
    targets = [3, 4]                                    # z=1.5, 2.0
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2), sharex=True)
    for col, variant in enumerate(VARIANTS):
        med = {}
        for scope in SCOPES:
            if (variant, scope) not in rec:
                continue
            th = rec[(variant, scope)]
            res = []
            for g in gals:
                cogs = s2.model_cogs(th, g, targets, variant)
                if cogs is None:
                    continue
                per = []
                for c, k in zip(cogs, targets):
                    d = g["data"][k]
                    per.append(c * (d[-1] / c[-1]) / d - 1.0)
                res.append(per)
            med[scope] = 100.0 * np.median(np.array(res), axis=0)
        for row, k in enumerate(targets):
            ax = axes[row, col]
            for scope, ks in SCOPES.items():
                if scope not in med:
                    continue
                ls = "-" if k in ks else "--"
                ax.semilogx(e.R, med[scope][row], ls,
                            color=SCOPE_COLORS[scope], lw=1.6,
                            label=f"fit {scope}" if (row, col) == (0, 0)
                            else None)
            ax.axhline(0.0, color="0.8", lw=0.9, zorder=0)
            ax.text(0.03, 0.06, rf"evaluated at $z={e.ANCHOR_Z[k]:g}$"
                    + (" (extrapolated for open scopes)" if col == 0
                       else ""),
                    transform=ax.transAxes, fontsize=8, color="0.35")
            if col == 0:
                ax.set_ylabel(r"median pinned residual [$\%$]")
        axes[0, col].set_title(f"{variant}")
        axes[1, col].set_xlabel(r"$R$ [kpc]")
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("where the extrapolation error lives: 148-pinned "
                 "median residual vs radius (dashed = epoch not in the "
                 "fit)", fontsize=11)
    fig.tight_layout()
    print("wrote",
          save_fig(fig, FIGDIR / f"exp43_residual_profiles{tag}")[0])
    plt.close(fig)

    # (3) the standard QA set for the star rung: 1ch-mof fitted at z07,
    # evaluated at ALL five epochs (three of them extrapolation)
    th = rec[("1ch-mof", "z07")]
    cogs = np.full((len(gals), 5, len(e.R)), np.nan)
    for i, g in enumerate(gals):
        out = s2.model_cogs(th, g, KS_ALL, "1ch-mof")
        if out is not None:
            cogs[i] = out
    qa.evaluate(cogs, data, e.R, [e.ANCHOR_Z[k] for k in KS_ALL],
                name=f"exp43_1ch-mof_z07{tag}", figdir=FIGDIR,
                figures=True, bin_by=logmh, bin_label="logMh")
    print(f"wrote the standard QA set for the z07 rung to {FIGDIR}",
          flush=True)


def demo():
    rows = np.load(POP_NPZ)["dev100"][:10]
    _w_init(rows)
    s2 = _W["s2"]
    rec = recorded_thetas()
    # (1) every reusable rung loads with the right layout
    for (variant, scope), th in rec.items():
        want = 12 if variant == "1ch-mof" else 16
        assert th.shape == (want,), (variant, scope, th.shape)
    assert ("1ch-mof", "z04") in rec and ("2ch-exp", "z20") in rec
    # (2) both variants run finite/monotone on every recorded theta
    g = _W["gals"][0]
    for (variant, scope), th in rec.items():
        cogs = s2.model_cogs(th, g, KS_ALL, variant)
        assert cogs is not None, (variant, scope)
        for c in cogs:
            assert np.isfinite(c).all() and np.all(np.diff(c) > -1e-9)
    # (3) the subset-loss identity holds for BOTH variants (the exp40
    # check, extended to 2ch-exp)
    for variant in VARIANTS:
        th = rec[(variant, "z20")]
        for ks in ([0], [0, 1]):
            lo = s2.gal_loss(th, g, ks, variant)
            cogs = s2.model_cogs(th, g, ks, variant)
            manual = np.mean([np.sqrt(np.mean(
                ((c - g["data"][k]) / g["data"][k]) ** 2))
                for c, k in zip(cogs, ks)])
            assert abs(lo - manual) < 1e-12, (variant, ks)
    # (4) the recorded z04-only 1ch-mof theta shows its documented
    # signature: no migration envelope (q ~ 0)
    assert abs(rec[("1ch-mof", "z04")][2]) < 0.05
    print("exp43 demo OK: all reusable ladder rungs load (12p/16p), both "
          "variants run finite/monotone on every rung, the subset loss "
          "decomposes exactly for 1ch and 2ch, and the z04-only fit shows "
          "its documented q~0 signature")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    dev = "--dev" in sys.argv
    if cmd == "demo":
        demo()
    elif cmd == "fit":
        cmd_fit(dev)
    elif cmd == "report":
        cmd_report(dev)
    elif cmd == "figures":
        cmd_figures(dev)
    else:
        sys.exit(f"unknown subcommand {cmd!r}")
