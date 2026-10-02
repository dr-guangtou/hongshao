"""exp87 — the figures, regenerated from saved artifacts only (the scoreboard,
the cells, `extras.npz`, the history files, the lockbox, the symbolic-regression
verdicts). Written to this experiment's `figures/qa/`.

  exp87_truncation   why the selection cut must be in the likelihood: the
                     scatter plot with the three mean curves, the calibration
                     read in bins of the prediction, the scores
  exp87_ladder       the method ladder for halo mass at z = 0.4 and the same
                     four rungs at every epoch, both populations
  exp87_radius       where in the profile the halo-mass information sits
  exp87_assembly     concentration and formation time with and without the
                     true halo mass; the accretion history's error by lookback
  exp87_calibration  predicted against true, the PIT, development vs lockbox
  exp87_symbolic     the symbolic-regression verdicts per cell and stage

Run:  PYTHONPATH=. uv run python -u experiments/exp87_inverse_halo/figures.py [name ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C                                       # noqa: E402
import harness as HN                                     # noqa: E402
import heads as H                                        # noqa: E402
import report as R                                       # noqa: E402
import scoring as S                                      # noqa: E402
from hongshao import qa                                  # noqa: E402

FIGDIR = C.FIGDIR
C_GREY, C_BLUE, C_RED, C_GREEN, C_PURPLE, C_ORANGE = "#777777", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"
RUNG_COLOR = {"L0": C_GREY, "L1": C_BLUE, "L2": C_GREEN, "L3": C_ORANGE, "L4": C_PURPLE, "L5": C_RED}


def style():
    plt.rcParams.update({"axes.labelsize": 10, "axes.titlesize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5})


def save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{name}.{ext}", dpi=150)
    plt.close(fig)
    print(f"wrote {FIGDIR / (name + '.png')}")


def truncation():
    z = np.load(C.OUTDIR / "extras.npz")
    js = json.loads((C.OUTDIR / "extras.json").read_text())
    used = z["fold"] >= 0
    x, y = z["mtot"][used], z["y"][used]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.3))
    ax = axes[0]
    ax.scatter(x, y, s=4, color="#444444", alpha=0.25, rasterized=True)
    fit_t = H.TruncLinear(C.PARENT_CUT).fit(x[:, None], y)
    fit_n = H.TruncLinear(None).fit(x[:, None], y)
    xs = np.linspace(x.min(), x.max(), 200)
    m_t, s_t = fit_t.latent(xs[:, None])
    ax.plot(xs, fit_n.latent(xs[:, None])[0], color=C_BLUE, lw=2, label="ordinary regression (ignores the cut)")
    ax.plot(xs, m_t, color=C_RED, lw=2, ls="--", label="latent mean (truncated likelihood)")
    ax.plot(xs, S.TruncNormal(m_t, s_t, C.PARENT_CUT).mean(), color=C_RED, lw=2, label="its mean above the cut")
    ax.axhline(C.PARENT_CUT, color="k", lw=1.0, ls=":")
    ax.text(x.max(), C.PARENT_CUT - 0.07, "the sample's halo-mass cut", ha="right", fontsize=9)
    ax.set_xlabel(qa._tex(r"log $M_\star$(<148 kpc) [$M_\odot$]"))
    ax.set_ylabel(qa._tex(r"log $M_{200c}$ [$M_\odot$]"))
    ax.set_ylim(12.6, 15.0)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("(a) a cut on the target bends the mean near the cut", fontsize=11)

    ax = axes[1]
    for key, col, lab in (("mtot_naive", C_BLUE, "ordinary normal"), ("mtot_trunc", C_RED, "truncated normal")):
        mean, pit = z[f"{key}_mean"][used], z[f"{key}_pit"][used]
        q = np.quantile(mean, np.linspace(0, 1, 9))
        xc, centre = [], []
        for lo_q, hi_q in zip(q[:-1], q[1:]):
            sel = (mean >= lo_q) & (mean <= hi_q)
            xc.append(np.median(mean[sel]))
            centre.append(np.mean(pit[sel]))
        ax.plot(xc, centre, marker="o", color=col, lw=1.8, label=lab)
    ax.axhline(0.5, color="k", lw=0.8)
    ax.set_xlabel(qa._tex(r"predicted log $M_{200c}$ (bins of the PREDICTION, never of the truth)"))
    ax.set_ylabel("mean PIT in the bin (0.5 = calibrated)")
    ax.set_ylim(0.2, 0.8)
    ax.legend(frameon=False, loc="upper center")
    ax.set_title("(b) the ordinary fit is miscalibrated at every predicted mass", fontsize=11)

    ax = axes[2]
    labels, vals, cols = [], [], []
    for key, name in (("mtot", "stellar mass alone"), ("raw24", "24-point profile")):
        t = js[f"{key}_truncation"]
        for lab, v, c in (("ordinary", t["naive"], C_BLUE), ("ordinary,\nrenormalised", t["double"], C_GREY), ("truncated\nlikelihood", t["trunc"], C_RED)):
            labels.append(lab)
            vals.append(v)
            cols.append(c)
    xpos = np.r_[np.arange(3), np.arange(3) + 3.6]
    ax.bar(xpos, vals, color=cols, width=0.8)
    for xi, v in zip(xpos, vals):
        ax.text(xi, v + 0.001, f"{v:.4f}", ha="center", fontsize=8.5)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.text(1.0, 0.118, "stellar mass alone", ha="center", fontsize=10)
    ax.text(4.6, 0.118, "24-point profile", ha="center", fontsize=10)
    ax.set_ylim(0.06, 0.124)
    ax.set_ylabel("CRPS [dex], out of fold (lower is better)")
    ax.set_title("(c) the predictive family is worth 11" + qa._tex("-") + "17" + qa._pct(), fontsize=11)
    fig.suptitle("exp87: a sample cut in halo mass needs the cut inside the likelihood (parent sample, z = 0.4)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "exp87_truncation")


def ladder():
    g = R.groups()
    rows = g[("parent", 0, "parent", "mh")]
    clim = R.find(rows, "mtot", "climatology")
    pick = [("mtot", "line", "stellar mass\n$M_\\star$(<148)"), ("M(<10)", "line", "$M_\\star$(<10 kpc)"), ("M(>50)", "line", "$M_\\star$(>50 kpc)"),
            ("outer_shell", "line", "outer shell\n132-148 kpc"), ("mass_size", "linear", "mass + sizes"), ("mtot+outer_shell", "linear", "mass +\nouter shell"),
            ("raw24", "linear", "24 cumulative\nmasses"), ("shells24", "ridge", "24 shell\nmasses"), ("shells24", "linear+pca6-quad", "shells +\nquadratic"),
            ("shells24", "linear+gbm", "shells +\nboosting"), ("raw24", "gbm", "boosting on\ncumulative"), ("raw24", "gp", "Gaussian\nprocess"),
            ("shells24", "gen-k24", "generative\ninverse")]
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 5.6), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axes[0]
    for i, (feat, meth, lab) in enumerate(pick):
        r = R.find(rows, feat, meth)
        if r is None:
            continue
        ax.bar(i, r["crps"], color=RUNG_COLOR[r["rung"]], width=0.78)
        ax.text(i, r["crps"] + 0.0006, f"{r['crps']:.4f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(pick)))
    ax.set_xticklabels([qa._tex(p[2]) for p in pick], fontsize=7.6, rotation=28, ha="right", rotation_mode="anchor")
    ax.set_ylim(0.070, 0.118)
    ax.set_ylabel("CRPS [dex], out of fold")
    ax.set_title(f"(a) halo mass at z = 0.4, the complete parent sample (knowing nothing: {clim['crps']:.3f})", fontsize=11)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=RUNG_COLOR[k], label=v) for k, v in (("L0", "one number"), ("L1", "linear"), ("L2", "curvature"),
                                                                       ("L3", "nonparametric"), ("L5", "generative inverse"))],
              frameon=False, ncol=5, loc="upper right", fontsize=8)

    ax = axes[1]
    series = [("mtot", "line", C_GREY, "stellar mass alone"), ("mtot+outer_shell", "linear", C_GREEN, "mass + outer shell"),
              ("shells24", "ridge", C_BLUE, "24 shell masses, linear"), ("shells24", "linear+pca6-quad", C_RED, "shells + quadratic")]
    for pop, ls, mk in (("asis", "-", "o"), ("complete", "--", "s")):
        for feat, meth, col, lab in series:
            xs, ys = [], []
            for k in range(5):
                key = ("parent", 0, "parent", "mh") if (k == 0 and pop == "asis") else ("curated", k, pop if k else "complete", "mh")
                r = R.find(g.get(key, []), feat, meth)
                if r is not None:
                    xs.append(C.ANCHOR_Z[k])
                    ys.append(r["crps"])
            ax.plot(xs, ys, marker=mk, ls=ls, color=col, lw=1.6, ms=5, mfc=col if pop == "asis" else "white",
                    label=lab if pop == "asis" else None)
    ax.set_xlabel("redshift")
    ax.set_ylabel("CRPS [dex], out of fold")
    ax.set_title("(b) every epoch: progenitors as they are (solid), complete above the cut (dashed)", fontsize=10.5)
    ax.legend(frameon=False, loc="upper left")
    fig.suptitle("exp87: predicting halo mass from the stellar curve of growth", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "exp87_ladder")


def radius():
    z = np.load(C.OUTDIR / "extras.npz")
    r = z["radii"]
    fig, ax = plt.subplots(1, 1, figsize=(8.2, 5.4))
    ax.plot(r, z["single_radius_crps"], marker="o", color=C_GREY, lw=1.8, label=qa._tex("one number: M*(<R) alone"))
    ax.plot(r, z["inner_crps"], marker="s", color=C_BLUE, lw=1.8, label="the profile INSIDE R (all points up to R)")
    ax.plot(r, z["outer_crps"], marker="^", color=C_RED, lw=1.8, label="the profile OUTSIDE R (all points from R to 148 kpc)")
    ax.axhline(z["inner_crps"][-1], color="k", lw=0.8, ls=":")
    ax.text(2.1, z["inner_crps"][-1] - 0.0022, "all 24 points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("R [kpc]")
    ax.set_ylabel("CRPS of the halo-mass prediction [dex]")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("exp87: the halo-mass information sits in the outer envelope (parent, z = 0.4)", fontsize=11)
    fig.tight_layout()
    save(fig, "exp87_radius")


def assembly():
    g = R.groups()
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.3))
    ax = axes[0]
    targets = [("logc", "concentration"), ("t50", "formation time $t_{50}$"), ("t75", "$t_{75}$"), ("t90", "$t_{90}$")]
    width = 0.2
    for i, (target, lab) in enumerate(targets):
        rows = g.get(("parent", 0, "parent", target), [])
        clim = R.find(rows, "mtot", "climatology")
        if clim is None:
            continue
        vals = []
        for sel in (R.find(rows, "none", "poly-mh", extra="mh"), R.find(rows, "mtot", "line"), R.best(rows, rungs=("L1", "L2", "L3")),
                    min([r for r in rows if r["extra"] != "none" and r["feature"] != "none"], key=lambda r: r["crps"])):
            vals.append(100 * (1 - sel["crps"] / clim["crps"]) if sel else np.nan)
        for j, (v, col) in enumerate(zip(vals, (C_GREY, C_ORANGE, C_BLUE, C_RED))):
            ax.bar(i + (j - 1.5) * width, v, width=width, color=col)
            ax.text(i + (j - 1.5) * width, max(v, 0) + 0.4, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([qa._tex(t[1]) for t in targets])
    ax.set_ylabel("CRPS skill over knowing nothing [" + qa._pct() + "]")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_GREY, label="true halo mass alone"), Patch(color=C_ORANGE, label="stellar mass alone"),
                       Patch(color=C_BLUE, label="the profile"), Patch(color=C_RED, label="the profile + true halo mass")], frameon=False,
              loc="upper left", ncol=2, fontsize=8)
    ax.set_ylim(-2, 31)
    ax.set_title("(a) assembly from the profile needs the halo mass", fontsize=11)

    for ax, epoch, tag in ((axes[1], 0, "(b)"), (axes[2], 4, "(c)")):
        z = np.load(C.OUTDIR / f"mah_epoch{epoch}.npz", allow_pickle=True)
        Y, used, taus = z["Y"], z["fold"] >= 0, z["lookbacks"]
        for name, col, ls, lab in (("climatology", C_GREY, "-", "knowing nothing"), ("mediated", C_ORANGE, "-", "the profile's own mass estimate"),
                                   ("ridge", C_BLUE, "-", "the profile"), ("oracle_mh", C_GREY, "--", "true halo mass alone"),
                                   ("cog_oracle", C_RED, "--", "the profile + true halo mass"), ("multi_epoch", C_GREEN, "-", "the stellar history (all earlier epochs)"),
                                   ("diffmah_truth", "k", ":", "floor of the DiffMAH form itself")):
            key = f"pred_{name}"
            if key not in z.files:
                continue
            pred = z[key]
            good = used & np.isfinite(pred).all(1)
            ax.plot(taus, np.sqrt(np.mean((pred[good] - Y[good]) ** 2, axis=0)), marker="o", color=col, ls=ls, lw=1.7, ms=4, label=lab)
        ax.set_xlabel("lookback before the observed epoch [Gyr]")
        ax.set_ylabel(qa._tex("rms error of log M(t - lookback) / M(t) [dex]"))
        ax.set_title(f"{tag} the halo's history before z = {C.ANCHOR_Z[epoch]}", fontsize=11)
        if epoch == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("exp87: concentration, formation time and the accretion history from the curve of growth", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "exp87_assembly")


def calibration():
    cell = HN.load_cell("parent__0__parent__mh__raw24__none__linear")
    used = cell["used"]
    y, mean, pit = cell["y"][used], cell["pred_mean"][used], cell["pit"][used]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.3))
    ax = axes[0]
    ax.hexbin(mean, y, gridsize=45, cmap="Greys", mincnt=1, bins="log")
    q = np.quantile(mean, np.linspace(0, 1, 11))
    xc = [np.median(mean[(mean >= a) & (mean <= b)]) for a, b in zip(q[:-1], q[1:])]
    for p, ls in ((16, ":"), (50, "-"), (84, ":")):
        ax.plot(xc, [np.percentile(y[(mean >= a) & (mean <= b)], p) for a, b in zip(q[:-1], q[1:])], color=C_RED, lw=1.8, ls=ls)
    ax.plot([13, 15], [13, 15], color=C_BLUE, lw=1.0)
    ax.set_xlabel(qa._tex(r"predicted log $M_{200c}$ (mean of the predictive)"))
    ax.set_ylabel(qa._tex(r"true log $M_{200c}$"))
    ax.set_xlim(13.0, 14.9)
    ax.set_ylim(12.95, 15.0)
    ax.set_title("(a) truth against prediction, 24-point linear model (median, 16-84" + qa._pct() + ")", fontsize=10.5)
    ax = axes[1]
    ax.hist(pit, bins=20, range=(0, 1), color=C_BLUE, alpha=0.75, density=True)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xlabel("PIT: the predictive's CDF at the truth")
    ax.set_ylabel("density (flat = calibrated)")
    ax.set_title(f"(b) calibration: PIT mean {pit.mean():.3f}, variance {pit.var():.4f} (uniform 0.500, 0.0833)", fontsize=10.5)
    ax = axes[2]
    lock = C.OUTDIR / "lockbox.json"
    if lock.exists():
        lb = json.loads(lock.read_text())
        xs = np.arange(len(lb["cells"]))
        ratio = np.array([r["crps_lockbox"] / r["crps_dev"] for r in lb["cells"]])
        lo = np.array([r["dev_lo"] / r["crps_dev"] for r in lb["cells"]])
        hi = np.array([r["dev_hi"] / r["crps_dev"] for r in lb["cells"]])
        ax.fill_between(xs, lo, hi, color=C_GREY, alpha=0.3, step="mid", label="development 99" + qa._pct() + " interval")
        inside = np.array([r["inside"] for r in lb["cells"]])
        ax.scatter(xs[inside], ratio[inside], color=C_BLUE, s=22, label="lockbox, inside")
        ax.scatter(xs[~inside], ratio[~inside], color=C_RED, s=36, marker="x", label="lockbox, outside")
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_xlabel("headline cell (halo mass z = 0.4; higher z; concentration; formation time)")
        ax.set_ylabel("lockbox CRPS / development CRPS")
        ax.legend(frameon=False, fontsize=8.5)
        ax.set_title(f"(c) the lockbox, scored once: {int((~inside).sum())} of {len(inside)} outside", fontsize=10.5)
    else:
        ax.text(0.5, 0.5, "the lockbox has not been scored", ha="center", va="center", transform=ax.transAxes)
    fig.suptitle("exp87: is the predictive distribution honest?", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "exp87_calibration")


def symbolic():
    path = C.OUTDIR / "stage4_cells.jsonl"
    if not path.exists():
        print("no symbolic-regression verdicts yet")
        return
    cells = [json.loads(line) for line in path.read_text().splitlines()]
    fig, ax = plt.subplots(1, 1, figsize=(13.5, 6.4))
    x, ticks = 0, []
    for cell in cells:
        for st in cell["stages"]:
            for v in st["verdicts"]:
                col = RUNG_COLOR["L4"] if v["accepted"] else (C_BLUE if v["significant"] else C_GREY)
                ax.errorbar(x, 100 * v["gain"], yerr=[[100 * (v["gain"] - v["gain_lo"])], [100 * (v["gain_hi"] - v["gain"])]], fmt="o", color=col, capsize=3)
                ticks.append(f"{cell['cell']}\n{v['stage']} {v['mode'][:3]}\n{v['inputs']}")
                x += 1
        ax.axvline(x - 0.5, color="k", lw=0.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(100 * C.GAIN_FLOOR, color=C_RED, lw=0.8, ls=":")
    ax.set_xticks(range(x))
    ax.set_xticklabels(ticks, fontsize=6.5, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_ylabel("CRPS gain over the linear model [" + qa._pct() + "], 95" + qa._pct() + " interval")
    ax.set_title("exp87: symbolic regression in latent space (purple = accepted: significant and recurring; blue = significant only)", fontsize=10.5)
    fig.tight_layout()
    save(fig, "exp87_symbolic")


def cross_epoch():
    out = json.loads((C.OUTDIR / "stage7_cross_epoch.json").read_text())
    zs = [z for z in out if z != "lag_scan"]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))
    ax = axes[0]
    series = [("M*(<148) at z=0.4", C_GREY, "-", "o", "stellar mass at z = 0.4"), ("ORACLE true Mh(z=0.4)", "k", ":", "x", "the TRUE halo mass at z = 0.4"),
              ("z=0.4 profile + quadratic", C_BLUE, "-", "o", "the z = 0.4 profile"), ("same-epoch profile + quadratic", C_RED, "-", "s", "the profile at that epoch"),
              ("both profiles + quadratic", C_GREEN, "-", "^", "both profiles"), ("ORACLE true Mh(0.4) + profile + quad", C_PURPLE, "--", "o", "z = 0.4 profile + true Mh(z = 0.4)")]
    for name, col, ls, mk, lab in series:
        ax.plot([float(z) for z in zs], [out[z]["rows"][name]["crps"] for z in zs], color=col, ls=ls, marker=mk, lw=1.8, label=lab)
    ax.set_xlabel("redshift of the halo mass being predicted")
    ax.set_ylabel("CRPS [dex], out of fold")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("(a) the main progenitor's halo mass at earlier epochs", fontsize=11)
    ax = axes[1]
    scan = out["lag_scan"]
    tau = [r["tau"] for r in scan]
    for key, col, ls, lab in (("profile", C_BLUE, "-", "the z = 0.4 profile"), ("mtot", C_GREY, "-", "stellar mass at z = 0.4"), ("mh0", "k", ":", "the TRUE halo mass at z = 0.4")):
        ax.plot(tau, [1 - r[key] / r["nothing"] for r in scan], color=col, ls=ls, marker="o", lw=1.8, label=lab)
    g = R.groups()
    now = R.find(g[("curated", 0, "complete", "mh")], "shells24", "ridge")
    clim = R.find(g[("curated", 0, "complete", "mh")], "mtot", "climatology")
    ax.scatter([0.0], [1 - now["crps"] / clim["crps"]], color=C_BLUE, marker="*", s=140, zorder=5, label="the z = 0.4 profile at z = 0.4 itself (truncated likelihood)")
    ax.set_xlabel("lookback before z = 0.4 [Gyr]")
    ax.set_ylabel("CRPS skill over knowing nothing")
    ax.set_ylim(0.2, 0.9)
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    ax.set_title("(b) the stars remember the halo as it was about 2.5 Gyr earlier", fontsize=11)
    fig.suptitle("exp87: what the z = 0.4 curve of growth knows about the halo's EARLIER mass", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "exp87_cross_epoch")


FIGS = dict(truncation=truncation, ladder=ladder, radius=radius, assembly=assembly, calibration=calibration, symbolic=symbolic,
            cross_epoch=cross_epoch)

if __name__ == "__main__":
    style()
    for name in (sys.argv[1:] or list(FIGS)):
        FIGS[name]()
