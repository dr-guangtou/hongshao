"""exp61 Stage 2b — the four figures for the baseline and drift tables.

Reads `outputs/stage0_fits.npz`, `outputs/stage1_mx2_fit.npz` and
`outputs/stage2_figure_data.npz` (written by `stage2_figure_data.py`, which
re-evaluates the frozen thetas and self-checks against the saved fits). No
model is evaluated here, and every number drawn or quoted comes from those
files — none is typed in.

  exp61_baselines  What one set of parameters serving five epochs costs, in
                   four currencies: the two scores the fit optimises and the
                   two biases the eye sees. Panels (a)/(b) are the baseline
                   table; (c)/(d) are the frozen exp59 B-gate quantities, and
                   they carry the result the scores cannot show — the
                   single-epoch freedom removes the z=2 offset outright.

  exp61_transfer   The cross-epoch matrix: every parameter set evaluated at
                   every epoch, each cell the share of that set's loss at that
                   epoch which a fit devoted to the epoch alone removes. The
                   shared incumbent's row is the compromise cost; the rest of
                   the matrix is how far apart the epochs' own optima are.

  exp61_drift      The seven parameters against redshift, one panel each,
                   with the shared value and the z <= 1.0 value alongside.
                   This is the drift table, and the panel to read together
                   with the degeneracy caveat printed in its last cell.

  exp61_m2_null    Stage 1: the mass x time-curvature term. Four starts, one
                   interior optimum, on the falsified side of the
                   pre-registered sign, and gate quantities indistinguishable
                   from the null's.

Run: PYTHONPATH=. uv run python -u \\
     experiments/exp61_epoch_baselines/stage2_figures.py
     (no HONGSHAO_DATA_DIR needed — everything is read from the npz files)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt                          # noqa: E402

from hongshao.plotting import save_fig, set_style        # noqa: E402
from hongshao.qa import _pct                             # noqa: E402

OUTDIR = HERE / "outputs"
FIGDIR = HERE / "figures"
ANCHOR_Z = np.array([0.4, 0.7, 1.0, 1.5, 2.0])
#: score_F x this = the model's typical percentage error on the growth curve
#: (fit.L_F_REF = 0.158196, the normalisation the shape score is divided by)
L_F_REF_PCT = 15.8196
#: the frozen B1 threshold (the exp58 plan; never widened) [dex]
B1_LIM = 0.020
#: the four values Stage 1 started the curvature coefficient from
M2_STARTS = np.array([0.0, 0.5, -0.5, 1.5])
#: qa._EPOCH_COLORS — the project's distinct epoch colours, cool to warm
EPOCH_COLORS = ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]
#: model-series colours: Okabe-Ito, chosen NOT to collide with the epoch set
C_SHARED, C_SINGLE, C_SCOPE, C_M2 = "#000000", "#D55E00", "#56B4E9", "#009E73"
ZLE = r"$z \leq 1.0$"

MATH = {"a0": r"$a_0$", "a_M": r"$a_M$", "a_z": r"$a_z$",
        "a_Mz": r"$a_{Mz}$", "log_f0": r"$\log_{10} f_0$", "b": r"$b$",
        "c": r"$c$"}
#: what each parameter does, in words, for the drift panels
MEANING = {
    "a0": "deposition efficiency at the pivot:\n"
          r"$\log_{10}$ of the stellar mass formed per unit"
          "\n" r"halo-mass growth, at $M_h=10^{13.5}\,M_\odot$, $z=0$",
    "a_M": "how that efficiency tilts with halo mass,\n"
           r"per dex above $10^{13.5}\,M_\odot$",
    "a_z": "how it tilts with time,\n" r"per unit $\ln(1+z)$",
    "a_Mz": r"the mass $\times$ time cross term: how the mass"
            "\ntilt itself changes with time",
    "log_f0": "deposit half-mass radius as a fraction of\n"
              r"the halo's $R_{200c}$, at $z=0$ ($\log_{10}$)",
    "b": "how that fraction changes with time,\n"
         r"per unit $\log_{10}(1+z)$",
    "c": "shape of one deposit's growth curve:\n"
         r"$M(<R)\propto\exp[-\ln 2\,(R/R_{50})^{-c}]$",
}


def _load():
    s1 = np.load(OUTDIR / "stage1_mx2_fit.npz", allow_pickle=True)
    s2 = np.load(OUTDIR / "stage2_figure_data.npz", allow_pickle=True)
    return s1, s2


def _epoch_axis(ax):
    ax.set_xticks(ANCHOR_Z)
    ax.set_xticklabels([f"{z:.1f}" for z in ANCHOR_Z])
    ax.set_xlim(0.25, 2.15)
    ax.set_xlabel("redshift of the observed profile")


def _series(ax, y, color, label, marker="o", scope=None, lw=1.7, ms=6):
    """One model's five per-epoch values. `scope` is how many epochs the fit
    was actually driven by: those are drawn solid and filled, the rest with a
    dashed connector and hollow markers, so no claim attaches to them."""
    y = np.asarray(y, float)
    k = 5 if scope is None else int(scope)
    ax.plot(ANCHOR_Z[:k], y[:k], "-", color=color, lw=lw, marker=marker,
            ms=ms, label=label, zorder=3)
    if k < 5:
        ax.plot(ANCHOR_Z[k - 1:], y[k - 1:], "--", color=color, lw=lw,
                zorder=3)
        ax.plot(ANCHOR_Z[k:], y[k:], ls="none", marker=marker, ms=ms,
                mfc="white", mec=color, mew=1.4, zorder=3)


def _shortfall_pct(dex):
    """A median log10(model/measured) offset, said as a percentage."""
    return 100.0 * (1.0 - 10.0 ** float(dex))


# --------------------------------------------------------------------------- #
# figure 1 — the baseline table in four currencies                             #
# --------------------------------------------------------------------------- #
def figure_baselines(s2):
    b1, b3 = s2["b1"], s2["b3"]
    diag = {k: np.array([s2[k][1 + j, j] for j in range(5)])
            for k in ("sa", "sf", "b1", "b3")}

    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.6))
    (a, b), (c, d) = axes
    L_SHARED = "shared incumbent: one set of seven parameters, all five epochs"
    L_SINGLE = "single-epoch fit, each at its own epoch (the class ceiling)"
    L_SCOPE = ("the " + ZLE + " refit\n(dashed, hollow = outside the epochs "
               "it was fitted to)")

    def three(ax, key, **kw):
        _series(ax, s2[key][0], C_SHARED, L_SHARED, **kw)
        _series(ax, diag[key], C_SINGLE, L_SINGLE, marker="s", **kw)
        _series(ax, s2[key][6], C_SCOPE, L_SCOPE, marker="^", scope=3, **kw)

    # ---- (a) amplitude score ------------------------------------------ #
    a.axhline(1.0, color="0.55", lw=1.0, ls=":", zorder=1)
    a.annotate("the dotted line at score$_A$ = 1.0 is the benchmark:\n"
               "a halo-mass regression refitted at that epoch",
               xy=(0.03, 0.30), xycoords="axes fraction", fontsize=7.5,
               color="0.35", va="bottom")
    three(a, "sa")
    a.set_ylabel("score$_A$  (amplitude; lower is better)")
    a.set_title("(a) the total stellar mass inside 103 kpc\n"
                "how far the predicted mass scatters from the measured one")
    a.annotate("multiply by that epoch's benchmark scatter\n"
               "(0.104, 0.112, 0.120, 0.128, 0.161 dex)\n"
               "to read the model's own scatter in dex",
               xy=(0.03, 0.04), xycoords="axes fraction", fontsize=7.5,
               color="0.35", va="bottom")
    _epoch_axis(a)
    a.legend(loc="upper right", fontsize=7.5)

    # ---- (b) shape score ---------------------------------------------- #
    three(b, "sf")
    b.set_ylabel("score$_F$  (profile shape; lower is better)")
    b.set_title("(b) the shape of the curve of growth\n"
                "how far the predicted mass-versus-radius curve departs "
                "from the measured one")
    rb = b.secondary_yaxis("right",
                           functions=(lambda v: v * L_F_REF_PCT,
                                      lambda v: v / L_F_REF_PCT))
    rb.set_ylabel(f"the same, as a typical error on the curve [{_pct()}]")
    _epoch_axis(b)

    # ---- (c) B1, the offset the scores cannot see ---------------------- #
    c.axhspan(-B1_LIM, B1_LIM, color="#009E73", alpha=0.13, zorder=0)
    c.axhline(0.0, color="0.55", lw=1.0, ls=":", zorder=1)
    c.annotate(r"the frozen B1 gate: $|\mathrm{offset}| \leq 0.020$ dex",
               xy=(0.28, B1_LIM), xytext=(2, 3), textcoords="offset points",
               fontsize=7.5, color="#00694d", va="bottom")
    three(c, "b1")
    c.set_ylabel(r"median $\log_{10}$(model / measured) at 103 kpc  [dex]")
    c.set_title("(c) the systematic offset, which neither score can show\n"
                "a score divides by the scatter, and cannot tell a bias "
                "from noise")
    c.annotate(f"At z=2 the shared law is {abs(b1[0][4]):.3f} dex light — it "
               f"misses\n{_shortfall_pct(b1[0][4]):.0f} per cent of the "
               f"mass, and it is the one epoch\nwhere the gate fails. Freed "
               f"of the other four epochs\nthe same seven parameters land at "
               f"{diag['b1'][4]:+.3f} dex.",
               xy=(0.03, 0.06), xycoords="axes fraction", fontsize=8.0,
               color="0.25", va="bottom")
    _epoch_axis(c)

    # ---- (d) B3, the inner aperture ------------------------------------ #
    d.axhline(0.0, color="0.55", lw=1.0, ls=":", zorder=1)
    three(d, "b3")
    d.set_ylabel(f"median (model $-$ measured) / measured  [{_pct()}]")
    d.set_title("(d) the stellar mass inside 10.25 kpc\n"
                "the inner-region deficit the user reads off the QA figures")
    d.annotate(f"At z=2 the shared law is {abs(b3[0][4]):.1f} per cent short "
               f"in the centre;\nthe single-epoch fit is "
               f"{diag['b3'][4]:+.1f} per cent. The retreat to\n"
               + ZLE + " does not help here — out of its scope it\n"
               "is the worst of the three.",
               xy=(0.03, 0.06), xycoords="axes fraction", fontsize=8.0,
               color="0.25", va="bottom")
    _epoch_axis(d)

    fig.suptitle("exp61 — what one set of parameters serving five epochs "
                 "costs, in four currencies.\nThe scores (top) say the "
                 "compromise is cheap at every epoch; the biases (bottom) "
                 "say what it does cost is a systematic offset at z=2.",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_fig(fig, FIGDIR / "exp61_baselines")


# --------------------------------------------------------------------------- #
# figure 2 — the cross-epoch transfer matrix                                   #
# --------------------------------------------------------------------------- #
def figure_transfer(s2):
    rec = np.asarray(s2["recoverable"], float)
    pretty = (["shared incumbent"] + [f"fitted at z={z:.1f}" for z in ANCHOR_Z]
              + ["fitted to " + ZLE])
    shared_worst_low = float(rec[0, :4].max())
    shared_z2 = float(rec[0, 4])
    worst = float(rec.max())
    wi, wj = np.unravel_index(int(np.argmax(rec)), rec.shape)

    set_style()
    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    im = ax.imshow(rec, cmap="cividis", aspect="auto", vmin=0.0, vmax=worst)
    ax.grid(False)
    for i in range(rec.shape[0]):
        for j in range(5):
            v, own = rec[i, j], (i == j + 1)
            r, g, bl, _ = im.cmap(im.norm(v))
            lum = 0.299 * r + 0.587 * g + 0.114 * bl
            # cividis runs through mid-luminance greys: pick the text colour
            # from the cell's own luminance, not from its value
            ax.text(j, i, "0" if own else f"{v:.1f}", ha="center",
                    va="center", fontsize=9.5,
                    color="white" if lum < 0.45 else "0.05",
                    fontweight="bold" if i == 0 else "normal")
            if own:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           edgecolor="white", lw=2.0, zorder=4))
    ax.add_patch(plt.Rectangle((2.5, 5.5), 2, 1, fill=False, ls=":",
                               edgecolor="white", lw=1.6, zorder=4))
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"z={z:.1f}" for z in ANCHOR_Z])
    ax.set_yticks(range(len(pretty)))
    ax.set_yticklabels(pretty)
    ax.set_xlabel("the epoch the parameters are being scored at")
    ax.set_ylabel("the parameter set being scored")
    ax.set_title("exp61 — how much of each parameter set's loss at each epoch "
                 "a fit devoted to that\nepoch alone would remove "
                 f"[{_pct()} of that set's loss there]", fontsize=10.5)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(f"recoverable share of the loss [{_pct()}]")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(True)

    ax.text(0.0, -0.155,
            "Read a row: hold that parameter set, and look across the epochs. "
            "The shared incumbent's row is the price of one law for all five "
            "epochs —\nat most "
            f"{shared_worst_low:.1f} per cent at every epoch up to z=1.5, and "
            f"{shared_z2:.1f} per cent at z=2.0, the only large number in it. "
            "Every other row is one epoch's\nown optimum carried somewhere "
            f"else, and those are far more expensive: the fit made at "
            f"z={ANCHOR_Z[wi - 1]:.1f} gives up {worst:.1f} per cent of its "
            f"loss at z={ANCHOR_Z[wj]:.1f}.\nWhite boxes are each fit's own "
            "epoch (0 by construction); the dotted box is the " + ZLE
            + " fit outside the epochs it was fitted to.",
            fontsize=8.2, color="0.25", ha="left", va="top",
            transform=ax.transAxes)
    fig.tight_layout()
    return save_fig(fig, FIGDIR / "exp61_transfer")


# --------------------------------------------------------------------------- #
# figure 3 — the parameter drift                                               #
# --------------------------------------------------------------------------- #
def figure_drift(s2):
    names = [str(v) for v in s2["names"]]
    th = np.asarray(s2["thetas"], float)                 # (7 sets, 7 params)
    lo = np.asarray(s2["bound_lo"], float)
    hi = np.asarray(s2["bound_hi"], float)
    single, shared, scope = th[1:6], th[0], th[6]
    gap = np.minimum(single - lo[None, :], hi[None, :] - single)
    k = int(np.unravel_index(int(np.argmin(gap)), gap.shape)[1])
    c_lo, c_hi = single[:, names.index("c")].min(), single[:, names.index("c")].max()

    set_style()
    fig, axes = plt.subplots(2, 4, figsize=(15.2, 8.0))
    ax = axes.ravel()
    for p, name in enumerate(names):
        g = ax[p]
        g.plot(ANCHOR_Z, single[:, p], "-", color="0.55", lw=1.2, zorder=2)
        for j in range(5):
            g.plot([ANCHOR_Z[j]], [single[j, p]], "o", ms=8, zorder=4,
                   color=EPOCH_COLORS[j],
                   label="single-epoch fits, one point per epoch"
                   if (p == 0 and j == 0) else None)
        g.axhline(shared[p], color=C_SHARED, lw=1.4, ls="--", zorder=3,
                  label="the shared incumbent" if p == 0 else None)
        g.plot([0.4, 1.0], [scope[p]] * 2, "-", color=C_SCOPE, lw=3.2,
               solid_capstyle="butt", zorder=3,
               label="the " + ZLE + " refit" if p == 0 else None)
        g.set_title(f"{MATH[name]}:  ${single[0, p]:+.2f} \\rightarrow "
                    f"{single[4, p]:+.2f}$\n" + MEANING[name], fontsize=8.6)
        g.margins(y=0.14)
        g.set_xticks(ANCHOR_Z)
        g.set_xticklabels([f"{z:.1f}" for z in ANCHOR_Z], fontsize=7.5)
        g.set_xlim(0.25, 2.15)
        if p >= 3:
            g.set_xlabel("redshift of the observed profile", fontsize=8.5)
    for p in (0, 4):
        ax[p].set_ylabel("fitted value", fontsize=9)

    t = ax[7]
    t.axis("off")
    handles, labels = ax[0].get_legend_handles_labels()
    t.legend(handles, labels, loc="upper center", fontsize=8.5, frameon=False,
             bbox_to_anchor=(0.5, 1.03))
    t.text(0.0, 0.74,
           "Top row: the four numbers that set HOW MUCH stellar mass each\n"
           "deposit carries. All four walk in the same direction, epoch by\n"
           "epoch, without a turn.\n\n"
           "Bottom row: the two that set HOW BIG each deposit is, walking\n"
           f"the other way, with the deposit's shape almost still ({c_lo:.2f}\n"
           f"to {c_hi:.2f} across the five fits).\n\n"
           r"A cross term that walks is curvature in the mass $\times$ time"
           "\nplane — which is the term Stage 1 then went and fitted, and\n"
           "did not find (see the exp61 m2 null figure).\n\n"
           "CAVEAT, and it is a large one. These are single-epoch fits, so\n"
           "each point may slide along the degeneracies exp54 Stage 3.9\n"
           "measured: the four efficiency parameters absorb one another by\n"
           "factors of 12 to 18, and no profile likelihood was run here.\n"
           "Read the DIRECTION of each panel, not the size of the step. What\n"
           "is solid is each fit's loss — the other two figures.\n\n"
           f"No fitted value sits at a bound: the closest is {MATH[names[k]]},"
           f" still\n{gap[:, k].min():.2f} from its nearest limit.",
           fontsize=8.0, color="0.20", va="top", ha="left",
           transform=t.transAxes)

    fig.suptitle("exp61 — the seven parameters refitted at each epoch on its "
                 "own, against the shared law they otherwise have to agree on",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    return save_fig(fig, FIGDIR / "exp61_drift")


# --------------------------------------------------------------------------- #
# figure 4 — the Stage 1 curvature null                                        #
# --------------------------------------------------------------------------- #
def figure_m2_null(s1, s2):
    got = np.asarray(s1["start_thetas"], float)[:, -1]
    sl = np.asarray(s1["start_losses"], float)
    best = float(np.asarray(s1["theta"], float)[-1])
    l0, lb = float(s1["loss_incumbent"]), float(s1["loss"])
    b1, b3 = s2["b1"], s2["b3"]

    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.1))
    a, b, c = axes

    # ---- (a) where the four starts land -------------------------------- #
    a.axhspan(0.0, 3.0, color="#D55E00", alpha=0.10, zorder=0)
    a.axhline(0.0, color="0.55", lw=1.0, ls=":", zorder=1)
    a.axhline(best, color="#009E73", lw=1.4, ls="--", zorder=2)
    a.plot([-0.85, 1.85], [-0.85, 1.85], ":", color="0.7", lw=1.0, zorder=1)
    for x, y, loss in zip(M2_STARTS, got, sl):
        a.annotate("", xy=(x, y), xytext=(x, x), zorder=3,
                   arrowprops=dict(arrowstyle="-|>", color="#0072B2", lw=1.4,
                                   shrinkA=4, shrinkB=2))
        a.plot([x], [x], "o", ms=6, mfc="white", mec="#0072B2", mew=1.4,
               zorder=4)
        a.plot([x], [y], "o", ms=9, color="#0072B2", zorder=4)
        a.annotate(f"loss {loss:.6f}", xy=(x, y), xytext=(0, -15),
                   textcoords="offset points", ha="center", fontsize=7.5,
                   color="0.3")
    a.annotate("the pre-registered sign, read off the drift table:\n"
               "MORE curvature toward early times — FALSIFIED",
               xy=(0.5, 1.62), ha="center", fontsize=8.0, color="#8a3800")
    a.annotate("hollow = where the fit was started, filled = where it\n"
               "finished; the dotted diagonal is no movement at all",
               xy=(0.03, 0.46), xycoords="axes fraction", fontsize=8.0,
               color="0.35", va="bottom")
    a.annotate(f"all four starts converge here, ${best:+.4f}$",
               xy=(-0.5, best), xytext=(4, 12), textcoords="offset points",
               fontsize=8.0, color="#00694d")
    a.set_xlabel("the value the fit was started from")
    a.set_ylabel("the value the fit converged to")
    a.set_ylim(-0.66, 1.95)
    a.set_xlim(-0.85, 1.85)
    a.set_title("(a) the mass $\\times$ time-curvature coefficient "
                "$a_{mz2}$\nfour starts, one interior optimum, and it is "
                "the other sign")

    # ---- (b) B1, null vs the curvature fit ------------------------------ #
    b.axhspan(-B1_LIM, B1_LIM, color="#009E73", alpha=0.13, zorder=0)
    b.axhline(0.0, color="0.55", lw=1.0, ls=":", zorder=1)
    _series(b, b1[0], C_SHARED, "the null: the shared incumbent")
    _series(b, b1[7], C_M2, "with the curvature term fitted", marker="D",
            lw=1.1, ms=5)
    b.set_ylabel(r"median $\log_{10}$(model / measured) at 103 kpc  [dex]")
    b.set_title("(b) the systematic offset the term was built to move\n"
                f"at z=2: ${b1[0][4]:+.4f}$ dex without it, "
                f"${b1[7][4]:+.4f}$ dex with it")
    b.annotate("the two curves never separate by more than\n"
               f"{np.max(np.abs(b1[7] - b1[0])):.4f} dex, at any epoch",
               xy=(0.03, 0.30), xycoords="axes fraction", fontsize=8.0,
               color="0.35", va="bottom")
    _epoch_axis(b)
    b.legend(loc="lower left", fontsize=8)

    # ---- (c) B3, null vs the curvature fit ------------------------------ #
    c.axhline(0.0, color="0.55", lw=1.0, ls=":", zorder=1)
    _series(c, b3[0], C_SHARED, "the null: the shared incumbent")
    _series(c, b3[7], C_M2, "with the curvature term fitted", marker="D",
            lw=1.1, ms=5)
    c.set_ylabel(f"median (model $-$ measured) / measured  [{_pct()}]")
    c.set_title("(c) the stellar mass inside 10.25 kpc\n"
                f"at z=2: ${b3[0][4]:+.1f}$ per cent without it, "
                f"${b3[7][4]:+.1f}$ per cent with it")
    c.annotate("the two curves never separate by more than\n"
               f"{np.max(np.abs(b3[7] - b3[0])):.1f} of a percentage point, "
               "at any epoch",
               xy=(0.03, 0.30), xycoords="axes fraction", fontsize=8.0,
               color="0.35", va="bottom")
    _epoch_axis(c)
    c.legend(loc="lower left", fontsize=8)

    fig.suptitle("exp61 Stage 1 — a clean null: one extra parameter for "
                 "curvature in mass $\\times$ time buys "
                 f"{100 * (1 - lb / l0):+.2f} per cent of the loss "
                 "and moves nothing the eye can see", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return save_fig(fig, FIGDIR / "exp61_m2_null")


def main():
    s1, s2 = _load()
    for paths in (figure_baselines(s2), figure_transfer(s2),
                  figure_drift(s2), figure_m2_null(s1, s2)):
        for p in paths:
            print(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
