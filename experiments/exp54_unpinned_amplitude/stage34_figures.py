"""exp54 Stage 3.4 — figures for the representational ceiling (v2).

Six figures. Each exists to stop a specific misreading, and each is generated
from the SAVED FULL-SAMPLE OUTPUT rather than from whatever happened to be in
memory -- v1's ladder PNG was silently overwritten by a 60-galaxy smoke run,
which is why the smoke path now writes a `_smoke` suffix and why this module
reads `stage34_ceiling.npz` directly.

  `stage34_ladder_<label>`    the freedom ladder on the fixed z=2-threshold
      cohort, and where each bound's residual sits in radius. Carries the
      PERMUTED control -- one galaxy's deposits fitted directly to ANOTHER
      galaxy's measured profile -- because the ceiling's margin over the model
      is partly generic overcompleteness of the dictionary.

  `stage34_temporal_<label>`  the ladder in TEMPORAL RESOLUTION: K contiguous
      deposit groups, K = 1 to free. The misreading it prevents is the one v1
      made: with a single amplitude for everything before z=2, a rescaling
      cannot change a profile renormalised at 100 kpc, so K = 1 and K = 2 must
      score exactly what the fitted model scores at z=2. That identity is not
      evidence of no headroom, and the curve shows headroom appearing as soon
      as K = 3 puts a second component before z=2.

  `stage34_galaxies_<label>`  individual galaxies, because the experiment is
      about fitting individual objects and a population median hides which ones
      fail. Two whose centre declined, two whose did not.

  `stage34_rms_<label>`       the DISTRIBUTION of per-galaxy error, not its
      median.

  `stage34_pareto`            the basis-free monotone bound re-solved with the
      z=2 epoch weighted from 0.2 to 20. Monotonicity guarantees a conflict
      exists; it does not fix which epoch pays for it, and this is the curve
      along which the payment slides.

  `stage34_cz_<label>`        the redshift-dependent deposit shape scanned at
      two freedom levels, from `stage34_basis_scan.py`.

TERMS ON THE AXES, none of which should be read as obvious:

  **score_F**  the production shape loss divided by `L_F_ref = 0.158196`, the
    previous-generation model's value, so 1.0 means "as good as the incumbent"
    and lower is better. The loss is the root-mean-square fractional residual
    of `M*(<R)/M*(<100 kpc)` over the 24 radii, averaged over epochs and then
    over galaxies.

  **rung**  one bound, distinguished by how much freedom it gets PER GALAXY.

  **cohort**  the 840 galaxies whose halo is above the z=2 threshold, followed
    at all five epochs. The same objects at every epoch, which is what makes an
    evolution statement well posed. A fixed massive-progenitor cohort, not a
    general high-redshift population.

  **mh-complete**  above the halo mass at which our completeness first reaches
    60 per cent of its z=0.4 ceiling of 0.781 -- absolute completeness about
    0.47. A threshold, not completeness 1.

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python -u \\
     experiments/exp54_unpinned_amplitude/stage34_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink", HERE):
    sys.path.insert(0, str(p))

# imported before `halo`/`model` bind other experiments' same-named siblings
import fit as _fit_import_order_guard                    # noqa: E402,F401
import stage34_ceiling as C                              # noqa: E402
from hongshao.plotting import save_fig, set_style        # noqa: E402
from hongshao.qa import _pct, _tex                       # noqa: E402

R = C.R
Z = np.array(C.Z)
FIGDIR = HERE / "figures"

#: colour, linestyle, width, label -- ordered from most to least freedom
STYLE = {
    "monotone_obj": ("#999999", "-", 1.6,
                     "monotone in time (basis-free, absolute-frac loss)"),
    "epoch-alone":  ("#56B4E9", "-", 1.6, "each epoch fitted alone (not causal)"),
    "ceiling":      ("#0072B2", "-", 2.4, "ceiling: ~71 causal weights/galaxy"),
    "ceiling_fb":   ("#0072B2", ":", 1.4, r"the same with $\varepsilon \leq f_b$"),
    "bins8":        ("#CC79A7", "-", 1.8, "8 deposit-group weights/galaxy"),
    "bins3":        ("#F0E442", "-", 1.6, "3 deposit-group weights/galaxy"),
    "interval5":    ("#E69F00", "-", 1.8, "5 epoch-interval weights/galaxy"),
    "model":        ("#D55E00", "-", 2.4, "fitted model: 7 global parameters"),
    "permuted":     ("#009E73", "--", 1.6, "ceiling fitted to ANOTHER galaxy"),
}
LADDER_KEYS = ("monotone_obj", "epoch-alone", "ceiling", "bins3", "bins8",
               "interval5", "model")


def _caption(fig, text, width=190, fontsize=7.0):
    """Lay a wrapped caption under the axes and return the rect to leave for it.

    matplotlib's own `wrap=True` measures against the figure width, but the
    house style saves with `bbox_inches="tight"`, so an over-long line silently
    widens the canvas instead of wrapping. Pre-wrapping keeps the saved aspect
    ratio equal to the requested `figsize`.
    """
    import textwrap
    lines = textwrap.wrap(text, width=width)
    fig.text(0.5, 0.004, _tex("\n".join(lines)), ha="center", va="bottom",
             fontsize=fontsize, style="italic", color="0.35", linespacing=1.45)
    return 0.010 + len(lines) * (fontsize * 1.45 / 72.0) / fig.get_size_inches()[1]


class Saved:
    """Read `stage34_ceiling.npz`, whose keys are `label|sample|field`."""

    def __init__(self, path=None):
        self.z = np.load(path or C.OUT, allow_pickle=True)
        self.labels = [str(v) for v in self.z["labels"]]
        self.samples = [str(v) for v in self.z["samples"]]
        self.k_ladder = [int(v) for v in self.z["k_ladder"]]
        self.rungs = [str(v) for v in self.z["ladder"]]

    def get(self, label, tag, field):
        return self.z[f"{label}|{tag}|{field}"]

    def has(self, label, tag, field):
        return f"{label}|{tag}|{field}" in self.z

    def scores(self, label, tag, rung):
        return (self.get(label, tag, f"{rung}_sA"),
                self.get(label, tag, f"{rung}_sF"))


# --------------------------------------------------------------------------- #
def ladder_figure(label, res, data, mask, name=None):
    """score_F per epoch for every rung, and where each residual sits."""
    import matplotlib.pyplot as plt
    set_style()
    name = name or label
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.0))
    ax = ax.ravel()

    for key in LADDER_KEYS + ("ceiling_fb", "permuted"):
        if key not in res["scores"]:
            continue
        col, ls, lw, lab = STYLE[key]
        ax[0].plot(Z, res["scores"][key][1], ls, c=col, lw=lw, marker="o",
                   ms=3.5, label=_tex(lab))
    ax[0].axhline(1.0, c="0.75", lw=0.8, ls=":")
    ax[0].text(1.98, 1.01, _tex("the previous-generation model"), fontsize=6.5,
               color="0.5", va="bottom", ha="right")
    ax[0].set_xlabel(_tex("redshift z"))
    ax[0].set_ylabel(_tex("score_F   (shape loss / L_F_ref; lower is better)"))
    ax[0].set_title(_tex("A. the freedom ladder"), fontsize=10)
    ax[0].set_ylim(0.0, 1.72)
    ax[0].legend(fontsize=6.4, loc="upper center", ncol=2, frameon=False,
                 handlelength=2.4, columnspacing=1.1)

    worst = 0.0
    for k in range(5):
        a = ax[k + 1]
        for key in LADDER_KEYS:
            if key not in res["preds"]:
                continue
            col, ls, lw, _ = STYLE[key]
            g = mask[:, k]
            if g.sum() < 5:
                continue
            med = 100.0 * np.median(res["preds"][key][g, k] / data[g, k] - 1.0,
                                    axis=0)
            worst = max(worst, float(np.nanmax(np.abs(med))))
            a.plot(R, med, ls, c=col, lw=lw)
        a.axhline(0, c="0.6", lw=0.8)
        for y in (-10, 10):
            a.axhline(y, c="0.88", lw=0.7, ls=":")
        a.set_xscale("log")
        a.set_xlabel(_tex("R [kpc]"))
        a.set_title(_tex(f"z = {Z[k]}   (n = {int(mask[:, k].sum())})"),
                    fontsize=9.5)
        if k in (0, 2):
            a.set_ylabel(_tex(f"median (prediction - truth)/truth  [{_pct()}]"))
    lim = float(np.clip(1.15 * worst, 8.0, 60.0))
    for a in ax[1:]:
        a.set_ylim(-lim, lim)

    fig.suptitle(_tex(f"exp54 {label} — the representational ceiling on the "
                      f"fixed z=2-threshold cohort"), fontsize=12)
    bot = _caption(fig, (
        "Every curve is a BOUND on the deposition-only framework, differing "
        "only in how much freedom it gets per galaxy, and all are scored with "
        "the production loss on the same objects at every epoch. The basis "
        "draws any single epoch's profile to about one per cent "
        "(`epoch-alone`), so the profile family and the size law are not the "
        "limitation; requiring ONE causal weight vector to serve all five "
        "epochs is what costs. Read the ceiling's margin over the fitted model "
        "against the green dashed control, which is the same ~71-weight solve "
        "fitted to a DIFFERENT galaxy's measured profile: it has no "
        "halo-specific basis information but full access to the target's "
        "stellar data, so where it lies near the ceiling, the dictionary is "
        "simply overcomplete."))
    fig.tight_layout(rect=(0, bot, 1, 1))
    print("wrote", save_fig(fig, FIGDIR / f"stage34_ladder_{name}")[0],
          flush=True)


def temporal_figure(sv, label, name=None):
    """How the bound falls as a global law is given finer time resolution."""
    import matplotlib.pyplot as plt
    set_style()
    name = name or label
    ks = [k for k in sv.k_ladder if k != 0]
    free_x = max(ks) * 2.2
    fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.7))
    cols = {"all5": "#999999", "masked": "#56B4E9", "cohort": "#0072B2"}

    panels = ((lambda s, t: float(np.nanmean(sv.scores(label, t, s)[1])),
               "A. mean over epochs", "score_F  (lower is better)"),
              (lambda s, t: float(sv.scores(label, t, s)[1][4]),
               "B. at z = 2 only", "score_F at z = 2"),
              (lambda s, t: float(np.nanmean(sv.scores(label, t, s)[0])),
               "C. amplitude, mean over epochs", "score_A"))
    for j, (getter, ttl, ylab) in enumerate(panels):
        for tag in sv.samples:
            y = [getter(f"bins{k}", tag) for k in ks] + [getter("bins0", tag)]
            ax[j].plot(ks + [free_x], y, "-o", c=cols[tag], ms=4,
                       label=_tex(f"{tag} sample"))
            ax[j].axhline(getter("model", tag), c=cols[tag], ls=":", lw=1.1)
        ax[j].set_xscale("log")
        ax[j].set_xticks(ks + [free_x])
        ax[j].set_xticklabels([str(k) for k in ks] + ["free"], fontsize=7)
        ax[j].set_xlabel(_tex("K = free deposit-group amplitudes per galaxy"))
        ax[j].set_ylabel(_tex(ylab))
        ax[j].set_title(_tex(ttl), fontsize=10)
    ax[1].axvspan(0.9, 2.45, color="0.92", zorder=0)
    lo, hi = ax[1].get_ylim()
    ax[1].annotate(
        _tex("K <= 2 is an IDENTITY, not a result:\none amplitude covering "
             "everything before\nz=2 cannot change a profile\nnormalised at "
             "100 kpc"),
        xy=(2.0, hi - 0.06 * (hi - lo)), xytext=(7.0, hi - 0.10 * (hi - lo)),
        fontsize=6.6, color="0.3", va="top", ha="center",
        arrowprops=dict(arrowstyle="->", color="0.55", lw=0.9))
    ax[0].plot([], [], ":", c="0.4", label=_tex("dotted = the fitted model"))
    ax[0].legend(fontsize=7.0)

    fig.suptitle(_tex(f"exp54 {label} — how much of the gap to the ceiling is "
                      f"reachable at a given TIME RESOLUTION"), fontsize=12)
    bot = _caption(fig, (
        "Each point gives every galaxy K free non-negative amplitudes, one per "
        "contiguous group of deposits, with the distribution WITHIN a group "
        "held at the fitted law's. K = 1 is a single per-galaxy normalisation "
        "of the global law and K = free is the ceiling. The shaded region in "
        "panel B is an identity rather than a result: with one amplitude "
        "covering everything laid down before z = 2, rescaling it cannot "
        "change a profile renormalised at 100 kpc, so K <= 2 must reproduce "
        "the fitted model's z = 2 score exactly. An earlier version of this "
        "analysis read that identity as evidence of no headroom at z = 2; the "
        "headroom appears as soon as K = 3 puts a second component before "
        "z = 2. These rungs remain oracles fitted to the stellar data, so they "
        "bound what a global law could reach, not what one will."))
    fig.tight_layout(rect=(0, bot, 1, 1))
    print("wrote", save_fig(fig, FIGDIR / f"stage34_temporal_{name}")[0],
          flush=True)


def galaxy_figure(sv, label, pop_data, tag="cohort", name=None, n_each=2):
    """Individual galaxies, chosen by whether their centre declined."""
    import matplotlib.pyplot as plt
    set_style()
    name = name or label
    idx = sv.get(label, tag, "idx")
    data = pop_data[idx]
    dec = C.decline_flag(data)
    rms = sv.get(label, tag, "rms")
    pick = []
    for flag in (True, False):
        cand = np.where(dec == flag)[0]
        order = np.argsort(np.abs(rms[cand, 4] - np.median(rms[cand, 4])))
        pick += list(cand[order[:n_each]])
    keys = ("epoch-alone", "ceiling", "bins8", "model")

    ncol = len(pick)
    fig, ax = plt.subplots(2, ncol, figsize=(3.9 * ncol, 7.6), sharex=True,
                           height_ratios=[2, 1])
    cols = ["#332288", "#117733", "#999933", "#CC6677", "#882255"]
    for c, i in enumerate(pick):
        d = data[i]
        for k in range(5):
            ax[0, c].plot(R, np.log10(np.clip(d[k], 1.0, None)), "-",
                          c=cols[k], lw=1.9,
                          label=_tex(f"z = {Z[k]}") if c == 0 else None)
            ax[0, c].plot(R, np.log10(np.clip(
                sv.get(label, tag, "pred_ceiling")[i, k], 1.0, None)), "--",
                c=cols[k], lw=1.2)
        for key in keys:
            col, ls, lw, _ = STYLE[key]
            p = sv.get(label, tag, f"pred_{key}")[i]
            ax[1, c].plot(R, 100 * (p[4] / d[4] - 1), ls, c=col, lw=lw)
        ax[1, c].axhline(0, c="0.6", lw=0.8)
        for a in (ax[0, c], ax[1, c]):
            a.set_xscale("log")
        ax[1, c].set_xlabel(_tex("R [kpc]"))
        ttl = "centre DECLINED" if dec[i] else "centre did NOT decline"
        ax[0, c].set_title(_tex(f"{ttl}\nrow {int(idx[i])}, ceiling RMS at "
                                f"z=2 = {100 * rms[i, 4]:.1f} {_pct()}"),
                           fontsize=8.5)
    ax[0, 0].set_ylabel(_tex("log10 M*(<R)  [Msun]"))
    ax[1, 0].set_ylabel(_tex(f"z=2 residual  [{_pct()}]"))
    ax[0, 0].legend(fontsize=7, loc="upper left")
    for key in keys:
        col, ls, lw, lab = STYLE[key]
        ax[1, 0].plot([], [], ls, c=col, lw=lw, label=_tex(lab))
    ax[1, 0].legend(fontsize=6.0, loc="lower right")

    fig.suptitle(_tex(f"exp54 {label} — four individual galaxies: truth solid, "
                      f"causal ceiling dashed, one colour per epoch"),
                 fontsize=12)
    bot = _caption(fig, (
        "The top row shows each galaxy's measured curve of growth at all five "
        "epochs (solid) with the causal-ceiling fit over it (dashed); the "
        "bottom row shows the z = 2 residual of four rungs. The two left "
        "galaxies had their enclosed mass at 4.9 kpc FALL between z = 2 and "
        "z = 0.4, which no static non-negative deposition model can reproduce; "
        "the two right ones did not. The population's z = 2 central deficit is "
        "concentrated in the declining half, and a compact deposit channel "
        "cannot help there, because the problem is not that the central mass "
        "is missing but that it later has to leave."))
    fig.tight_layout(rect=(0, bot, 1, 1))
    print("wrote", save_fig(fig, FIGDIR / f"stage34_galaxies_{name}")[0],
          flush=True)


def rms_figure(sv, label, tag="cohort", name=None):
    """The DISTRIBUTION of per-galaxy error, not its median."""
    import matplotlib.pyplot as plt
    set_style()
    name = name or label
    fig, ax = plt.subplots(1, 5, figsize=(16.0, 3.9), sharey=True)
    series = (("epoch-alone", sv.get(label, tag, "rms_alone")),
              ("ceiling", sv.get(label, tag, "rms")),
              ("model", sv.get(label, tag, "rms_model")))
    bins = np.geomspace(0.1, 300.0, 46)
    for k in range(5):
        for nm, arr in series:
            col, ls, lw, lab = STYLE[nm]
            ax[k].hist(np.clip(100 * arr[:, k], bins[0], bins[-1]), bins=bins,
                       histtype="step", color=col, lw=1.7,
                       label=_tex(lab) if k == 0 else None)
            ax[k].axvline(100 * np.median(arr[:, k]), c=col, ls=":", lw=1.0)
        ax[k].set_xscale("log")
        ax[k].set_title(_tex(f"z = {Z[k]}"), fontsize=10)
        ax[k].set_xlabel(_tex(f"per-galaxy RMS of (pred-truth)/truth "
                              f"[{_pct()}]"))
    ax[0].set_ylabel(_tex("galaxies"))
    ax[0].legend(fontsize=6.6)
    fig.suptitle(_tex(f"exp54 {label} — the distribution of per-galaxy error, "
                      f"not its median (cohort; dotted = median)"), fontsize=12)
    bot = _caption(fig, (
        "The RMS fractional error of the ABSOLUTE curve of growth for one "
        "galaxy at one epoch, over the 24 radii. The ceiling and the fitted "
        "model are separated by roughly a factor of three in the median, but "
        "the distributions overlap and the model's tail reaches far beyond "
        "anything the bound produces, so a population score understates how "
        "badly the model fails on its worst objects."))
    fig.tight_layout(rect=(0, bot, 1, 1))
    print("wrote", save_fig(fig, FIGDIR / f"stage34_rms_{name}")[0], flush=True)


def pareto_figure(path=None, name="pareto"):
    """Monotonicity fixes that a conflict exists, not which epoch pays."""
    import matplotlib.pyplot as plt
    set_style()
    p = Path(path or (HERE / "outputs" / "stage34_objective.npz"))
    if not p.exists():
        print(f"  (no {p.name}; skipping the trade-off figure)")
        return
    z = np.load(p, allow_pickle=True)
    w = np.asarray(z["pareto_w"], float)
    par = np.asarray(z["pareto"], float)          # (n_w, 3): loss, z=2, z=0.4

    fig, ax = plt.subplots(1, 2, figsize=(11.8, 4.8))
    sc = ax[0].scatter(par[:, 1], par[:, 2], c=np.log10(w), cmap="cividis",
                       s=80, zorder=3, edgecolor="k", lw=0.5)
    ax[0].plot(par[:, 1], par[:, 2], "-", c="0.6", lw=1.1, zorder=2)
    for j, ww in enumerate(w):
        ax[0].annotate(f"{ww:g}", (par[j, 1], par[j, 2]), fontsize=6.5,
                       xytext=(5, 4), textcoords="offset points")
    fig.colorbar(sc, ax=ax[0], label=_tex("log10 of the z=2 epoch weight"))
    ax[0].axhline(0, c="0.85", lw=0.8)
    ax[0].axvline(0, c="0.85", lw=0.8)
    ax[0].set_xlabel(_tex(f"z = 2 residual at 2.0 kpc  [{_pct()}]"))
    ax[0].set_ylabel(_tex(f"z = 0.4 residual at 2.0 kpc  [{_pct()}]"))
    ax[0].set_title(_tex("A. where the unavoidable error is placed"),
                    fontsize=10)

    ax[1].plot(par[:, 1], par[:, 0], "-o", c="#0072B2", ms=5)
    for j, ww in enumerate(w):
        ax[1].annotate(f"{ww:g}", (par[j, 1], par[j, 0]), fontsize=6.5,
                       xytext=(5, 4), textcoords="offset points")
    ax[1].set_xlabel(_tex(f"z = 2 residual at 2.0 kpc  [{_pct()}]"))
    ax[1].set_ylabel(_tex("score_F of the monotone bound"))
    ax[1].set_title(_tex("B. what moving it costs"), fontsize=10)

    fig.suptitle(_tex("exp54 — monotonicity guarantees a conflict; it does not "
                      "fix which epoch pays for it"), fontsize=12)
    bot = _caption(fig, (
        "Each point is the best BASIS-FREE set of profiles that never "
        "decreases with time, optimised under the production amplitude-plus-"
        "shape loss with the z = 2 epoch up-weighted by the labelled factor. A "
        "deposition-only model can place the z = 2 central residual anywhere "
        "along this curve, so quoting any single value of it as irreducible is "
        "wrong: what is structural is the trade-off, not the number. Driving "
        "the z = 2 residual to zero costs a large central overshoot at z = 0.4 "
        "and a substantially worse loss."))
    fig.tight_layout(rect=(0, bot, 1, 1))
    print("wrote", save_fig(fig, FIGDIR / f"stage34_{name}")[0], flush=True)


def cz_figure(scan_npz, label="gompertz_log-E2-S2", name=None):
    """The redshift-dependent deposit shape, scanned at two freedom levels."""
    import matplotlib.pyplot as plt
    set_style()
    name = name or label
    z = np.load(scan_npz, allow_pickle=True)
    fitted = z["fitted"]
    fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.6))

    for i, (key, ttl) in enumerate((
            ("scan_shape", "B. at the ~71-weight ceiling"),
            ("scan_shape_i5", "C. at the 5-weight `interval5` rung"))):
        rec = z[key]
        c0s, czs = np.unique(rec[:, 2]), np.unique(rec[:, 3])
        grid = np.full((len(c0s), len(czs)), np.nan)
        for row in rec:
            grid[np.searchsorted(c0s, row[2]),
                 np.searchsorted(czs, row[3])] = row[4]
        # the surface spans a factor of four, so a linear scale hides the
        # structure that matters; saturate at the median and say so
        im = ax[i].pcolormesh(czs, c0s, grid, cmap="cividis_r",
                              vmin=float(np.nanmin(grid)),
                              vmax=float(np.nanmedian(grid)), shading="nearest")
        fig.colorbar(im, ax=ax[i], extend="max",
                     label=_tex("ceiling in score_F (saturated at the median)"))
        b = rec[np.argmin(rec[:, 4])]
        ax[i].plot(b[3], b[2], "*", c="#F0E442", ms=17, mec="k", mew=0.9,
                   label=_tex(f"grid optimum: c_z = {b[3]:+.2f}"))
        ax[i].plot(0.0, fitted[2], "o", mfc="none", mec="k", mew=1.4, ms=13,
                   label=_tex(f"fitted model: c = {fitted[2]:.3f}, c_z = 0"))
        ax[i].set_xlabel(r"$c_z$" + _tex("  in  c = c0 + c_z ln(1+z)"))
        ax[i].set_ylabel(r"$c_0$")
        ax[i].set_title(_tex(ttl), fontsize=10)
        ax[i].legend(fontsize=7, loc="upper left", framealpha=0.9,
                     frameon=True, facecolor="white", edgecolor="0.8")

    rec, rec5 = z["scan_shape"], z["scan_shape_i5"]
    c0_cut = np.unique(rec[:, 2])[np.argmin(np.abs(np.unique(rec[:, 2])
                                                   - fitted[2]))]
    sel = np.abs(rec[:, 2] - c0_cut) < 1e-9
    sel5 = np.abs(rec5[:, 2] - c0_cut) < 1e-9
    ax[2].plot(rec[sel, 3], rec[sel, 4], "-o", c="#0072B2", ms=3.5,
               label=_tex("ceiling (~71 weights)"))
    ax[2].plot(rec[sel, 3], rec[sel, 5], "--", c="#0072B2", lw=1.2,
               label=_tex("its permuted control"))
    ax[2].plot(rec5[sel5, 3], rec5[sel5, 4], "-o", c="#E69F00", ms=3.5,
               label=_tex("interval5 (5 weights)"))
    ax[2].plot(rec5[sel5, 3], rec5[sel5, 5], "--", c="#E69F00", lw=1.2,
               label=_tex("its permuted control"))
    ax[2].axvline(0.0, c="0.6", lw=0.9, ls=":")
    ax[2].set_xlabel(r"$c_z$")
    ax[2].set_ylabel(_tex("score_F  (lower is better)"))
    ax[2].set_title(_tex(f"D. cut at the fitted c0 = {c0_cut:.3f}"),
                    fontsize=10)
    ax[2].legend(fontsize=7)

    fig.suptitle(_tex(f"exp54 {label} — would a redshift-dependent deposit "
                      f"shape raise the ceiling? No, at either freedom level"),
                 fontsize=12)
    bot = _caption(fig, (
        "Each grid point gets exactly the same per-galaxy freedom; only the "
        "GLOBAL deposit-shape parameters move, so a drop in the surface would "
        "be a statement about the basis rather than about degrees of freedom. "
        "The parameter was proposed against the model's high-redshift "
        "diffuseness, on the reasoning that the basis cannot make compact "
        "enough deposits at early times. The optimum is at c_z = 0 in both "
        "panels and the surface rises in both directions. This is strong "
        "negative evidence against the proposed mechanism rather than a proof "
        "of impossibility: the scan minimises the ceiling's surrogate loss, "
        "and an over-complete dictionary can be insensitive to a basis change "
        "that a rigid global law would still feel."))
    fig.tight_layout(rect=(0, bot, 1, 1))
    print("wrote", save_fig(fig, FIGDIR / f"stage34_cz_{name}")[0], flush=True)


if __name__ == "__main__":
    import stage2_multiepoch as s2
    s2._w_init(None)
    pop = np.load(C.POP)
    sv = Saved()
    all_data = pop["data"][sv.z["rows"]]
    for label in sv.labels:
        idx = sv.get(label, "cohort", "idx")
        res = dict(
            scores={k: sv.scores(label, "cohort", k) for k in
                    sv.rungs + ["permuted", "truth-vs-truth"]
                    if sv.has(label, "cohort", f"{k}_sF")},
            preds={k: sv.get(label, "cohort", f"pred_{k}") for k in sv.rungs
                   if sv.has(label, "cohort", f"pred_{k}")})
        ladder_figure(label, res, all_data[idx],
                      sv.get(label, "cohort", "use"))
        temporal_figure(sv, label)
        galaxy_figure(sv, label, all_data)
        rms_figure(sv, label)
    pareto_figure()
    scan = HERE / "outputs" / "stage34_basis_scan.npz"
    if scan.exists():
        cz_figure(scan)
