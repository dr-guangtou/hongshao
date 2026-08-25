"""exp57 — the QA figure.

Six panels, and the figure exists to let a reader check the one claim that
matters: that this expansion term does NOT repeat the first model's failure.

  (a) THE EXPANSION FACTOR, which is NOT the same thing as the reach. `g`
      against the deposit's own formation time, as seen at redshift 0.4. For
      the homologous laws `g` multiplies every radius, so it IS the reach; for
      `X3` it is the CORE expansion only, and everything beyond `Rc` stays
      where it was. `X3` therefore has the largest `g` on this panel and moves
      the least mass of any model — panel (b) is where the reach actually
      lives, and the two panels must be read together or the reader will
      conclude the opposite of the truth. The old kernel's fitted exponent is
      drawn dashed so the original failure mode is visible rather than
      described, and the shaded band shows where the stellar mass was actually
      deposited, so a curve drawn where there is no mass is recognisable as
      decoration.

  (b) WHERE THE MOVED MASS GOES (gate G2). For each fitted model, the fraction
      of displaced mass delivered beyond 50 and beyond 148 kpc, against the
      preregistered limits and against what exp52 measured on the failing
      model. This is the panel the experiment turns on.

  (c) THE MECHANISM SPLIT (gate G1). The share of `M*[50,100]` growth that is
      material moved rather than newly deposited, epoch pair by epoch pair,
      against exp52's 18% -> 96.5% ramp. The vertical line marks z = 1.2, where
      minor-merger accretion should take over and where the first model crossed.

  (d) THE TARGET (gate G5). The model's central-mass error against epoch, split
      on whether the MEASURED centre declined. The vertical extent of each
      curve is the quantity under test; the null's 0.270 dex on the declining
      galaxies is what has to close.

  (e) THE CAPABILITY SCAN (Stage 1). What the expansion strength buys, and why
      one global value cannot buy all of it: the fraction of galaxies whose
      centres decline, and the median over all of them, against the two
      measured targets.

  (f) WHERE ADDED MASS LANDS (gate G1b). The fraction of a galaxy's mass growth
      inside 148 kpc that lands beyond 50 kpc — model against data, the same
      operator on both, so there is no threshold of mine in this panel at all.

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python -u \\
     experiments/exp57_expansion_term/stage5_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP54 = ROOT / "experiments/exp54_unpinned_amplitude"
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink", EXP54, HERE):
    sys.path.insert(0, str(p))

import expand as X                                       # noqa: E402
import gates as G                                        # noqa: E402
import stage0_cost as S0                                 # noqa: E402
from hongshao.plotting import save_fig, set_style        # noqa: E402
from hongshao.qa import _pct, _tex                       # noqa: E402

FIGDIR = HERE / "figures"
OUT = HERE / "outputs"
FITDIR = OUT / "stage3_fits"
Z = (0.4, 0.7, 1.0, 1.5, 2.0)
LAW_COLOR = {"null": "#666666", "X1": "#0072B2", "X2": "#009E73",
             "X0": "#D55E00", "X3": "#CC79A7"}
LAW_LABEL = {"X1": "X1  homologous, bounded factor",
             "X2": "X2  homologous, shaped in time",
             "X0": "X0  homologous power law (the OLD form)",
             "X3": "X3  CORE-ONLY, bounded reach in kpc"}


#: characters that are LaTeX control sequences and must not reach a label.
#: `#` arrives from the multi-start basin tags (`X3#smallcore`).
def _safe(t):
    return (str(t).replace("gompertz_log-E2-S2", "").replace("+", "")
            .replace("#", " ").replace("_", " ").strip())


def _fits():
    """Keyed by LABEL, not by law — `X3#smallcore` and `X3#bigcore` are two
    different fits of the same law and keying by law silently kept one."""
    out = {}
    for f in sorted(FITDIR.glob("*.npz")):
        if ".PARTIAL" in f.name:
            continue
        z = np.load(f, allow_pickle=True)
        lab = str(z["label"])
        if "frozen" in lab:                 # a control, not a candidate
            continue
        out[lab] = dict(z)
    # The 9-start X3 run and the `#bigcore` reproduction of its start 1 are the
    # SAME optimum, deliberately: one confirms the other. Plotting both would
    # draw one curve twice and imply two results where there is one. Keep the
    # shorter label so the panel says `X3`, not `X3 bigcore`.
    for a in list(out):
        for b in list(out):
            if a != b and a in out and b in out and len(b) > len(a):
                if np.allclose(out[a]["theta"], out[b]["theta"], rtol=1e-6):
                    del out[b]
    return out


def _color(lab, law):
    if law == "X3":
        return "#CC79A7" if "small" in lab else "#7570B3"
    return LAW_COLOR.get(law, "#333333")


def _label(lab, law):
    base = LAW_LABEL.get(law, law)
    if law == "X3":
        base = ("X3  core-only, SMALL core (Rc = 2.6 kpc)" if "small" in lab
                else "X3  core-only, large core (Rc = 92 kpc)")
    return base


def figure(name="exp57_expansion"):
    import matplotlib.pyplot as plt

    fits = _fits()
    if not fits:
        raise SystemExit("no fits on disk — run stage3_fit.py first")
    recs, data, mask, lmh, base, theta, slow = S0.build(False)
    sp1 = X.XSpec(base, "X1")
    pr = X.ExpandingProblem(sp1, recs, data, mask, epochs=(0, 1, 2, 3, 4))
    t_dep = pr.t_dep[np.isfinite(pr.t_dep)]
    t_view = pr.t_view[0]

    set_style()
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 8.4))
    ax = axes.ravel()

    # ---- (a) the reach law ------------------------------------------- #
    a = ax[0]
    tt = np.linspace(0.2, t_view, 300)
    # where the stellar mass actually is, as a function of deposit time
    import model as M
    eff, size, shape = base.unpack(theta)
    dm = 10.0 ** np.clip(M.log_eps(base, eff, pr.dep), -30, 10) * pr.dep.dmh
    w = (dm * pr.em[:, 0, :]).ravel()
    tdw = pr.t_dep.ravel()
    good = np.isfinite(tdw) & np.isfinite(w) & (w > 0)
    # the deposited-mass distribution goes on its OWN axis, behind everything,
    # so a reader never mistakes its height for an expansion factor. Coarse
    # bins because the snapshot grid makes a fine histogram spiky in a way that
    # is a property of the grid, not of the galaxies.
    hist, edges = np.histogram(tdw[good], bins=18, weights=w[good])
    hist = hist / hist.max()
    a_bg = a.twinx()
    a_bg.fill_between(0.5 * (edges[1:] + edges[:-1]), 0.0, hist,
                      color="0.90", zorder=0, lw=0)
    a_bg.set_ylim(0, 3.2)
    a_bg.set_yticks([])
    a_bg.set_zorder(0)
    a.set_zorder(2)
    a.patch.set_visible(False)
    a_bg.text(0.985, 0.30, _tex("grey: where the stellar mass\nwas deposited"),
              transform=a_bg.transAxes, fontsize=7, color="0.45",
              ha="right", va="bottom")
    for lab, z in fits.items():
        law = str(z["law"])
        th = np.asarray(z["theta"], float)
        xt = th[base.n_theta:]
        a.plot(tt, X.g_factor(law, xt, tt, t_view), lw=2.2,
               color=_color(lab, law), label=_tex(
                   f"{_label(lab, law)}: "
                   + ", ".join(f"{n}={v:+.2f}" for n, v
                               in zip(X.LAW_NAMES[law], xt))))
    a.plot(tt, X.g_factor("X0", np.array([0.91]), tt, t_view), ls="--", lw=1.5,
           color="#999999",
           label=_tex(r"the OLD kernel's fitted $q=0.91$"))
    a.axhline(1.0, color="0.6", lw=0.8)
    a.set_yscale("log")
    a.set_xlabel(_tex("cosmic time the deposit was laid down [Gyr]"))
    a.set_ylabel(_tex("expansion factor g, seen at z = 0.4"))
    a.set_title("(a) the expansion factor — NOT the reach\n"
                "for X3 it applies only inside Rc; see (b)", fontsize=9.5)
    a.text(0.03, 0.05,
           _tex("X3 has the LARGEST g and moves the LEAST mass:\n"
                "its g applies only within Rc, so the tail stays put"),
           transform=a.transAxes, fontsize=7, color="#CC79A7", va="bottom",
           bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.6))
    a.legend(fontsize=7, loc="upper right")

    # ---- (b) G2, where the moved mass goes --------------------------- #
    b = ax[1]
    rows = []
    for lab, z in fits.items():
        f = OUT / f"stage2_gates_{lab}.npz"
        if f.exists():
            g = np.load(f, allow_pickle=True)
            rows.append((lab, str(z["law"]), float(np.nanmax(g["g2_b50"])),
                         float(np.nanmax(g["g2_b148"]))))
    if rows:
        xx = np.arange(len(rows))
        b.bar(xx - 0.19, [r[2] for r in rows], 0.36,
              color=[_color(r[0], r[1]) for r in rows])
        b.bar(xx + 0.19, [r[3] for r in rows], 0.36,
              color="white", edgecolor=[_color(r[0], r[1]) for r in rows],
              hatch="////", lw=1.2)
        b.set_xticks(xx)
        b.set_xticklabels([_tex(_safe(r[0])) for r in rows], fontsize=7.5,
                          rotation=20, ha="right")
        from matplotlib.patches import Patch
        b.legend(handles=[Patch(fc="0.35", label=_tex("beyond 50 kpc")),
                          Patch(fc="white", ec="0.35", hatch="////",
                                label=_tex("beyond 148 kpc"))],
                 fontsize=7.5, loc="upper left")
    b.axhline(G.G2_MAX_BEYOND_50, color="#D55E00", ls=":", lw=1.4)
    b.axhline(G.G2_MAX_BEYOND_148, color="#D55E00", ls="--", lw=1.4)
    b.axhline(G.EXP52_REACH_50, color="0.4", ls="-", lw=1.0)
    b.text(0.985, G.EXP52_REACH_50 + 0.012,
           _tex(f"exp52's FAILING model: {100 * G.EXP52_REACH_50:.0f}"
                f"{_pct()} beyond 50 kpc"),
           fontsize=7, color="0.35", ha="right",
           transform=b.get_yaxis_transform())
    b.text(0.985, G.G2_MAX_BEYOND_50 + 0.012, _tex("limit, beyond 50 kpc"),
           fontsize=7, color="#D55E00", ha="right",
           transform=b.get_yaxis_transform())
    b.text(0.985, G.G2_MAX_BEYOND_148 + 0.012, _tex("limit, beyond 148 kpc"),
           fontsize=7, color="#D55E00", ha="right",
           transform=b.get_yaxis_transform())
    b.set_ylabel(_tex("fraction of DISPLACED mass delivered there"))
    b.set_title("(b) the reach, gate G2\nworst epoch pair per model",
                fontsize=9.5)

    # ---- (c) G1, the mechanism split --------------------------------- #
    c = ax[2]
    for lab, z in fits.items():
        f = OUT / f"stage2_gates_{lab}.npz"
        if not f.exists():
            continue
        g = np.load(f, allow_pickle=True)
        sel = np.array([v == "M[50,100]" for v in g["g1_shell"]])
        pairs = g["g1_pair"][sel][:4]
        share = g["g1_share"][sel][:4]
        zl = [float(str(p).split("-> ")[1]) for p in pairs]
        c.plot(zl, share, "o-", color=_color(lab, str(z["law"])), lw=2.0,
               label=_tex(_label(lab, str(z["law"]))))
    c.plot([1.5, 1.0, 0.7, 0.4], [0.180, 0.384, 0.769, 0.965], "s--",
           color="0.4", lw=1.4, ms=4,
           label=_tex("exp52's first model"))
    c.axvline(G.G1_Z_LIMIT, color="#D55E00", ls=":", lw=1.4)
    c.axhline(0.5, color="0.75", lw=0.8)
    c.text(G.G1_Z_LIMIT, 0.02, _tex(" z = 1.2: accretion should take over"),
           fontsize=7, color="#D55E00", va="bottom")
    c.invert_xaxis()
    c.set_xlabel(_tex("redshift of the later epoch"))
    c.set_ylabel(_tex("share of M*[50,100] growth that is MOVED mass"))
    c.set_title("(c) the mechanism split, gate G1", fontsize=9.5)
    c.legend(fontsize=7.5, loc="upper right")

    # ---- (d) G5, the target ------------------------------------------ #
    d = ax[3]
    for tag, col, ls in [("null", LAW_COLOR["null"], "--")] + [
            (lab, _color(lab, str(z["law"])), "-") for lab, z in fits.items()]:
        f = OUT / f"stage4_target_{tag}.npz"
        if not f.exists():
            continue
        t = np.load(f, allow_pickle=True)
        d.plot(Z, t["med_dec"], ls, color=col, lw=2.0, marker="o", ms=4,
               label=_tex(f"{_safe(tag)} declined, "
                          f"span {float(t['span_dec']):.3f}"))
        d.plot(Z, t["med_not"], ls, color=col, lw=1.2, alpha=0.5, marker="s",
               ms=3)
    d.axhline(0.0, color="0.75", lw=0.8)
    d.invert_xaxis()
    d.set_xlabel(_tex("redshift"))
    d.set_ylabel(_tex("median log10(model / truth), M*(<2 kpc)"))
    d.set_title("(d) the target, gate G5\nbold: centres that DECLINED; "
                "faint: those that did not", fontsize=9.5)
    d.legend(fontsize=7, loc="lower right")

    # ---- (e) the capability scan ------------------------------------- #
    e = ax[4]
    s1 = np.load(OUT / "stage1_contract.npz")
    e.plot(s1["A"], s1["frac"], "o-", color="#0072B2", lw=2.0,
           label=_tex("fraction of centres that DECLINE"))
    e.axhline(float(s1["truth_frac"]), color="#0072B2", ls=":", lw=1.4)
    e2 = e.twinx()
    e2.plot(s1["A"], s1["all_median"], "s-", color="#CC79A7", lw=2.0,
            label=_tex("median over ALL galaxies [dex]"))
    e2.axhline(float(s1["truth_all_median"]), color="#CC79A7", ls=":", lw=1.4)
    e2.set_ylabel(_tex("median dlog10 M*(<4.92 kpc) [dex]"), color="#CC79A7")
    for lab, z in fits.items():
        if str(z["law"]) == "X1":
            e.axvline(float(np.asarray(z["theta"])[base.n_theta]),
                      color=LAW_COLOR["X1"], lw=1.6)
            e.text(float(np.asarray(z["theta"])[base.n_theta]), 0.5,
                   _tex(" fitted A (X1)"), fontsize=7.5,
                   color=LAW_COLOR["X1"], rotation=90, va="center")
    e.set_xlabel(_tex("expansion strength A  (X1)"))
    e.set_ylabel(_tex("fraction declining"), color="#0072B2")
    e.set_title("(e) the capability scan\ndotted: the measured targets",
                fontsize=9.5)
    e.legend(fontsize=7, loc="upper left")

    # ---- (f) G1b, where added mass lands ----------------------------- #
    f_ = ax[5]
    for tag, col, ls in [("null", LAW_COLOR["null"], "--")] + [
            (lab, _color(lab, str(z["law"])), "-") for lab, z in fits.items()]:
        p = OUT / f"stage2_gates_{tag}.npz"
        if not p.exists():
            continue
        g = np.load(p, allow_pickle=True)
        zl = [1.5, 1.0, 0.7, 0.4]
        f_.plot(zl, g["g1b_model"][:4], ls, color=col, lw=2.0, marker="o",
                ms=4, label=_tex(_safe(tag) or "null"))
        if tag == "null":
            f_.plot(zl, g["g1b_data"][:4], "-", color="black", lw=2.4,
                    marker="D", ms=4.5, label=_tex("MEASURED"))
    f_.invert_xaxis()
    f_.set_xlabel(_tex("redshift of the later epoch"))
    f_.set_ylabel(_tex("fraction of added mass landing beyond 50 kpc"))
    f_.set_title("(f) where added mass lands, gate G1b\nno threshold of mine "
                 "in this panel", fontsize=9.5)
    f_.legend(fontsize=7.5, loc="upper left")

    fig.suptitle(_tex(
        "exp57 — an empirical expansion term on exp54's deposition model, "
        "2397 galaxies. Deposits grow after they are laid down, mass conserved "
        "exactly, so the enclosed mass can now FALL. X0/X1/X2 scale every "
        "radius\n(homologous); X3 expands only the core and leaves everything "
        "beyond Rc where it was. The design question is not whether it fits "
        "but whether it repeats the first model's failure: right source, ten "
        "times too much reach."),
        fontsize=8.6, y=1.02)
    fig.tight_layout()
    paths = save_fig(fig, FIGDIR / name)
    print("wrote " + ", ".join(str(p) for p in paths))
    plt.close(fig)


if __name__ == "__main__":
    figure()
