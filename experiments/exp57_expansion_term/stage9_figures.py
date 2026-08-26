"""exp57 — the second QA figure: the mechanism, not the verdict.

`stage5_figures.py` shows whether each model passes its gates. This one shows
WHY, and it exists because three of the experiment's strongest results were
tables only.

  (a) WHERE THE MOVED MASS ACTUALLY GOES, shell by shell. The gate reports one
      number — the fraction delivered beyond 50 kpc — and one number cannot
      show that two models take mass from the SAME place and put it in wildly
      different places. Negative bars are mass drained out of a shell, positive
      bars mass delivered into it, both as a fraction of the galaxy's stellar
      mass, over the full redshift 2 to 0.4 baseline. The homologous law empties
      the centre into the stellar halo; the core-only law moves the same
      material a few kpc and stops.

  (b) THE TRADE, mapped (Stage 8). The reach gate and the two halves of the
      target gate against the core radius, at both fitted anchors. This is the
      panel that corrected my own framing: the reach gate holds for every `Rc`
      below about 40 kpc, so it is NOT the binding constraint once the
      mechanism is non-homologous — the conflict is between the two halves of
      G5. **It is a CONDITIONAL scan**, the other eight parameters held fixed,
      which exp54 Stage 3.9 showed over-states a parameter's determination by
      2.3x to 17.6x. It is used for the SHAPE of a trade, never to locate an
      optimum, and drawn at both anchors so dependence on the anchor is visible.

  (c) WHY exp54 IS NOT THE OLD MODEL. The measurement made before any code was
      written. The transport kernel deposited 1.5% of its stellar mass below
      redshift 1 while the halo assembled 47% of its mass there, so draining the
      centre was the only way it could grow an outskirt. exp54 deposits 29.3%
      there, and deposits it at 35-50 kpc. That is why the first model's
      DOMINANCE failure does not recur here even where its REACH failure does.

  (d) WHAT THE EXPANSION DID, AND WHAT REFITTING DID. Switching the expansion
      off at each fit's OWN base parameters separates the mechanism's
      contribution from the compensation the other seven parameters supplied.
      In every case the expansion does the work and the refit partially undoes
      it, so none of these is a reparameterisation carrying the term's credit.

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python -u \\
     experiments/exp57_expansion_term/stage9_figures.py
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
import model as M                                        # noqa: E402
import stage0_cost as S0                                 # noqa: E402
from hongshao.plotting import save_fig, set_style        # noqa: E402
from hongshao.qa import _pct, _tex                       # noqa: E402

FIGDIR = HERE / "figures"
OUT = HERE / "outputs"
FITDIR = OUT / "stage3_fits"
C_HOM, C_CORE, C_DATA = "#0072B2", "#CC79A7", "#000000"


def figure(name="exp57_mechanism"):
    import matplotlib.pyplot as plt

    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.2))
    ax = axes.ravel()
    recs, data, mask, lmh, base, theta, slow = S0.build(False)

    # ---- (a) where the moved mass goes, shell by shell ---------------- #
    a = ax[0]
    edges = np.array(G.REACH_EDGES)
    mid = np.sqrt(np.maximum(edges[:-1], 0.5) * np.minimum(edges[1:], 3e3))
    show = [("gompertz_log-E2-S2+X1", C_HOM,
             "X1 homologous (factor bounded at 2.47x)"),
            ("gompertz_log-E2-S2+X3#smallcore", C_CORE,
             "X3 core-only, Rc = 2.6 kpc")]
    w = 0.34
    for i, (lab, col, txt) in enumerate(show):
        f = OUT / f"stage2_gates_{lab}.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        sh = np.asarray(z["g2_shells"])[-1]        # the long baseline
        tot = float(np.abs(sh).sum()) / 2.0        # drained == delivered
        a.bar(np.arange(len(sh)) + (i - 0.5) * w, sh / max(tot, 1e-30), w,
              color=col, label=_tex(txt))
    a.axhline(0.0, color="0.4", lw=0.9)
    a.axvline(3.5, color="#D55E00", ls=":", lw=1.4)
    a.text(3.6, 0.62, _tex("50 kpc — the gate"), fontsize=7.5,
           color="#D55E00")
    a.set_xticks(np.arange(len(mid)))
    # NEVER hand `_tex` a string that already contains math mode: `_tex`
    # replaces a bare `>` with `$>$`, so `f"$>${x}"` becomes `$$>$$500`, which
    # under usetex renders as a 3400-pixel broken box and expands the whole
    # canvas through `savefig.bbox="tight"`. Plain words instead.
    a.set_xticklabels([_tex(f"{edges[i]:g}-{edges[i+1]:g}"
                            if edges[i + 1] < 1e4 else f"beyond {edges[i]:g}")
                       for i in range(len(mid))], rotation=35, ha="right",
                      fontsize=7.5)
    a.set_xlabel(_tex("shell [kpc]"))
    a.set_ylabel(_tex("mass moved into the shell / total mass moved"))
    a.set_title("(a) where the moved mass comes FROM and goes TO\n"
                "redshift 2 to 0.4; negative = drained, positive = delivered",
                fontsize=9.5)
    a.legend(fontsize=8, loc="upper left")

    # ---- (b) the trade curve ----------------------------------------- #
    b = ax[1]
    tf = OUT / "stage8_tradeoff.npz"
    if tf.exists():
        z = np.load(tf)
        for tag, ls, nm in (("smallcore", "-", "anchored at the small-core fit"),
                            ("bigcore", "--", "anchored at the large-core fit")):
            if f"{tag}_rc" not in z:
                continue
            rc = z[f"{tag}_rc"]
            short = "small-core anchor" if tag == "smallcore" else "large-core anchor"
            b.plot(rc, z[f"{tag}_b50"], ls, color="#D55E00", lw=2.0,
                   marker="o", ms=3.5, label=_tex(f"G2 reach, {short}"))
            b.plot(rc, np.abs(z[f"{tag}_core_gap"]), ls, color="#009E73",
                   lw=2.0, marker="s", ms=3.5,
                   label=_tex(f"G5(c) core gap, {short}"))
            b.plot(rc, z[f"{tag}_span_not"], ls, color="#7570B3", lw=2.0,
                   marker="^", ms=3.5,
                   label=_tex(f"G5(b) non-decliner span, {short}"))
        b.axhline(G.G2_MAX_BEYOND_50, color="#D55E00", ls=":", lw=1.2)
        b.axhline(0.0455, color="#009E73", ls=":", lw=1.2)
        b.axhline(0.060, color="#7570B3", ls=":", lw=1.2)
        b.axvspan(4.0, 40.0, color="#009E73", alpha=0.07, lw=0)
        b.axvspan(0.4, 3.0, color="#7570B3", alpha=0.07, lw=0)
        b.text(1.1, 0.55, _tex("(b) ok\n(c) fails"), fontsize=7.5,
               color="#7570B3", ha="center")
        b.text(12.0, 0.55, _tex("(c) ok\n(b) fails"), fontsize=7.5,
               color="#009E73", ha="center")
        b.set_xscale("log")
        b.set_yscale("log")
        b.set_ylim(2e-3, 1.5)
    b.set_xlabel(_tex("core radius Rc [kpc]  (everything else HELD FIXED)"))
    b.set_ylabel(_tex("gate value; dotted = its limit"))
    b.set_title("(b) the trade, mapped — a CONDITIONAL scan\n"
                "the reach gate holds below ~40 kpc, so it is not what binds",
                fontsize=9.5)
    b.legend(fontsize=6.6, loc="lower right", ncol=2, frameon=True,
             framealpha=0.92, edgecolor="none")

    # ---- (c) the deposition schedule --------------------------------- #
    c = ax[2]
    sp = X.XSpec(base, "X1")
    pr = X.ExpandingProblem(sp, recs, data, mask, epochs=(0, 1, 2, 3, 4))
    eff, size, shape = base.unpack(theta)
    d = pr.dep
    ok = np.isfinite(d.r200c) & (d.r200c > 0)
    dm = np.where(ok, 10.0 ** np.clip(M.log_eps(base, eff, d), -30, 10)
                  * d.dmh, 0.0) * pr.em[:, 0, :]
    dmh = np.where(ok, d.dmh, 0.0) * pr.em[:, 0, :]
    zb = np.array([0, 0.5, 1, 1.5, 2, 3, 5, 20])
    ss = np.array([dm[(d.z >= zb[i]) & (d.z < zb[i + 1])].sum()
                   for i in range(len(zb) - 1)])
    hh = np.array([dmh[(d.z >= zb[i]) & (d.z < zb[i + 1])].sum()
                   for i in range(len(zb) - 1)])
    xx = np.arange(len(ss))
    c.bar(xx - 0.19, ss / ss.sum(), 0.36, color="#0072B2",
          label=_tex("exp54's STELLAR deposits"))
    c.bar(xx + 0.19, hh / hh.sum(), 0.36, color="0.7",
          label=_tex("the halo's own mass growth"))
    c.set_xticks(xx)
    c.set_xticklabels([_tex(f"{zb[i]:g}-{zb[i+1]:g}") for i in range(len(ss))],
                      fontsize=8)
    c.set_xlabel(_tex("redshift the mass arrived"))
    c.set_ylabel(_tex("share of the total"))
    lo = (d.z < 1.0)
    c.set_title(f"(c) why exp54 is not the old model\n"
                f"it deposits {100 * dm[lo].sum() / ss.sum():.0f}"
                f"{_pct()} of its stars below z=1; the transport kernel "
                f"deposited 1.5{_pct()}", fontsize=9.5)
    c.legend(fontsize=8)

    # ---- (d) attribution --------------------------------------------- #
    dd = ax[3]
    sf = OUT / "stage7_summary.npz"
    if sf.exists():
        z = np.load(sf, allow_pickle=True)
        labs = [str(t).replace("gompertz_log-E2-S2+", "").replace("#", " ")
                for t in z["attr_tags"]]
        null_gap = float(z["null_core_gap"])
        exp_only = z["attr_core_full"] - z["attr_core_off"]
        refit = z["attr_core_off"] - null_gap
        yy = np.arange(len(exp_only))
        dd.barh(yy - 0.19, exp_only, 0.36, color="#0072B2",
                label=_tex("the EXPANSION alone"))
        dd.barh(yy + 0.19, refit, 0.36, color="#E69F00",
                label=_tex("re-optimising the other seven alone"))
        dd.set_yticks(yy)
        dd.set_yticklabels([_tex(l) for l in labs], fontsize=8)
        dd.axvline(0.0, color="0.4", lw=0.9)
    dd.set_xlabel(_tex("change in the core-density gap [dex]"))
    dd.set_title("(d) the mechanism did the work; the refit undid some of it",
                 fontsize=9.5)
    dd.legend(fontsize=8, loc="lower left")

    fig.suptitle(_tex(
        "exp57 — the MECHANISM behind the verdict. Homologous expansion takes "
        "mass from the right place (inside 10 kpc) and puts it in the wrong "
        "place (beyond 50 kpc);\nthe core-only remap takes it from the same "
        "place and moves it a few kpc. 2397 galaxies, exp54's incumbent as the "
        "base model."), fontsize=9.0, y=1.0)
    fig.tight_layout()
    paths = save_fig(fig, FIGDIR / name)
    print("wrote " + ", ".join(str(p) for p in paths))
    plt.close(fig)


if __name__ == "__main__":
    figure()
