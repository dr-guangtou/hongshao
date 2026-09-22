"""exp85 — the one figure that shows why the compact channel's expansion is
not the centre's mechanism (`figures/qa/exp85_probe_summary`):
  (a) the probe: the central change z = 2 -> 0.4 at 4.9 kpc for the galaxies
      whose true centre declined and for the rest — TNG (dashed) against the
      adopted mean (the value at 0) and each probe value of q_c (age driven)
      and q_ch (halo driven), the compact constants re-tuned; the split
      between the two groups never moves;
  (b) the anatomy of the model's centre by group at z = 0.4: how much of the
      4.9 kpc mass is the compact channel, how old its stars are, and how
      much the knob lowers the 4.9 kpc mass — the decliners hold LESS
      compact mass, so the knob acts on the wrong galaxies;
  (c) per galaxy: the model's response to q_c against TNG's central change —
      the correlation has the wrong sign.
Reads `outputs/stage0_probe.npz` and `outputs/stage0_anatomy.npz`; no build.
Run: uv run python -u experiments/exp85_compact_expansion/stage0_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from hongshao import qa                                  # noqa: E402

FIGDIR = HERE / "figures/qa"                             # the experiment's own figures, never the repo-level figures/qa
C_TRUTH, C_DEC, C_NON, C_Q, C_QH = "#777777", "#D55E00", "#0072B2", "#000000", "#999999"


def probe_table(z):
    keys = [str(k) for k in z["gate_keys"]]
    probes = [str(p) for p in z["probe_keys"]]
    g = np.asarray(z["gates"], float)
    return {p: dict(zip(keys, g[i])) for i, p in enumerate(probes)}


def main():
    zp = np.load(HERE / "outputs/stage0_probe.npz", allow_pickle=True)
    za = np.load(HERE / "outputs/stage0_anatomy.npz", allow_pickle=True)
    tab = probe_table(zp)
    ref = tab["adopted"]
    plt.rcParams.update({"axes.labelsize": 10, "axes.titlesize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5})
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))

    # (a) the probe
    ax = axes[0]
    for knob, col, mk, lab in (("q_c", C_Q, "o", "age driven, $q_c$"), ("q_ch", C_QH, "s", "halo driven, $q_{ch}$")):
        pts = sorted((float(p.split("=")[1]), tab[p]) for p in tab if p.startswith(knob + " ="))
        x = [0.0] + [v for v, _ in pts]
        for grp, c in (("dec", C_DEC), ("non", C_NON)):
            y = [ref[f"{grp}_change_model"]] + [g[f"{grp}_change_model"] for _, g in pts]
            ax.plot(x, y, marker=mk, color=c, ls="-" if knob == "q_c" else ":", lw=1.6, ms=6,
                    mfc=c if knob == "q_c" else "white", label=None)
    ax.axhline(ref["dec_change_truth"], color=C_DEC, ls="--", lw=1.4)
    ax.axhline(ref["non_change_truth"], color=C_NON, ls="--", lw=1.4)
    ax.text(0.82, ref["dec_change_truth"] + 0.006, "TNG, declining centres", color=C_DEC, fontsize=9, ha="right")
    ax.text(0.82, ref["non_change_truth"] + 0.006, "TNG, the rest", color=C_NON, fontsize=9, ha="right")
    ax.axhline(0, color="k", lw=0.6)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=C_DEC, lw=1.6, label="model, declining centres (42" + qa._pct() + ")"),
               Line2D([], [], color=C_NON, lw=1.6, label="model, the rest"),
               Line2D([], [], color="k", marker="o", lw=1.6, label="age driven $q_c$ (filled)"),
               Line2D([], [], color="k", marker="s", ls=":", mfc="white", lw=1.6, label="halo driven $q_{ch}$ (open)")]
    ax.legend(handles=handles, fontsize=8.5, loc="lower left", frameon=False)
    ax.set_xlabel("expansion exponent (0 = the adopted mean)")
    ax.set_ylabel(qa._tex("median change of log M*(<4.9 kpc), z = 2 to 0.4 [dex]"))
    ax.set_title("(a) the probe: the two groups never separate", fontsize=11)
    ax.set_xlim(-0.03, 0.85)
    ax.set_ylim(-0.16, 0.2)

    # (b) the anatomy at z = 0.4
    ax = axes[1]
    dec = np.asarray(za["dec"], bool)
    share = za["mc_0"] / (za["mc_0"] + za["me_0"])
    rows = [("compact share of\nM(<4.9 kpc)", share, 1.0),
            ("age of its stars,\nlog(t$_{obs}$/t') [dex]", za["age_in_0"], 1.0),
            ("response to $q_c$ = 0.3,\n$-$dlog M(<4.9) [dex]", -za["resp_0"], 1.0)]
    xs = np.arange(len(rows))
    for j, (grp, c, off) in enumerate((("declining centres", C_DEC, -0.18), ("the rest", C_NON, 0.18))):
        sel = dec if j == 0 else ~dec
        vals = [np.median(v[sel]) for _, v, _ in rows]
        lo = [np.percentile(v[sel], 25) for _, v, _ in rows]
        hi = [np.percentile(v[sel], 75) for _, v, _ in rows]
        ax.bar(xs + off, vals, width=0.34, color=c, alpha=0.85, label=grp)
        ax.errorbar(xs + off, vals, yerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)], fmt="none", ecolor="k", lw=0.9)
        for xi, v in zip(xs + off, vals):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([qa._tex(r[0]) for r in rows], fontsize=9)
    ax.set_ylabel("median (bars) and quartiles, z = 0.4")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.set_title("(b) the model's centre: the decliners hold less compact mass", fontsize=11)
    ax.set_ylim(0, 1.15)

    # (c) per galaxy
    ax = axes[2]
    x = np.asarray(za["dch_truth"], float)
    y = np.asarray(za["resp_0"], float) - np.asarray(za["resp_4"], float)
    ax.scatter(x, y, s=5, color="#444444", alpha=0.35, rasterized=True)
    q = np.quantile(x, np.linspace(0, 1, 13))
    xm = [np.median(x[(x >= q[i]) & (x <= q[i + 1])]) for i in range(12)]
    ym = [np.median(y[(x >= q[i]) & (x <= q[i + 1])]) for i in range(12)]
    ax.plot(xm, ym, color=C_DEC, lw=2.2, label="running median")
    from scipy.stats import spearmanr
    rho = spearmanr(x, y)[0]
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel(qa._tex("TNG's change of log M*(<4.9 kpc), z = 2 to 0.4 [dex] (declines negative)"))
    ax.set_ylabel(qa._tex("what $q_c$ = 0.3 does to the model's change [dex]"))
    ax.set_title(f"(c) per galaxy: the wrong sign (rank correlation {rho:+.2f})", fontsize=11)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.set_xlim(-0.8, 0.8)
    fig.suptitle("exp85: an age-driven expansion of the compact deposits cannot produce TNG's central declines in the adopted model",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"exp85_probe_summary.{ext}", dpi=150)
    plt.close(fig)
    print(f"wrote {FIGDIR / 'exp85_probe_summary.png'}")


if __name__ == "__main__":
    main()
