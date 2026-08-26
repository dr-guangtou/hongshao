"""exp59 — THE figure of the Stage 0b negative: five candidate trajectories,
one admissible box, no crossings.

One panel. Each conditioning variable traces a curve in the plane of the two
gate quantities — the decliner-minus-rest amplitude split at z=2 (what gate B2
requires to close) against the same split at z=0.4 (what guard B2b requires
not to grow) — as its coefficient sweeps −4 to +4. The green box is where a
candidate must land to qualify. Every trajectory runs diagonally PAST the box:
the two epochs' splits move in lockstep along every axis, which is the whole
result. The dot marks the incumbent (coefficient 0), where all five curves
meet.

Run: HONGSHAO_DATA_DIR is not needed — this reads outputs/stage0_leverage.npz.
     PYTHONPATH=. uv run python -u \\
     experiments/exp59_conditioned_efficiency/stage0_figure.py
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

#: Okabe-Ito, the project's colorblind-safe palette (qa._EPOCH_COLORS order)
COLORS = ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]
#: gate limits, matching stage0_leverage.py exactly
B2_LIM, B2B_LIM = 0.060, 0.092
LABELS = {
    "F1": "assembly state  $f_{\\rm form}-0.5$",
    "C1": "concentration excess  $\\Delta\\log c$",
    "R1": "accretion burstiness  $\\log({\\rm d}M_h/M_h)$",
    "F2": "assembly state $\\times$ deposit epoch",
    "H1": "stalled-halo switch (no leverage: the curve is a point)",
}


def main():
    z = np.load(HERE / "outputs" / "stage0_leverage.npz", allow_pickle=True)
    cands = [str(c) for c in z["cands"]]
    set_style()
    fig, ax = plt.subplots(figsize=(7.0, 5.6))

    ax.axhline(0, color="0.8", lw=0.8, zorder=0)
    ax.axvline(0, color="0.8", lw=0.8, zorder=0)
    ax.add_patch(plt.Rectangle((-B2_LIM, -B2B_LIM), 2 * B2_LIM, 2 * B2B_LIM,
                               facecolor="#009E73", alpha=0.15,
                               edgecolor="#009E73", lw=1.2, zorder=1))
    ax.annotate("the box a candidate\nmust land in\n(B2 and the B2b guard)",
                xy=(0.0, -0.055), ha="center", va="top", fontsize=8,
                color="#00694d")

    for c, color in zip(cands, COLORS):
        cur = np.asarray(z[f"curve_{c}"], float)        # (n, 4): e, s2, s0, med
        e, s2, s0 = cur[:, 0], cur[:, 1], cur[:, 2]
        ax.plot(s2, s0, "-", color=color, lw=1.6, zorder=3,
                label=f"{c}:  {LABELS[c]}")
        ax.plot(s2, s0, ".", color=color, ms=4, zorder=3)
        # arrow the direction of increasing coefficient, at the +4 end
        if c != "H1":
            ax.annotate("", xy=(s2[-1], s0[-1]),
                        xytext=(s2[-2], s0[-2]), zorder=4,
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.4))
            for k in (0, len(e) - 1):
                ax.annotate(f"{e[k]:+.0f}", xy=(s2[k], s0[k]), fontsize=7,
                            color=color, xytext=(3, 3),
                            textcoords="offset points")

    i0 = int(np.where(np.asarray(z["curve_F1"], float)[:, 0] == 0.0)[0][0])
    n2, n0 = np.asarray(z["curve_F1"], float)[i0, 1:3]
    ax.plot([n2], [n0], "o", color="black", ms=7, zorder=5)
    ax.annotate("the incumbent\n(coefficient 0)", xy=(n2, n0),
                xytext=(-14, -30), textcoords="offset points",
                ha="right", fontsize=8,
                arrowprops=dict(arrowstyle="-", color="0.4", lw=0.8))

    ax.set_xlabel("decliner-minus-rest amplitude split at z=2  [dex]\n"
                  "(measured need: close this to 0)")
    ax.set_ylabel("the same split at z=0.4  [dex]\n(must not grow)")
    ax.set_title("exp59 Stage 0b — every conditioning variable moves the two "
                 "epochs' splits in lockstep,\nso no coefficient reaches the "
                 "box: all five candidates disqualified before any fit")
    ax.legend(loc="upper left", framealpha=0.95)
    lo = min(ax.get_xlim()[0], -0.25)
    ax.set_xlim(lo, 0.20)
    paths = save_fig(fig, HERE / "figures" / "exp59_leverage")
    for p in paths:
        print(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
