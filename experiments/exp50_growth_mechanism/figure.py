"""exp50 figure — the model's growth mechanism versus the physical picture.

Left: the transport share of the model's own outskirt growth, epoch pair by
epoch pair. Transport = redistribution of stars already assembled; the
established picture for massive galaxies says late-time outskirt growth should
instead be dominated by ex-situ accretion in minor mergers.

Right: cumulative stellar-mass growth inside 500 kpc from z=2 to z=0.4 --
measured, against what the kernel's OWN deposition supplies. The gap is
manufactured by the M(<500) normalization, which pins the model to the data.

Numbers hard-coded from decompose.py on the full sample (n=2397); rerun that
first if the adopted theta changes.

Run: PYTHONPATH=. uv run python experiments/exp50_growth_mechanism/figure.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from hongshao.plotting import OKABE_ITO, save_fig, set_style   # noqa: E402
from hongshao.qa import _pct, _tex                             # noqa: E402

# epoch pair -> (mid redshift, transport share of M[50,100], of M[100,148])
PAIRS = [("2.0-1.5", 1.75, 18.0, 17.4), ("1.5-1.0", 1.25, 38.4, 32.8),
         ("1.0-0.7", 0.85, 76.9, 63.3), ("0.7-0.4", 0.55, 96.5, 91.7)]
# cumulative growth factors, z=2 -> 0.4
MEASURED = [1.000, 1.340, 1.830, 2.199, 2.645]
MODEL_OWN = [1.000, 1.094, 1.124, 1.124, 1.115]
ZS = [2.0, 1.5, 1.0, 0.7, 0.4]


def main():
    set_style()
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.3))
    z = [p[1] for p in PAIRS]

    ax[0].plot(z, [p[2] for p in PAIRS], "o-", color=OKABE_ITO[5], lw=2,
               label=_tex("M*[50,100] kpc"))
    ax[0].plot(z, [p[3] for p in PAIRS], "s--", color=OKABE_ITO[4], lw=2,
               label=_tex("M*[100,148] kpc"))
    ax[0].axhline(50, color="0.5", lw=1, ls=":")
    ax[0].text(1.72, 53, "transport starts to dominate", fontsize=8,
               color="0.35")
    for zz, _, a, _b in [(p[1], 0, p[2], 0) for p in PAIRS]:
        ax[0].annotate(f"{a:.0f}{_pct()}", (zz, a), textcoords="offset points",
                       xytext=(0, 9), ha="center", fontsize=8)
    ax[0].set_xlim(2.0, 0.35)
    ax[0].set_ylim(0, 108)
    ax[0].set_xlabel("redshift (midpoint of the epoch pair)")
    ax[0].set_ylabel(f"transport share of the model's own growth [{_pct()}]")
    ax[0].set_title("How the model grows the outskirts")
    ax[0].legend(loc="upper left")

    ax[1].plot(ZS, MEASURED, "o-", color="k", lw=2, label="measured (TNG300)")
    ax[1].plot(ZS, MODEL_OWN, "s-", color=OKABE_ITO[5], lw=2,
               label="the kernel's own deposition")
    ax[1].fill_between(ZS, MODEL_OWN, MEASURED, color=OKABE_ITO[0], alpha=0.30,
                       label=f"supplied by the M({_tex('<')}500) pinning")
    ax[1].set_xlim(2.05, 0.35)
    ax[1].set_xlabel("redshift")
    ax[1].set_ylabel(_tex("cumulative M*(<500 kpc) growth, relative to z=2"))
    ax[1].set_title("Where the stellar mass comes from")
    ax[1].legend(loc="upper left")
    ax[1].annotate(f"{100*np.log(MODEL_OWN[-1])/np.log(MEASURED[-1]):.0f}"
                   f"{_pct()} of the growth\nis the model's own physics",
                   (0.62, 1.55), fontsize=9, color="0.25")

    fig.suptitle("exp50 — the adopted kernel's late-time growth is "
                 "redistribution, not accretion", fontsize=11)
    fig.tight_layout()
    for p in save_fig(fig, HERE / "figures" / "exp50_growth_mechanism"):
        print("wrote", p)


if __name__ == "__main__":
    main()
