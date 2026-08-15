"""exp48 — what the candidate deposit profiles actually look like.

Every family is drawn at the SAME half-mass radius and the same total
mass, so the comparison is about SHAPE alone. Without that normalization
a cuspier profile would look better in the centre simply by being smaller
overall, which is not the question.

Run: PYTHONPATH=. uv run python experiments/exp48_objective_profile/\
profile_figure.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

FIGDIR = HERE / "figures"
R50_TARGET = 10.0                       # kpc, all families matched to this
OK = ["#000000", "#D55E00", "#E69F00", "#009E73", "#0072B2", "#CC79A7"]


def cog_of(family, par, r):
    """Unit-total CoG on radius grid r for one family (scale rc = 1)."""
    from scipy.special import betainc, gammainc
    from profiles import b_n
    if family == "moffat":
        gam = par
        return 1.0 - (1.0 + r ** 2) ** (1.0 - gam)
    if family == "cuspy":
        gam, a = par
        p, q = 1.0 - a / 2.0, gam - 1.0 + a / 2.0
        X = r ** 2
        return betainc(p, q, X / (1.0 + X))
    if family == "sersic":
        n = par
        return gammainc(2.0 * n, (r * b_n(n)) ** (1.0 / n))
    raise ValueError(family)


def matched(family, par, R):
    """CoG and Sigma on R [kpc], rescaled so R50 = R50_TARGET."""
    u = np.geomspace(1e-4, 1e4, 4000)
    c = cog_of(family, par, u)
    r50_u = float(np.interp(0.5, c, u))          # R50 in scale units
    s = R50_TARGET / r50_u                        # kpc per scale unit
    cog = np.interp(R / s, u, c)
    # Sigma = (1/2 pi R) dM/dR, differentiated on the log grid
    dM = np.gradient(cog, R)
    return cog, np.clip(dM / (2.0 * np.pi * R), 1e-30, None)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hongshao import plotting, qa

    plotting.set_style()
    R = np.geomspace(0.05, 300.0, 600)
    cases = [("moffat", 1.38, r"Moffat $\gamma=1.38$ (ADOPTED)", OK[0], "-"),
             ("cuspy", (1.38, 0.5), r"cuspy $\alpha=0.5$", OK[1], "-"),
             ("cuspy", (1.38, 1.0), r"cuspy $\alpha=1.0$", OK[2], "-"),
             ("cuspy", (1.38, 1.5), r"cuspy $\alpha=1.5$", OK[3], "-"),
             ("sersic", 1.0, "Sersic $n=1$", OK[4], "--"),
             ("sersic", 4.0, "Sersic $n=4$", OK[5], "--")]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    frac5 = {}
    for fam, par, lab, col, ls in cases:
        cog, sig = matched(fam, par, R)
        axes[0].plot(R, sig, color=col, ls=ls, lw=2.0, label=lab)
        axes[1].plot(R, cog, color=col, ls=ls, lw=2.0)
        frac5[lab] = float(np.interp(5.0, R, cog))

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("R [kpc]")
    axes[0].set_ylabel(r"$\Sigma(R)$  [arbitrary, same total mass]")
    axes[0].set_title("1. the deposit's surface density", fontsize=10)
    axes[0].set_ylim(1e-8, 1e1)
    axes[0].legend(fontsize=7.5, loc="lower left", framealpha=0.9)
    axes[0].text(0.97, 0.96, "ALL matched to the same\n"
                 r"$R_{50}=10$ kpc and total mass",
                 transform=axes[0].transAxes, fontsize=7.5, va="top",
                 ha="right",
                 bbox=dict(fc="white", ec="0.8", alpha=0.9, pad=3))

    axes[1].set_xscale("log")
    axes[1].axvline(5.0, color="0.5", ls=":", lw=1.2)
    axes[1].text(5.4, 0.05, "5 kpc", fontsize=7.5, color="0.4")
    axes[1].set_xlabel("R [kpc]")
    axes[1].set_ylabel(r"$M(<R)/M_{\rm tot}$")
    axes[1].set_title("2. the curve of growth", fontsize=10)
    axes[1].set_ylim(0, 1)

    # panel 3: the lever — central mass fraction vs alpha
    al = np.linspace(0.0, 1.8, 40)
    f5 = []
    for a in al:
        cog, _ = matched("cuspy", (1.38, a), R)
        f5.append(float(np.interp(5.0, R, cog)))
    f5 = np.array(f5)
    axes[2].plot(al, f5 / f5[0], color=OK[1], lw=2.4)
    axes[2].axhline(1.0, color=OK[0], lw=1.5, ls="-")
    axes[2].text(0.05, 1.02, "Moffat (alpha=0)", fontsize=8, color=OK[0])
    # the measured shortfall this is meant to close
    axes[2].axhline(1.0 / (1.0 - 0.436), color="0.35", ls="--", lw=1.4)
    axes[2].text(0.05, 1.0 / (1.0 - 0.436) + 0.03,
                 "x1.77 = what compact galaxies need\n"
                 "(exp47: M*(<5 kpc) is -43.6% at z=0.4)",
                 fontsize=7.5, color="0.25")
    axes[2].set_xlabel(r"cusp slope $\alpha$")
    axes[2].set_ylabel(r"$M(<5\,{\rm kpc})$ relative to Moffat")
    axes[2].set_title(r"3. the lever $\alpha$ provides", fontsize=10)

    fig.suptitle("exp48 — candidate deposit profiles at fixed "
                 r"$R_{50}$: only the SHAPE differs", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FIGDIR.mkdir(exist_ok=True)
    plotting.save_fig(fig, str(FIGDIR / "exp48_profiles"))
    plt.close(fig)

    print("M(<5 kpc)/Mtot at fixed R50 = 10 kpc:")
    for k, v in frac5.items():
        base = frac5[r"Moffat $\gamma=1.38$ (ADOPTED)"]
        print(f"   {k:<34s} {v:.4f}   ({v / base:.2f}x Moffat)")
    need = 1.0 / (1.0 - 0.436)
    hit = al[np.argmin(np.abs(f5 / f5[0] - need))]
    print(f"\n   compact galaxies are 43.6% short => need {need:.2f}x, "
          f"reached near alpha = {hit:.2f}")
    print(f"   wrote {FIGDIR / 'exp48_profiles'}.png/.pdf")


if __name__ == "__main__":
    main()
