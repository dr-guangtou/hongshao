"""Tech note 04, figure 04b: the Sersic curve of growth and the deposition
closed form are the SAME one-dimensional integral (a Gamma density) taken over
different variables — and are different functions of radius.

  Sersic       M(<R)/M_tot = P(2n, b_n (R/R_e)^(1/n))
               = the cumulative of a Gamma(2n) density in u = b_n (R/R_e)^(1/n),
               integrated over RADIUS up to R.
  deposition   M(<R) = A R^beta (ln2)^(-q) c^-1 gamma(q, X),  q = beta/c,
               X = ln2 (s_max/R)^c  (power-law history, s_min -> 0)
               = the cumulative of a Gamma(q) density in X, integrated over
               DEPOSIT SIZE at fixed R; the radius enters through (s/R)^c.

Run: PYTHONPATH=. uv run python doc/tech_note/scripts/04b_sersic_vs_deposition.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.special import gammainc, gammaincinv, gamma as Gamma, gammaincc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hongshao.plotting import set_style, save_fig
from hongshao.qa import _tex

OUT = Path(__file__).resolve().parents[1] / "figures" / "04b_sersic_vs_deposition"


def sersic_cog(x, n):
    """M(<R)/M_tot for a Sersic profile, x = R/R_e."""
    b = gammaincinv(2 * n, 0.5)
    return gammainc(2 * n, b * x ** (1.0 / n))


def deposition_cog(R, beta, c, s_max, s_min=1e-6):
    """M(<R)/M_tot for the closed form (tech note 04, Eq. 5), with M_tot = A s_max^beta / beta."""
    q = beta / c
    def upper(qq, x):
        return gammaincc(qq, x) * Gamma(qq)
    X_max, X_min = np.log(2) * (s_max / R) ** c, np.log(2) * (s_min / R) ** c
    m = R ** beta * np.log(2) ** (-q) / c * (upper(q, X_min) - upper(q, X_max))
    return m / (s_max ** beta / beta)


def half_radius(R, m):
    return float(np.interp(0.5, m, R))


def main():
    set_style()
    R = np.geomspace(1e-3, 1e3, 2000)
    cases = [("Sérsic n=1", "S", 1.0, "#009E73"), ("Sérsic n=4", "S", 4.0, "#E69F00"),
             (r"deposition $\beta$=1.0, c=0.8 (incumbent-like)", "D", (1.0, 0.8), "#0072B2"),
             (r"deposition $\beta$=0.5, c=1.3 (Stage 2 extended channel)", "D", (0.5, 1.3), "#D55E00")]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for label, kind, prm, col in cases:
        if kind == "S":
            m = sersic_cog(R, prm)                     # x = R / R_e already
            x = R
        else:
            m = deposition_cog(R, prm[0], prm[1], s_max=1.0)
            x = R / half_radius(R, m)                   # in units of the profile's own half-mass radius
        ls = "-" if kind == "S" else "--"
        ax[0].loglog(x, m, ls, color=col, lw=2, label=label)
        slope = np.gradient(np.log(m), np.log(x))
        ax[1].semilogx(x, slope, ls, color=col, lw=2, label=label)
        ax[2].loglog(x, 1 - m, ls, color=col, lw=2, label=label)
    ax[0].set_xlim(1e-2, 1e2); ax[0].set_ylim(1e-4, 1.2)
    ax[0].set_xlabel(r"$R / R_{50}$"); ax[0].set_ylabel(_tex("M(<R) / M_tot"))
    ax[0].set_title("(a) the curves of growth")
    ax[0].legend(fontsize=7, loc="lower right")
    ax[1].set_xlim(1e-2, 1e2); ax[1].set_ylim(0, 2.2)
    ax[1].axhline(2, color="gray", ls=":", lw=1); ax[1].text(1.2e-2, 2.03, "finite central density: slope 2", fontsize=7, color="gray")
    ax[1].set_xlabel(r"$R / R_{50}$"); ax[1].set_ylabel(r"local slope $d\log M / d\log R$")
    ax[1].set_title(r"(b) Sérsic starts at 2; deposition starts at $\beta$")
    ax[2].set_xlim(1e-1, 1e2); ax[2].set_ylim(1e-4, 1.2)
    ax[2].set_xlabel(r"$R / R_{50}$"); ax[2].set_ylabel(_tex("1 - M(<R) / M_tot"))
    ax[2].set_title("(c) approach to the total: stretched exponential vs power law")
    ax[2].legend(fontsize=7, loc="lower left")
    fig.suptitle("the same Gamma integral, different variables: Sérsic integrates over radius, the deposition form over deposit size", fontsize=10.5)
    fig.tight_layout()
    save_fig(fig, OUT)
    print("wrote", OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
