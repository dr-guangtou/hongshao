"""Pedagogical figures for tech note 1 — the statistical emulator.

Run: PYTHONPATH=. uv run python doc/tech_note/scripts/make_figures_note1.py

Figures (doc/tech_note/figures/):
  note1_mah_family        what the four DiffMAH numbers encode (curve families)
  note1_pca_modes         what each PCA shape mode does to the growth curve
  note1_pca_sweep.gif     the same sweep animated
  note1_heteroscedastic   halo-dependent scatter + why sampling (not the mean)
                          reproduces the population
  note1_ar1_latent        the AR(1)-in-epoch latent: coherent multi-epoch draws
  note1_block_pinning     the block representation: monotone by construction
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from common import (ANCHOR_Z, FIGDIR, T_OBS, load_catalog, save_fig,
                    set_style, z_of_t)

RHO = 0.62                       # measured AR(1) epoch correlation (poly2)
SWEEP_COLORS = [plt.cm.plasma(v) for v in np.linspace(0.05, 0.85, 5)]
SWEEP_PCT = [5, 25, 50, 75, 95]


# --------------------------------------------------------------------------- #
# 1. the DiffMAH feature anatomy                                              #
# --------------------------------------------------------------------------- #
def fig_mah_family():
    from hongshao.diffmah import log_mah
    from common import MEDIAN_HALO
    sweeps = [("logtc", r"transition time $\log_{10} t_c$", [0.33, 0.97]),
              ("early", r"early index $\alpha_{\rm early}$", [1.16, 3.39]),
              ("late", r"late index $\alpha_{\rm late}$", [0.05, 2.87])]
    t = np.geomspace(0.4, T_OBS, 200)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0), sharey=True)
    for ax, (key, label, rng) in zip(axes, sweeps):
        vals = np.linspace(rng[0], rng[1], 5)
        for v, color in zip(vals, SWEEP_COLORS):
            p = dict(MEDIAN_HALO)
            p[key] = v
            lm = log_mah(np.log10(t), np.atleast_1d(p["logmp"]),
                         np.atleast_1d(p["logtc"]), np.atleast_1d(p["early"]),
                         np.atleast_1d(p["late"]), np.log10(T_OBS))[0]
            ax.plot(t, lm, color=color, lw=1.7, label=f"{v:.2f}")
        ax.set(xscale="log", xlabel=r"cosmic time $t$ [Gyr]", title=label,
               ylim=(9.8, 13.6))
        ax.legend(fontsize=7, title="value", loc="lower right")
    axes[0].set_ylabel(r"$\log_{10} M_h(t)\ [M_\odot]$")
    fig.suptitle("the DiffMAH encoding: four numbers per halo history "
                 r"(anchored at $\log_{10} M_h = 13.3$ at $z=0.4$)", y=1.02)
    fig.tight_layout()
    print("wrote", save_fig(fig, FIGDIR / "note1_mah_family")[0])
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 2. the PCA shape modes of the growth curve                                  #
# --------------------------------------------------------------------------- #
def _pca_shapes():
    X, cog, radii = load_catalog()
    anchor = cog[:, -1]
    shape = cog[:, :-1] - anchor[:, None]
    mean_shape = shape.mean(0)
    dev = shape - mean_shape
    _, sval, Vt = np.linalg.svd(dev, full_matrices=False)
    modes = Vt[:3]
    scores = dev @ modes.T
    var_frac = sval ** 2 / (sval ** 2).sum()
    return radii[:-1], mean_shape, modes, scores, var_frac


def fig_pca_modes():
    r, mean_shape, modes, scores, var_frac = _pca_shapes()
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0), sharey=True)
    for k, ax in enumerate(axes):
        pcts = np.percentile(scores[:, k], SWEEP_PCT)
        for a, pct, color in zip(pcts, SWEEP_PCT, SWEEP_COLORS):
            frac = 10.0 ** (mean_shape + a * modes[k])
            ax.loglog(r, frac, color=color, lw=1.7,
                      label=rf"{pct}th pct ($a_{k + 1}={a:+.2f}$)")
        ax.loglog(r, 10.0 ** mean_shape, color="0.15", lw=1.2, ls=":",
                  label="population mean")
        ax.set(xlabel=r"$R$ [kpc]",
               title=rf"PC{k + 1} ({100 * var_frac[k]:.0f}$\%$ of "
                     "shape variance)")
        ax.legend(fontsize=7, loc="lower right")
        ax.axhline(0.5, color="0.85", lw=0.8, zorder=0)
    axes[0].set_ylabel(r"enclosed fraction $M_\star(<R)\,/\,M_{\star,\rm tot}$")
    fig.suptitle("what each growth-curve shape mode does "
                 r"(TNG300 $z=0.4$, amplitude removed)", y=1.02)
    fig.tight_layout()
    print("wrote", save_fig(fig, FIGDIR / "note1_pca_modes")[0])
    plt.close(fig)


def fig_pca_sweep_gif():
    r, mean_shape, modes, scores, var_frac = _pca_shapes()
    sd = scores.std(0)
    n_frame = 48
    phase = np.sin(2.0 * np.pi * np.arange(n_frame) / n_frame)
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), sharey=True)
    lines, texts = [], []
    for k, ax in enumerate(axes):
        ax.loglog(r, 10.0 ** mean_shape, color="0.6", lw=1.2, ls=":")
        ln, = ax.loglog(r, 10.0 ** mean_shape, color="#0072B2", lw=2.0)
        lines.append(ln)
        texts.append(ax.text(0.04, 0.92, "", transform=ax.transAxes,
                             fontsize=9))
        ax.set(xlabel=r"$R$ [kpc]", ylim=(6e-3, 1.3),
               title=rf"PC{k + 1} ({100 * var_frac[k]:.0f}$\%$ of variance)")
    axes[0].set_ylabel(r"$M_\star(<R)\,/\,M_{\star,\rm tot}$")
    fig.tight_layout()

    def update(frame):
        for k, (ln, tx) in enumerate(zip(lines, texts)):
            a = 2.0 * sd[k] * phase[frame]
            ln.set_ydata(10.0 ** (mean_shape + a * modes[k]))
            tx.set_text(rf"$a_{k + 1} = {a:+.2f}$"
                        rf"  (${a / sd[k]:+.1f}\,\sigma$)")
        return [*lines, *texts]

    anim = FuncAnimation(fig, update, frames=n_frame, blit=False)
    out = FIGDIR / "note1_pca_sweep.gif"
    anim.save(out, writer=PillowWriter(fps=12), dpi=100)
    print("wrote", out)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 3. heteroscedastic conditional Gaussian + generative use                    #
# --------------------------------------------------------------------------- #
def fig_heteroscedastic():
    from hongshao.emulator import fit
    from hongshao.profile_emulator import aperture_targets
    X, cog, radii = load_catalog()
    Y = aperture_targets(cog, radii, [10.0, 30.0, 50.0, 100.0])
    good = np.isfinite(Y).all(1)
    X, Y = X[good], Y[good]
    emu = fit(X, Y)
    mu, sigma, _ = emu.predict(X)
    draws = emu.sample(X, size=40, rng=0)
    j = 3                                        # the 50-100 kpc annulus
    late = X[:, 3]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.4, 4.1))
    ax.scatter(late, sigma[:, j], s=4, c="#0072B2", alpha=0.4, lw=0)
    ax.set(xlabel=r"halo feature: late accretion index $\alpha_{\rm late}$",
           ylabel=r"predicted scatter $\sigma_j(X)$ [dex]",
           title=r"the scatter is halo-dependent "
                 r"($M_\star$(50--100 kpc) bin)")
    med = np.median(sigma[:, j])
    ax.axhline(med, color="0.6", lw=0.9, ls="--")
    ax.text(0.97, med * 1.03, "population median", fontsize=8, color="0.4",
            ha="right", transform=ax.get_yaxis_transform())

    bins = np.linspace(Y[:, j].min(), Y[:, j].max(), 45)
    bx.hist(Y[:, j], bins=bins, density=True, color="0.75",
            label="TNG300 truth")
    bx.hist(mu[:, j], bins=bins, density=True, histtype="step", lw=2.0,
            color="#D55E00", label="conditional means (under-dispersed)")
    bx.hist(draws[:, :, j].ravel(), bins=bins, density=True, histtype="step",
            lw=2.0, color="#0072B2", label="draws (generative path)")
    bx.set(xlabel=r"$\log_{10} M_\star$(50--100 kpc) $[M_\odot]$",
           ylabel="probability density",
           title="use it generatively: draws restore the population")
    bx.legend(fontsize=8)
    fig.tight_layout()
    print("wrote", save_fig(fig, FIGDIR / "note1_heteroscedastic")[0])
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 4. the AR(1)-in-epoch latent                                                #
# --------------------------------------------------------------------------- #
def fig_ar1_latent():
    rng = np.random.default_rng(3)
    E = len(ANCHOR_Z)
    idx = np.arange(E)
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9),
                             gridspec_kw=dict(width_ratios=[1.05, 1, 1]))
    a, b, c = axes
    C = RHO ** np.abs(idx[:, None] - idx[None, :])
    im = a.imshow(C, cmap="cividis", vmin=0, vmax=1)
    for i in range(E):
        for k in range(E):
            a.text(k, i, f"{C[i, k]:.2f}", ha="center", va="center",
                   fontsize=8, color="w" if C[i, k] < 0.55 else "k")
    a.set_xticks(idx, [f"{z:g}" for z in ANCHOR_Z])
    a.set_yticks(idx, [f"{z:g}" for z in ANCHOR_Z])
    a.set(xlabel=r"$z$", ylabel=r"$z$",
          title=rf"$\rho^{{|i-j|}}$ with $\rho={RHO}$ (measured)")
    fig.colorbar(im, ax=a, shrink=0.85)

    for ax, rho, label in ((b, 0.0, r"independent epochs ($\rho=0$)"),
                           (c, RHO, rf"AR(1) latent ($\rho={RHO}$)")):
        L = np.linalg.cholesky(rho ** np.abs(idx[:, None] - idx[None, :])
                               + 1e-12 * np.eye(E))
        eps = (L @ rng.standard_normal((E, 14))).T
        for e in eps:
            ax.plot(ANCHOR_Z, e, "-o", ms=3, lw=1.1, alpha=0.75)
        ax.set(xlabel=r"redshift $z$", ylim=(-3.2, 3.2), title=label)
        ax.invert_xaxis()
        ax.axhline(0.0, color="0.85", lw=0.8, zorder=0)
    b.set_ylabel(r"standardized residual $\varepsilon$")
    fig.suptitle("one latent per galaxy, correlated across epochs: a "
                 "scattered-high galaxy stays high in its whole history",
                 y=1.02)
    fig.tight_layout()
    print("wrote", save_fig(fig, FIGDIR / "note1_ar1_latent")[0])
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 5. the block representation                                                 #
# --------------------------------------------------------------------------- #
def fig_block_pinning():
    X, cog, radii = load_catalog()
    i = int(np.argmin(np.abs(cog[:, -1] - 11.5)))     # a typical massive one
    c = cog[i]
    edges = [2.0, 10.0, 30.0, 50.0, 100.0, radii[-1]]
    idx = [int(np.argmin(np.abs(radii - e))) for e in edges]
    total = 10.0 ** c[-1]
    shades = ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 4.1))
    ax.semilogx(radii, 10.0 ** c / total, color="0.15", lw=2.0, zorder=5)
    for k, (a_i, b_i) in enumerate(zip(idx[:-1], idx[1:])):
        frac = (10.0 ** c[b_i] - 10.0 ** c[a_i]) / total
        ax.axvspan(radii[a_i], radii[b_i], color=shades[k], alpha=0.16)
        ax.text(np.sqrt(radii[a_i] * radii[b_i]), 0.055,
                rf"$b_{k + 1}$" "\n" rf"{100 * frac:.0f}$\%$", ha="center",
                fontsize=8, color=shades[k])
    cen = 10.0 ** c[0] / total
    ax.text(radii[0] * 1.05, 0.9, r"central $M_\star(<2\,{\rm kpc})$"
            rf" $= {100 * cen:.0f}\%$ of the total", fontsize=8)
    ax.set(xlabel=r"$R$ [kpc]",
           ylabel=r"$M_\star(<R)\,/\,M_{\star,\rm tot}$", ylim=(0, 1.05),
           title="the block representation of one galaxy")

    rng = np.random.default_rng(0)
    for s in range(25):
        jitter = rng.normal(0.0, 0.18, len(idx) - 1)
        blocks = np.array([(10.0 ** c[b_i] - 10.0 ** c[a_i])
                           for a_i, b_i in zip(idx[:-1], idx[1:])])
        blocks = blocks * 10.0 ** jitter
        shells = []
        for k, (a_i, b_i) in enumerate(zip(idx[:-1], idx[1:])):
            w = np.diff(10.0 ** c[a_i:b_i + 1])
            shells.append(w / w.sum() * blocks[k])
        cum = 10.0 ** c[0] + np.concatenate([[0.0],
                                             np.cumsum(np.concatenate(shells))])
        cum = cum / cum[-1]
        bx.semilogx(radii[:len(cum)], cum, color="#0072B2", lw=0.8, alpha=0.4)
    bx.semilogx(radii, 10.0 ** c / total, color="0.15", lw=2.0,
                label="the galaxy")
    bx.plot([], [], color="#0072B2", lw=1.0, alpha=0.7,
            label="25 draws of the block masses")
    bx.set(xlabel=r"$R$ [kpc]", ylabel=r"$M_\star(<R)\,/\,M_\star(<148)$",
           ylim=(0, 1.05),
           title="draw the blocks, not the curve: monotone by construction")
    bx.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    print("wrote", save_fig(fig, FIGDIR / "note1_block_pinning")[0])
    plt.close(fig)


if __name__ == "__main__":
    set_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig_mah_family()
    fig_pca_modes()
    fig_heteroscedastic()
    fig_ar1_latent()
    fig_block_pinning()
    fig_pca_sweep_gif()
