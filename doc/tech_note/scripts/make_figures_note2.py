"""Pedagogical figures for tech note 2 — the 1ch-mof transport kernel.

Run: PYTHONPATH=. uv run python doc/tech_note/scripts/make_figures_note2.py

Figures (doc/tech_note/figures/):
  note2_deposit_shapes    why a power-law tail: Gaussian vs exponential vs
                          Moffat at matched half-mass radius
  note2_efficiency_window the lognormal window turning the accretion history
                          into a deposit budget
  note2_migration_clock   the alpha=1 clock: retained fraction + migrated
                          radius per birth time
  note2_assembly_epochs   one halo's profile at the five anchor epochs +
                          the z=0.4 decomposition by deposit birth era
  note2_assembly.gif      deposits arriving and migrating as cosmic time runs
  note2_stochastic_layer  the 1-D empirical layer: the heavy-tailed d_sig
                          pool and the drawn profile family
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from common import (ANCHOR_T, ANCHOR_Z, EPOCH_COLORS, FIGDIR, STAGE1_DIST_NPZ,
                    T_OBS, deposit_basis, efficiency_weights, kernel_cogs,
                    moffat_cog, save_fig, set_style, sigma_of_cog, theta_z15,
                    toy_mah, z_of_t)

R_FINE = np.geomspace(1.0, 500.0, 220)
MSTAR_NORM = 10.0 ** 11.3                       # toy M*(<500 kpc) at z=0.4
SIG_BOX = (0.05, 2.0)                           # the sig physical box (exp41)


# --------------------------------------------------------------------------- #
# 1. deposit-shape anatomy                                                    #
# --------------------------------------------------------------------------- #
def sigma_gauss(r, s):
    return np.exp(-r ** 2 / (2.0 * s ** 2)) / (2.0 * np.pi * s ** 2)


def sigma_expo(r, a):
    return np.exp(-r / a) / (2.0 * np.pi * a ** 2)


def sigma_moffat(r, rc, gam):
    return (gam - 1.0) / (np.pi * rc ** 2) * (1.0 + (r / rc) ** 2) ** (-gam)


def fig_deposit_shapes():
    gam = float(theta_z15()[5])
    r50 = 10.0                                          # match all at 10 kpc
    s = r50 / np.sqrt(2.0 * np.log(2.0))                # Gaussian scale
    a = r50 / 1.67835                                   # exponential scale
    x50 = np.sqrt(0.5 ** (1.0 / (1.0 - gam)) - 1.0)     # Moffat R50/rc
    rc = r50 / x50
    r = np.geomspace(0.5, 400.0, 400)
    shapes = [
        ("Gaussian", sigma_gauss(r, s),
         1.0 - np.exp(-r ** 2 / (2.0 * s ** 2)), "#0072B2", "--"),
        ("exponential ($n=1$)", sigma_expo(r, a),
         1.0 - (1.0 + r / a) * np.exp(-r / a), "#009E73", "-."),
        (rf"Moffat ($\gamma={gam:.2f}$, adopted)", sigma_moffat(r, rc, gam),
         moffat_cog(r, np.array([rc]), gam)[:, 0], "#D55E00", "-"),
    ]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for label, sig, cog, color, ls in shapes:
        ax.loglog(r, sig, ls, color=color, lw=1.8, label=label)
        bx.loglog(r, 1.0 - cog, ls, color=color, lw=1.8)
    ax.axvline(r50, color="0.75", lw=0.8, zorder=0)
    ax.text(r50 * 1.1, 2e-3, r"$R_{50}$", color="0.4", fontsize=9)
    ax.set(xlabel=r"$R$ [kpc]",
           ylabel=r"$\Sigma(R)$ per unit deposit mass [kpc$^{-2}$]",
           ylim=(1e-9, 1e-1),
           title="unit-mass deposit profiles, matched half-mass radius")
    ax.legend(loc="lower left")
    bx.axvline(r50, color="0.75", lw=0.8, zorder=0)
    for frac in (0.1, 0.01):
        bx.axhline(frac, color="0.85", lw=0.7, zorder=0)
    bx.set(xlabel=r"$R$ [kpc]", ylabel=r"fraction of deposit mass beyond $R$",
           ylim=(1e-6, 1.2),
           title="the tail: what feeds the outskirts")
    bx.annotate("power-law tail:\nmass reaches 100s of kpc",
                xy=(150, 4e-2), xytext=(30, 3e-4), fontsize=8, color="#D55E00",
                arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.0))
    fig.tight_layout()
    print("wrote", save_fig(fig, FIGDIR / "note2_deposit_shapes")[0])
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 2. the efficiency window                                                    #
# --------------------------------------------------------------------------- #
def fig_efficiency_window():
    th = theta_z15()
    mu, sig = th[3], th[4]
    mah = toy_mah()
    w = efficiency_weights(mah["z"], mu, sig)
    budget = w * mah["dMh"]
    budget = budget / budget.sum()
    frac_h = mah["dMh"] / mah["dMh"].sum()
    z_peak = np.expm1(mu)

    fig, axes = plt.subplots(3, 1, figsize=(6.4, 7.4), sharex=True)
    a, b, c = axes
    a.plot(mah["t"], mah["logmh"][1:], color="0.2", lw=1.8)
    a.set(ylabel=r"$\log_{10} M_h\ [M_\odot]$",
          title="from halo history to deposit budget "
                "(median-halo DiffMAH curve)")
    tg = np.geomspace(0.35, T_OBS, 400)
    b.plot(tg, efficiency_weights(z_of_t(tg), mu, sig), color="#0072B2",
           lw=1.8)
    b.axvline(np.interp(z_peak, mah["z"][::-1], mah["t"][::-1]),
              color="#0072B2", lw=0.8, ls=":")
    b.text(0.97, 0.85, rf"peak $z=e^{{\mu}}-1={z_peak:.1f}$"
           "\n" rf"$\mu={mu:.2f}$, $\sigma_z={sig:.2f}$",
           transform=b.transAxes, ha="right", fontsize=9, color="#0072B2")
    b.set(ylabel=r"efficiency window $w(z)$")
    c.bar(mah["t"], frac_h, width=np.diff(mah["t_edge"]), color="0.8",
          label=r"accretion share $\Delta M_{h,i}/\sum \Delta M_h$")
    c.bar(mah["t"], budget, width=np.diff(mah["t_edge"]), color="#D55E00",
          alpha=0.75, label=r"deposit budget $\propto w(z_i)\,\Delta M_{h,i}$")
    c.legend(loc="upper right")
    c.set(xlabel=r"cosmic time $t$ [Gyr]", ylabel="fraction per step")
    ztop = np.array([6.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.7, 0.4])
    for ax in axes:
        ax.set_xlim(0.3, T_OBS * 1.02)
    top = a.secondary_xaxis("top")
    from common import cosmic_age
    top.set_xticks(cosmic_age(ztop))
    top.set_xticklabels([f"{z:g}" for z in ztop])
    top.set_xlabel(r"redshift $z$")
    fig.tight_layout()
    print("wrote", save_fig(fig, FIGDIR / "note2_efficiency_window")[0])
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 3. the migration clock                                                      #
# --------------------------------------------------------------------------- #
def fig_migration_clock():
    th = theta_z15()
    log_rc, g, q, gam = th[0], th[1], th[2], th[5]
    x50 = np.sqrt(0.5 ** (1.0 / (1.0 - gam)) - 1.0)
    births = np.array([1.0, 2.0, 4.0, 8.0])
    colors = ["#0072B2", "#009E73", "#E69F00", "#CC79A7"]
    tg = np.linspace(0.5, T_OBS, 400)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for ti, color in zip(births, colors):
        tk = tg[tg >= ti]
        fc = np.exp(-(tk - ti) / ti)
        ax.plot(tk, fc, color=color, lw=1.8,
                label=rf"born $t_i={ti:g}$ Gyr ($z={z_of_t(ti):.1f}$)")
        rc0 = 10.0 ** log_rc * (ti / T_OBS) ** g
        rcw = rc0 * (tk / ti) ** q
        r50_mix = x50 * (fc * rc0 + (1.0 - fc) * rcw)
        bx.plot(tk, r50_mix, color=color, lw=1.8)
        bx.plot(ti, x50 * rc0, "o", color=color, ms=5)
    ax.set(xlabel=r"cosmic time $t_k$ [Gyr]",
           ylabel=r"retained fraction $f_c = e^{-(t_k-t_i)/t_i}$",
           title=r"the $\alpha=1$ migration clock ($\tau_i = t_i$)")
    ax.legend(fontsize=8)
    bx.set(xlabel=r"cosmic time $t_k$ [Gyr]",
           ylabel=r"deposit half-mass radius [kpc]", yscale="log",
           title="each deposit is born compact, then migrates outward")
    bx.text(0.03, 0.94,
            r"$r_{c,i} = 10^{\log r_c}\,(t_i/t_{\rm obs})^{g}$ at birth,"
            "\n" r"envelope at $r_{c,i}\,(t_k/t_i)^{q}$",
            transform=bx.transAxes, fontsize=9, va="top")
    fig.tight_layout()
    print("wrote", save_fig(fig, FIGDIR / "note2_migration_clock")[0])
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 4. assembly across the anchor epochs                                        #
# --------------------------------------------------------------------------- #
ERAS = [(4.0, np.inf, r"born $z>4$", "#CC79A7"),
        (2.0, 4.0, r"born $2<z<4$", "#D55E00"),
        (1.0, 2.0, r"born $1<z<2$", "#E69F00"),
        (-np.inf, 1.0, r"born $z<1$", "#0072B2")]


def fig_assembly_epochs():
    th = theta_z15()
    mah = toy_mah()
    cogs = kernel_cogs(th, mah, ANCHOR_T, R_FINE, mstar_norm=MSTAR_NORM)
    scale = MSTAR_NORM / kernel_cogs(th, mah, [ANCHOR_T[0]], R_FINE)[0, -1]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.4, 4.2))
    for k in range(len(ANCHOR_Z) - 1, -1, -1):
        sig, mid = sigma_of_cog(cogs[k], R_FINE)
        ax.loglog(mid, sig, color=EPOCH_COLORS[k], lw=1.8,
                  label=rf"$z={ANCHOR_Z[k]:g}$")
    ax.set(xlabel=r"$R$ [kpc]", ylabel=r"$\Sigma_\star\ [M_\odot\,"
           r"\mathrm{kpc}^{-2}]$", ylim=(1e2, 3e10),
           title="the model profile through cosmic time")
    ax.legend()
    # z=0.4 decomposition by deposit birth era
    w = efficiency_weights(mah["z"], th[3], th[4])
    dM = w * mah["dMh"]
    dM = dM / dM.sum()
    B = deposit_basis(th, mah["t"], ANCHOR_T[0], R_FINE)
    total = scale * (B @ dM)
    sig_t, mid = sigma_of_cog(total, R_FINE)
    bx.loglog(mid, sig_t, color="0.15", lw=2.2, label=r"total ($z=0.4$)")
    for zlo, zhi, label, color in ERAS:
        sel = (mah["z"] >= zlo) & (mah["z"] < zhi)
        if not sel.any():
            continue
        part = scale * (B @ (dM * sel))
        sig_p, _ = sigma_of_cog(part, R_FINE)
        bx.loglog(mid, sig_p, color=color, lw=1.5, ls="--",
                  label=label + rf" ({100 * dM[sel].sum():.0f}"
                  r"$\%$ of budget)")
    bx.set(xlabel=r"$R$ [kpc]", ylabel=r"$\Sigma_\star\ [M_\odot\,"
           r"\mathrm{kpc}^{-2}]$", ylim=(1e2, 3e10),
           title="who built which radius (deposit birth era)")
    bx.legend(fontsize=8)
    fig.tight_layout()
    print("wrote", save_fig(fig, FIGDIR / "note2_assembly_epochs")[0])
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 5. the assembly animation                                                   #
# --------------------------------------------------------------------------- #
def fig_assembly_gif():
    th = theta_z15()
    mah = toy_mah()
    w = efficiency_weights(mah["z"], th[3], th[4])
    dM = w * mah["dMh"]
    dM = dM / dM.sum()
    scale = MSTAR_NORM / kernel_cogs(th, mah, [ANCHOR_T[0]], R_FINE)[0, -1]
    t_frames = np.geomspace(0.8, T_OBS, 56)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 4.3))
    ax.plot(mah["t"], mah["logmh"][1:], color="0.2", lw=1.6)
    ax2 = ax.twinx()
    bars = ax2.bar(mah["t"], np.zeros_like(mah["t"]),
                   width=np.diff(mah["t_edge"]), color="#D55E00", alpha=0.8)
    ax2.set_ylim(0, 1.15 * dM.max())
    ax2.set_ylabel("deposit budget per step", color="#D55E00")
    ax2.tick_params(axis="y", labelcolor="#D55E00")
    cursor = ax.axvline(t_frames[0], color="#0072B2", lw=1.4)
    title = ax.set_title("")
    ax.set(xlabel=r"cosmic time $t$ [Gyr]",
           ylabel=r"$\log_{10} M_h\ [M_\odot]$", xlim=(0.0, T_OBS * 1.03))

    line_tot, = bx.loglog([], [], color="0.15", lw=2.2, label="total")
    era_lines = []
    for _, _, label, color in ERAS:
        ln, = bx.loglog([], [], color=color, lw=1.3, ls="--", label=label)
        era_lines.append(ln)
    bx.set(xlabel=r"$R$ [kpc]",
           ylabel=r"$\Sigma_\star\ [M_\odot\,\mathrm{kpc}^{-2}]$",
           xlim=(1.0, 500.0), ylim=(1e2, 1e11),
           title="deposits arrive compact, then migrate outward")
    bx.legend(fontsize=7, loc="upper right")
    fig.tight_layout()

    def update(frame):
        tk = t_frames[frame]
        arrived = mah["t"] <= tk
        for bar, h, on in zip(bars, dM, arrived):
            bar.set_height(h if on else 0.0)
        cursor.set_xdata([tk, tk])
        title.set_text(rf"$t={tk:.1f}$ Gyr,  $z={float(z_of_t(tk)):.2f}$")
        B = deposit_basis(th, mah["t"], tk, R_FINE)
        tot = scale * (B @ (dM * arrived))
        sig_t, mid = sigma_of_cog(tot, R_FINE)
        good = sig_t > 0
        line_tot.set_data(mid[good], sig_t[good])
        for ln, (zlo, zhi, _, _) in zip(era_lines, ERAS):
            sel = arrived & (mah["z"] >= zlo) & (mah["z"] < zhi)
            if sel.any():
                sig_p, _ = sigma_of_cog(scale * (B @ (dM * sel)), R_FINE)
                gp = sig_p > 0
                ln.set_data(mid[gp], sig_p[gp])
            else:
                ln.set_data([], [])
        return [line_tot, cursor, title, *era_lines, *bars]

    anim = FuncAnimation(fig, update, frames=len(t_frames), blit=False)
    out = FIGDIR / "note2_assembly.gif"
    anim.save(out, writer=PillowWriter(fps=7), dpi=100)
    print("wrote", out)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 6. the stochastic layer                                                     #
# --------------------------------------------------------------------------- #
def fig_stochastic_layer():
    th = theta_z15()
    d = np.load(STAGE1_DIST_NPZ)
    d_sig = np.asarray(d["d_sig_2d"], float)
    pool = d_sig - d_sig.mean()                      # the mean-centered pool
    rng = np.random.default_rng(41)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.4, 4.1))
    bins = np.linspace(-0.6, 0.9, 61)
    ax.hist(pool, bins=bins, density=True, color="#0072B2", alpha=0.75,
            label="measured pool (resampled)")
    from scipy.stats import median_abs_deviation
    s_rob = 1.4826 * median_abs_deviation(pool)
    xg = np.linspace(bins[0], bins[-1], 400)
    ax.plot(xg, np.exp(-xg ** 2 / (2 * s_rob ** 2))
            / np.sqrt(2 * np.pi * s_rob ** 2), color="#D55E00", lw=1.8,
            label=rf"Gaussian, robust $\sigma={s_rob:.3f}$")
    ax.set_yscale("log")
    ax.set(xlabel=r"$\delta_\sigma$ (per-galaxy window-width deviation)",
           ylabel="probability density", ylim=(3e-3, 30),
           title="the diversity axis is heavy-tailed")
    ax.legend(fontsize=8)

    mah = toy_mah()
    draws = pool[rng.integers(0, len(pool), 30)]
    for dv in draws:
        th_d = th.copy()
        th_d[4] = np.clip(th[4] + dv, *SIG_BOX)
        cog = kernel_cogs(th_d, mah, [ANCHOR_T[0]], R_FINE,
                          mstar_norm=MSTAR_NORM)[0]
        sig, mid = sigma_of_cog(cog, R_FINE)
        bx.loglog(mid, sig, color="#0072B2", lw=0.7, alpha=0.35)
    cog0 = kernel_cogs(th, mah, [ANCHOR_T[0]], R_FINE,
                       mstar_norm=MSTAR_NORM)[0]
    sig0, mid = sigma_of_cog(cog0, R_FINE)
    bx.loglog(mid, sig0, color="0.1", lw=2.2, label="mean model")
    bx.loglog([], [], color="#0072B2", lw=0.9, alpha=0.6,
              label=r"30 draws of $\delta_\sigma$ (same halo)")
    bx.set(xlabel=r"$R$ [kpc]", ylabel=r"$\Sigma_\star\ [M_\odot\,"
           r"\mathrm{kpc}^{-2}]$", ylim=(1e2, 3e10),
           title="one halo, a population of galaxies")
    bx.legend(fontsize=8)
    fig.tight_layout()
    print("wrote", save_fig(fig, FIGDIR / "note2_stochastic_layer")[0])
    plt.close(fig)


if __name__ == "__main__":
    set_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig_deposit_shapes()
    fig_efficiency_window()
    fig_migration_clock()
    fig_assembly_epochs()
    fig_stochastic_layer()
    fig_assembly_gif()
