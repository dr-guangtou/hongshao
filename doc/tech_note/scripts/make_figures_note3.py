"""Pedagogical figures for tech note 3 — the two-channel alternative.

Run: PYTHONPATH=. uv run python doc/tech_note/scripts/make_figures_note3.py

Figures (doc/tech_note/figures/):
  note3_two_channel_anatomy  the compact + wide channel decomposition and the
                             mass-dependent split f_ex(M_h)
  note3_one_vs_two           why one heavy-tailed deposit supersedes the
                             split (deposit level and full-model level)

The 2ch-exp theta is the fitted exp38 stage-2 value (joint 5-epoch); the
1ch-mof comparison uses the same 5-epoch theta so both kernels are shown at
the SAME fit scope (the adopted product later moved to the z<=1.5 scope).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

from common import (ANCHOR_T, FIGDIR, ROOT, T_OBS, deposit_basis,
                    efficiency_weights, moffat_cog, save_fig, set_style,
                    sigma_of_cog, toy_mah)

R_FINE = np.geomspace(1.0, 500.0, 220)
MSTAR_NORM = 10.0 ** 11.3
E38_NPZ = (ROOT / "experiments/exp38_deposit_rethink/outputs/"
           "stage2_multiepoch.npz")
MH_SCALE_NPZ = (ROOT / "experiments/exp32_full_population/outputs/"
                "um_slope_diffmah.npz")


def _thetas():
    d = np.load(E38_NPZ)
    return d["theta_2ch-exp"], d["theta_1ch-mof"]


def channel_basis(shape, log_s0, g, q, ti, tk, t_obs=T_OBS):
    """Migrating-deposit basis for a Gaussian or exponential channel —
    the exp35 ``basis_ext`` / exp38 ``basis_exp`` math with the shared
    alpha=1 clock."""
    ti = np.asarray(ti, float)
    s0 = np.clip(10.0 ** log_s0 * (ti / t_obs) ** g, 1e-4, 1e5)
    fc = np.exp(-np.clip(tk - ti, 0.0, None) / ti)
    sw = np.clip(s0 * (tk / ti) ** max(q, 0.0), 1e-4, 1e5)

    def cog(s):
        x = R_FINE[:, None] / s[None, :]
        if shape == "gauss":
            return 1.0 - np.exp(-x ** 2 / 2.0)
        return 1.0 - (1.0 + x) * np.exp(-x)               # exponential (n=1)
    return fc[None, :] * cog(s0) + (1.0 - fc)[None, :] * cog(sw)


def two_channel_parts(th, mah, tk, f_ex=None):
    """(compact, wide, total) linear CoGs of the 2ch-exp model at ``tk``
    (population theta, conditioning at zero)."""
    log_s0, g, q, mu, sig = th[:5]
    ls_ex, fa = th[11], th[12]
    if f_ex is None:
        f_ex = float(expit(fa))
    w = efficiency_weights(mah["z"], mu, sig)
    dM = w * mah["dMh"]
    dM = dM / dM.sum()
    dM = dM * (mah["t"] <= tk)
    m_in = (1.0 - f_ex) * (channel_basis("gauss", log_s0, g, q, mah["t"], tk)
                           @ dM)
    m_ex = f_ex * (channel_basis("expo", ls_ex, g, q, mah["t"], tk) @ dM)
    return m_in, m_ex, m_in + m_ex


def fig_two_channel_anatomy():
    th2, _ = _thetas()
    mah = toy_mah()
    m_in, m_ex, m_tot = two_channel_parts(th2, mah, ANCHOR_T[0])
    scale = MSTAR_NORM / m_tot[-1]
    f_ex0 = float(expit(th2[12]))

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 4.1))
    for m, label, color, ls, lw in (
            (m_tot, "total", "0.15", "-", 2.2),
            (m_in, rf"compact channel ($1-f_{{\rm ex}}={1 - f_ex0:.2f}$, "
             "Gaussian)", "#0072B2", "--", 1.7),
            (m_ex, rf"wide channel ($f_{{\rm ex}}={f_ex0:.2f}$, "
             "exponential)", "#D55E00", "-.", 1.7)):
        sig, mid = sigma_of_cog(scale * m, R_FINE)
        good = sig > 0
        ax.loglog(mid[good], sig[good], ls, color=color, lw=lw, label=label)
    ax.set(xlabel=r"$R$ [kpc]",
           ylabel=r"$\Sigma_\star\ [M_\odot\,\mathrm{kpc}^{-2}]$",
           ylim=(1e2, 3e10),
           title=r"the two-channel deposit at $z=0.4$ (2ch-exp theta)")
    ax.legend(fontsize=8)

    mh_mean, mh_sd = np.load(MH_SCALE_NPZ)["mh_scale"]
    logmh = np.linspace(13.0, 15.0, 200)
    fb_mh = th2[13]
    f_ex = expit(th2[12] + fb_mh * (logmh - mh_mean) / mh_sd)
    bx.plot(logmh, f_ex, color="#D55E00", lw=2.0)
    bx.set(xlabel=r"$\log_{10} M_h\ [M_\odot]$",
           ylabel=r"wide-channel deposit share $f_{\rm ex}$",
           ylim=(0.0, 1.0),
           title="the split is conditioned on the halo")
    bx.text(0.05, 0.9,
            r"$f_{\rm ex} = \mathrm{expit}(f_a + f_b\,\hat m_h + \dots)$",
            transform=bx.transAxes, fontsize=10)
    bx.text(0.05, 0.8, "more massive halos deposit\na larger wide-channel "
            "share", transform=bx.transAxes, fontsize=8, color="0.35")
    fig.tight_layout()
    print("wrote", save_fig(fig, FIGDIR / "note3_two_channel_anatomy")[0])
    plt.close(fig)


def fig_one_vs_two():
    th2, th1 = _thetas()
    gam = float(th1[5])

    # (a) deposit level: a Gaussian + exponential mixture vs one Moffat,
    # matched half-mass radius
    r50 = 10.0
    s_g = r50 / np.sqrt(2.0 * np.log(2.0))
    x50_m = np.sqrt(0.5 ** (1.0 / (1.0 - gam)) - 1.0)
    f_mix = 0.45
    a_ex = 5.0 * s_g                                  # the wide role: ~5x scale
    r = np.geomspace(0.5, 400.0, 400)

    def mix_cog(rr):
        return ((1.0 - f_mix) * (1.0 - np.exp(-rr ** 2 / (2.0 * s_g ** 2)))
                + f_mix * (1.0 - (1.0 + rr / a_ex) * np.exp(-rr / a_ex)))

    r50_mix = np.interp(0.5, mix_cog(r), r)
    rc_m = r50_mix / x50_m                            # Moffat matched to mix
    s_g1 = r50_mix / np.sqrt(2.0 * np.log(2.0))       # single Gaussian, same R50

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 4.1))
    for label, one_minus_cog, color, ls in (
            ("single Gaussian", np.exp(-r ** 2 / (2.0 * s_g1 ** 2)),
             "#0072B2", "--"),
            (rf"Gaussian $+$ exponential split ($f_{{\rm ex}}={f_mix}$)",
             1.0 - mix_cog(r), "#009E73", "-."),
            (rf"one Moffat deposit ($\gamma={gam:.2f}$)",
             1.0 - moffat_cog(r, np.array([rc_m]), gam)[:, 0],
             "#D55E00", "-")):
        ax.loglog(r, one_minus_cog, ls, color=color, lw=1.8, label=label)
    ax.set(xlabel=r"$R$ [kpc]", ylabel=r"fraction of deposit mass beyond $R$",
           ylim=(1e-4, 1.2),
           title="matched half-mass radius: the tail the split was buying")
    ax.legend(fontsize=8, loc="lower left")

    # (b) full-model level: both fitted kernels on the same halo at z=0.4
    mah = toy_mah()
    _, _, m2 = two_channel_parts(th2, mah, ANCHOR_T[0])
    w = efficiency_weights(mah["z"], th1[3], th1[4])
    dM = w * mah["dMh"]
    dM = dM / dM.sum()
    m1 = deposit_basis(th1[:6], mah["t"], ANCHOR_T[0], R_FINE) @ dM
    for m, label, color, ls in (
            (m2, "2ch-exp (16 parameters)", "#009E73", "-."),
            (m1, "1ch-mof (12 parameters, adopted)", "#D55E00", "-")):
        sig, mid = sigma_of_cog(m * MSTAR_NORM / m[-1], R_FINE)
        good = sig > 0
        bx.loglog(mid[good], sig[good], ls, color=color, lw=1.9, label=label)
    bx.set(xlabel=r"$R$ [kpc]",
           ylabel=r"$\Sigma_\star\ [M_\odot\,\mathrm{kpc}^{-2}]$",
           ylim=(1e2, 3e10),
           title="two roads to the same outskirts (same halo, $z=0.4$)")
    bx.legend(fontsize=8)
    fig.tight_layout()
    print("wrote", save_fig(fig, FIGDIR / "note3_one_vs_two")[0])
    plt.close(fig)


if __name__ == "__main__":
    set_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig_two_channel_anatomy()
    fig_one_vs_two()
