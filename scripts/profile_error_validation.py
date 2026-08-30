"""Is the profile error model good enough to fit with? Three tests.

Nothing in this programme has ever fitted with an error bar, so before the C17
likelihood is built the error model has to be shown to be the right SIZE. These
are the three checks the user asked for.

  A. THE CURVE-OF-GROWTH ERROR MODEL. Fit a flexible five-parameter profile to
     each measured curve of growth by generalised least squares under each
     candidate covariance, and look at the reduced chi-square. A well-sized
     error model on a family flexible enough to describe the data gives values
     near 1. Much larger means the errors are too small; much smaller means they
     are too big. Four nested covariances are compared, so the answer says WHICH
     term is missing, not just that something is.

  B. THE DENSITY PROFILE AS AN INDEPENDENT CHECK. The stored curve of growth and
     the curve rebuilt from the 1-D density profile are two routes to the same
     quantity. Their difference is a real, per-galaxy, per-radius discrepancy,
     so dividing it by the predicted error gives a pull distribution: if the
     error model is right-sized the pull has width about 1.

  C. THE PROJECTION EFFECT AT z = 0.4. The only epoch with the same galaxies in
     three sky projections. How big is it, how does it correlate between radii,
     what does it depend on, and -- the question that matters for modelling --
     what fraction of the galaxy-to-galaxy scatter the model is asked to explain
     is actually just orientation?

Run (after --build and --build-v2):

    HONGSHAO_DATA_DIR=... uv run python scripts/profile_error_validation.py
    ... --n 400          fewer galaxies (default 800) for a quick pass
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "exp54_unpinned_amplitude"))

from hongshao import plotting                                       # noqa: E402
from hongshao.profile_data import (APER6_SMA_KPC, COG_RAD_KPC,      # noqa: E402
                                   EPOCHS, N_EPOCH, REDSHIFTS,
                                   SHARED_GRID, load_profiles)
from hongshao.profiles import _integrate                            # noqa: E402
from hongshao.qa import _zcolors                                    # noqa: E402

LN10 = np.log(10.0)
FIGDIR = ROOT / "figures"
N_PARAM = 5                       # the profile family's degrees of freedom
DOF = len(COG_RAD_KPC) - N_PARAM  # 19


# =============================================================================
# the covariance candidates, in LOG10 space
# =============================================================================
def cov_log(sigma_iso_abs, mass, sigma_shape_dex=None, proj_cov_dex=None,
            full=True):
    """Covariance of ``log10 M*(<R)`` under a chosen set of error terms.

    ``full`` selects the exact cumulative structure ``Cov(M_i,M_j) =
    Var(M_min(i,j))`` rather than a diagonal. ``sigma_shape_dex`` enters as a
    rank-one all-ones block because one ellipticity multiplies the whole curve
    coherently. ``proj_cov_dex`` is a measured 24x24 log-space block.
    """
    v = np.asarray(sigma_iso_abs, float) ** 2
    C = np.minimum(v[:, None], v[None, :]) if full else np.diag(v)
    m = np.asarray(mass, float)
    C = C / np.outer(m, m) / LN10 ** 2
    if sigma_shape_dex is not None and np.isfinite(sigma_shape_dex):
        C = C + sigma_shape_dex ** 2                 # coherent in radius
    if proj_cov_dex is not None:
        C = C + proj_cov_dex
    return C


# =============================================================================
# A. generalised-least-squares fit of the profile family
# =============================================================================
def _gls_reduced_chi2(logR, y, C):
    """Fit the five-parameter radial-DiffMAH curve of growth to ``y`` under
    covariance ``C`` and return the reduced chi-square.

    The residuals are whitened by the Cholesky factor of ``C``, so this is a
    genuine GLS fit and not a diagonal fit with a covariance quoted afterwards.
    """
    try:
        L = np.linalg.cholesky(C + 1e-12 * np.eye(len(C)) * np.trace(C) / len(C))
    except np.linalg.LinAlgError:
        return np.nan
    u = logR
    yl = y * LN10
    bo0 = float(np.clip((yl[-1] - yl[-4]) / (u[-1] - u[-4]), 0.02, 2.9))
    bi0 = float(np.clip((yl[3] - yl[0]) / (u[3] - u[0]), bo0 + 0.05, bo0 + 4.9))
    x0 = [yl[0], bo0, bi0 - bo0, float(np.median(u)), 0.5]
    lo = [yl[0] - 5, 0.0, 0.0, u[0], 0.05]
    hi = [yl[0] + 5, 3.0, 5.0, u[-1], 2.0]
    x0 = list(np.clip(x0, lo, hi))

    def resid(p):
        lnM0, b_out, db, u_c, Delta = p
        beta = b_out + db * expit((u_c - u) / Delta)
        r = (lnM0 + _integrate(beta, u) - yl) / LN10
        return np.linalg.solve(L, r)

    try:
        sol = least_squares(resid, x0, bounds=(lo, hi), method="trf",
                            max_nfev=600)
    except Exception:
        return np.nan
    return float(np.sum(sol.fun ** 2) / DOF)


def test_a(d, rows, proj_blocks, verbose=True):
    """Reduced chi-square of the profile fit under four nested covariances."""
    logR = np.log(COG_RAD_KPC)
    names = ["iso only, diagonal", "iso only, full covariance",
             "iso + shape", "iso + shape + projection"]
    out = {n: np.full((len(rows), N_EPOCH), np.nan) for n in names}
    t0 = time.time()
    for a, i in enumerate(rows):
        for k in range(N_EPOCH):
            M = d["cog_provided"][i, k]
            s = d["sigma_iso_abs"][i, k]
            if not (np.isfinite(M).all() and (M > 0).all() and np.isfinite(s).all()):
                continue
            y = np.log10(M)
            ssh = d["sigma_shape_dex"][i, k]
            variants = [
                cov_log(s, M, full=False),
                cov_log(s, M, full=True),
                cov_log(s, M, ssh, full=True),
                cov_log(s, M, ssh, proj_blocks[i] if k == 0 else None, full=True),
            ]
            for n, C in zip(names, variants):
                if n == names[3] and k != 0:
                    continue                      # projection is z=0.4 only
                out[n][a, k] = _gls_reduced_chi2(logR, y, C)
        if verbose and (a + 1) % 100 == 0:
            print(f"    A: {a + 1}/{len(rows)}  ({time.time() - t0:.0f} s)",
                  flush=True)
    return out, names


# =============================================================================
# C. the projection effect, measured
# =============================================================================
def projection_blocks(d, corr):
    """Per-galaxy 24x24 log-space projection covariance at z=0.4.

    The per-radius amplitude is the galaxy's own three-projection scatter,
    interpolated in log radius onto the CoG grid; the radius-to-radius
    correlation is the population matrix measured in ``projection_stats``. A
    galaxy seen edge-on is over-dense at EVERY radius, so this block is far from
    diagonal and using only its diagonal would understate the term badly.
    """
    n = len(COG_RAD_KPC)
    sp = d["sigma_proj_dex"]
    blocks = []
    lg6, lg24 = np.log(APER6_SMA_KPC), np.log(COG_RAD_KPC)
    for i in range(sp.shape[0]):
        v = sp[i]
        if not np.isfinite(v).all():
            blocks.append(None)
            continue
        s = np.interp(lg24, lg6, v)
        blocks.append(corr * np.outer(s, s))
    return blocks


def projection_stats(d):
    """Everything measurable about the projection term at z=0.4."""
    ap = d["aper_mass_proj"]                       # (n, 3, 6)
    with np.errstate(invalid="ignore", divide="ignore"):
        lap = np.log10(np.where(ap > 0, ap, np.nan))
    res = lap - np.nanmean(lap, axis=1, keepdims=True)   # per-projection residual
    flat = res.reshape(-1, len(APER6_SMA_KPC))
    ok = np.isfinite(flat).all(axis=1)
    corr6 = np.corrcoef(flat[ok], rowvar=False)
    # population correlation, interpolated onto the 24-radius grid
    lg6, lg24 = np.log(APER6_SMA_KPC), np.log(COG_RAD_KPC)
    tmp = np.array([np.interp(lg24, lg6, corr6[j]) for j in range(len(lg6))])
    corr24 = np.array([np.interp(lg24, lg6, tmp[:, j]) for j in range(len(lg24))]).T
    corr24 = 0.5 * (corr24 + corr24.T)
    np.fill_diagonal(corr24, 1.0)
    w, V = np.linalg.eigh(corr24)                 # nearest positive-definite
    corr24 = V @ np.diag(np.clip(w, 1e-6, None)) @ V.T
    dg = np.sqrt(np.diag(corr24))
    corr24 = corr24 / np.outer(dg, dg)
    return corr6, corr24, np.nanstd(lap, axis=1)


def population_scatter_at_fixed_mass(d, logmh, mask, n_bins=12):
    """Galaxy-to-galaxy scatter of log10 M*(<R) at fixed halo mass, z=0.4.

    This is the quantity a model is asked to explain. Comparing the projection
    term against it says what fraction of the target is orientation rather than
    physics.
    """
    ap = d["aper_mass_proj"][:, 0]                # the xy projection = the data
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.log10(np.where(ap > 0, ap, np.nan))
    ok = mask & np.isfinite(logmh)
    edges = np.quantile(logmh[ok], np.linspace(0, 1, n_bins + 1))
    sd = np.full(len(APER6_SMA_KPC), np.nan)
    for j in range(len(APER6_SMA_KPC)):
        acc = []
        for b in range(n_bins):
            m = ok & (logmh >= edges[b]) & (logmh < edges[b + 1]) & np.isfinite(y[:, j])
            if m.sum() > 30:
                acc.append(y[m, j] - np.median(y[m, j]))
        if acc:
            sd[j] = np.std(np.concatenate(acc))
    return sd


# =============================================================================
# figure
# =============================================================================
def make_figure(chi2, names, corr24, sp_med, pop_sd, pulls, out_stem):
    plotting.set_style()
    colors = _zcolors(N_EPOCH)
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.6))
    axA, axB, axC, axD = axes.ravel()

    x = np.arange(N_EPOCH)
    # 'iso only, full covariance' and 'iso + shape' land on top of each other --
    # that IS a result (the coherent shape term is a pure common mode, which the
    # full covariance already makes cheap), so draw the second one dashed and
    # say so rather than letting one hide the other.
    style = [dict(ls="-", marker="o", ms=5), dict(ls="-", marker="o", ms=5),
             dict(ls="--", marker="s", ms=4), dict(ls="none", marker="*", ms=18)]
    for j, n in enumerate(names):
        med = np.array([np.nanmedian(chi2[n][:, k]) for k in range(N_EPOCH)])
        axA.plot(x, med, lw=1.6, label=n,
                 color=plotting.OKABE_ITO[j % len(plotting.OKABE_ITO)],
                 **style[j])
    axA.axhline(1.0, color="0.35", lw=1.2, ls="--")
    axA.set_yscale("log")
    axA.set_xticks(x)
    axA.set_xticklabels([f"$z={z:g}$" for z in REDSHIFTS])
    axA.set_xlim(-0.35, N_EPOCH - 0.4)
    axA.set_ylabel("median reduced $\\chi^2$   (19 dof)")
    axA.set_title("A. is the CoG error model the right size?", loc="left")
    axA.legend(fontsize=7, loc="lower left", ncol=1)
    axA.set_ylim(top=3e1)
    axA.text(0.98, 0.97, "dashed grey = a correctly sized error model\n"
             "'+ shape' sits ON the full-covariance curve:\n"
             "a coherent term the covariance already absorbs",
             transform=axA.transAxes, ha="right", va="top", fontsize=7,
             color="0.35")

    for k in range(N_EPOCH):
        v = pulls[:, k]
        v = v[np.isfinite(v)]
        if v.size < 50:
            continue
        axB.hist(np.clip(v, -6, 6), bins=60, histtype="step", density=True,
                 color=colors[k], lw=1.4, label=f"$z={REDSHIFTS[k]:g}$")
    g = np.linspace(-6, 6, 200)
    axB.plot(g, np.exp(-g ** 2 / 2) / np.sqrt(2 * np.pi), color="0.35",
             lw=1.2, ls="--", label="unit normal")
    axB.set_xlabel("pull: (rebuilt $-$ stored) / predicted $\\sigma$, at 100 kpc")
    axB.set_ylabel("density")
    axB.set_title("B. the density profile as an independent check", loc="left")
    axB.legend(fontsize=7)

    im = axC.imshow(corr24, origin="lower", cmap="cividis", vmin=0.6, vmax=1.0,
                    extent=[0, 24, 0, 24])
    axC.set_title("C. projection error is correlated between radii", loc="left")
    tick = [0, 6, 12, 18, 23]
    axC.set_xticks([t + .5 for t in tick])
    axC.set_yticks([t + .5 for t in tick])
    lab = [f"{COG_RAD_KPC[t]:.0f}" for t in tick]
    axC.set_xticklabels(lab)
    axC.set_yticklabels(lab)
    axC.set_xlabel("radius [kpc]")
    axC.set_ylabel("radius [kpc]")
    fig.colorbar(im, ax=axC, fraction=0.046, label="correlation")

    axD.plot(APER6_SMA_KPC, pop_sd, marker="o", ms=5, lw=2.0, color="#333333",
             label="scatter the model must explain (at fixed $M_h$)")
    axD.plot(APER6_SMA_KPC, sp_med, marker="s", ms=5, lw=2.0, color="#D55E00",
             label="projection scatter (measured, 3 sky views)")
    axD.set_xscale("log")
    axD.set_yscale("log")
    axD.set_ylim(2e-3, 4e-1)
    axD.set_xlabel("aperture semi-major axis [kpc]")
    axD.set_ylabel("$\\sigma$ [dex]")
    axD.set_title("D. how much of the target is orientation? ($z=0.4$)",
                  loc="left")
    for xx, a_, b_ in zip(APER6_SMA_KPC, pop_sd, sp_med):
        axD.annotate(f"{100 * (b_ / a_) ** 2:.1f}", (xx, b_),
                     textcoords="offset points", xytext=(0, -15), ha="center",
                     fontsize=7.5, color="#D55E00")
    axD.legend(fontsize=7, loc="upper right")
    axD.text(0.02, 0.06, "orange labels: projection share of the VARIANCE, per cent\n"
             "the dominant MEASUREMENT error, but only a few per cent\n"
             "of the signal a model is asked to reproduce",
             transform=axD.transAxes, ha="left", va="bottom", fontsize=7,
             color="#D55E00")

    fig.tight_layout()
    paths = plotting.save_fig(fig, out_stem)
    plt.close(fig)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800,
                    help="galaxies used for the chi-square test (default 800)")
    a = ap.parse_args()

    import selection as S
    d = load_profiles()
    pop = np.load(ROOT / "experiments/exp32_full_population/outputs/population.npz",
                  allow_pickle=True)
    idx = pop["index"].astype(int)
    fitm = S.fitting_sample_mask(pop["data"], pop["logmh_zk_diffmah"], verbose=False)
    logmh_z04 = pop["logmh_zk_diffmah"][:, 0]
    in_fit = np.zeros(d["n_iso"].shape[0], bool)
    in_fit[idx[fitm]] = True
    logmh = np.full(d["n_iso"].shape[0], np.nan)
    logmh[idx] = logmh_z04
    print(f"fitting sample: {int(in_fit.sum())} galaxies")

    print("\n=== C. the projection effect at z=0.4 (measured first: A needs it)")
    corr6, corr24, sp = projection_stats(d)
    print("   radius-to-radius CORRELATION of the projection residual "
          "(6 measured radii)")
    print("        " + " ".join(f"{int(r):>7}" for r in APER6_SMA_KPC))
    for j, r in enumerate(APER6_SMA_KPC):
        print(f"   {int(r):>4} " + " ".join(f"{corr6[j, q]:7.3f}"
                                            for q in range(len(APER6_SMA_KPC))))
    print("   -> the projection error is strongly correlated across radii: a")
    print("      galaxy seen edge-on is over-dense at EVERY radius, so treating")
    print("      this term as diagonal would badly understate it.")
    sp_med = np.nanmedian(np.where(in_fit[:, None], sp, np.nan), axis=0)
    pop_sd = population_scatter_at_fixed_mass(d, logmh, in_fit)
    print("\n   projection vs the scatter a model must explain, z=0.4:")
    print(f"   {'radius [kpc]':>14} " + " ".join(f"{int(r):>8}" for r in APER6_SMA_KPC))
    print(f"   {'projection':>14} " + " ".join(f"{v:8.4f}" for v in sp_med))
    print(f"   {'at fixed Mh':>14} " + " ".join(f"{v:8.4f}" for v in pop_sd))
    print(f"   {'share of var':>14} " + " ".join(
        f"{100 * (b / a) ** 2:7.1f}%" for a, b in zip(pop_sd, sp_med)))

    print("\n=== B. the density profile as an independent check")
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = np.log10(d["cog_from_density"] / d["cog_provided"])
    sig_tot = np.sqrt(d["sigma_iso_dex"] ** 2 + d["sigma_shape_dex"][:, :, None] ** 2)
    pulls_all = np.where(in_fit[:, None, None], diff / sig_tot, np.nan)
    j100 = int(np.argmin(np.abs(COG_RAD_KPC - 100)))
    print("   pull = (CoG rebuilt from the density - stored CoG) / predicted sigma")
    print(f"   {'radius':>9} " + " ".join(f"{e:>9}" for e in EPOCHS))
    for j in (0, 6, 15, j100, 23):
        print(f"   {COG_RAD_KPC[j]:9.1f} " + " ".join(
            f"{np.nanstd(pulls_all[:, k, j]):9.2f}" for k in range(N_EPOCH)))
    print("   (values are the WIDTH of the pull; 1 = a correctly sized error model)")

    print(f"\n=== A. the CoG error model, {a.n} galaxies x 5 epochs x 4 covariances")
    blocks = projection_blocks(d, corr24)
    rng = np.random.default_rng(0)
    rows = rng.choice(np.where(in_fit)[0], min(a.n, int(in_fit.sum())),
                      replace=False)
    chi2, names = test_a(d, rows, blocks)
    print(f"\n   median reduced chi-square ({DOF} dof); 1 = correctly sized")
    print(f"   {'covariance':<28}" + "".join(f"{e:>10}" for e in EPOCHS))
    for n in names:
        print(f"   {n:<28}" + "".join(
            f"{np.nanmedian(chi2[n][:, k]):10.1f}" if np.isfinite(
                np.nanmedian(chi2[n][:, k])) else f"{'-':>10}"
            for k in range(N_EPOCH)))

    paths = make_figure(chi2, names, corr24, sp_med, pop_sd,
                        pulls_all[:, :, j100], FIGDIR / "profile_error_validation")
    for p in paths:
        print("\nwrote", p)


if __name__ == "__main__":
    main()
