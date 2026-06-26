"""exp26 — differential stellar-mass surface-density profiles between adjacent
redshift snapshots, and what functional form describes them.

The exp25 deposition kernel assumes each epoch's *added* stellar mass has a
centred 2-D Gaussian surface density. The TNG300 drop stores measured profiles at
5 redshifts (z=0.4/0.7/1.0/1.5/2.0 = snaps 72/59/50/40/33), so we can test that
assumption directly: build the differential profile

    dSigma(R) = Sigma(R, z_low) - Sigma(R, z_high)

for each adjacent pair [0.4,0.7], [0.7,1.0], [1.0,1.5], [1.5,2.0] (z_low is the
later/lower-z epoch), and ask whether dSigma(R) follows a simple form. We fit a
Sersic profile to each dSigma: its index n is the diagnostic, because

    n = 0.5  <=>  Gaussian  (the deposition-kernel assumption)
    n = 1.0  <=>  exponential
    n = 4.0  <=>  de Vaucouleurs.

Caveats: (1) data stop at z=2 (massive galaxies are compact ~exponential there);
(2) only 5 epochs, so the time sampling is coarse; (3) `intensity` is the isophote
surface density [Msun/kpc^2] (integrating 2 pi R dR overshoots the CoG by the
ellipticity factor 1/(1-e) ~ 1.3-1.5 — shapes are reliable, absolute mass is not);
(4) differencing two measured profiles amplifies noise, worst for the high-z,
small-Delta-t pairs.

Run: PYTHONPATH=. uv run python experiments/exp26_differential_profiles/differential_profiles.py [N] [--refit]
"""
# %% setup
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hongshao.tng_data import load_prof                                     # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO               # noqa: E402

set_style()
HERE = Path(__file__).resolve().parent
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"

# the 5 measured epochs and the 4 adjacent differential pairs. Each pair is
# (lo, hi) = (later/lower-z index, earlier/higher-z index); dSigma = Sigma_lo -
# Sigma_hi = later - earlier = mass GROWN over that interval (positive if growing).
ZKEYS = ["map_hist_z0p4", "map_hist_z0p7", "map_hist_z1", "map_hist_z1p5", "map_hist_z2"]
ZVALS = [0.4, 0.7, 1.0, 1.5, 2.0]
PAIRS = [(0, 1, "[0.4,0.7]"), (1, 2, "[0.7,1.0]"), (2, 3, "[1.0,1.5]"), (3, 4, "[1.5,2.0]")]
RGRID = np.geomspace(2.0, 150.0, 50)                  # storage grid [kpc]
FIT_LO, FIT_HI = 6.0, 100.0                           # resolved range (skip marginally-resolved core)
N_SERSIC = np.r_[np.linspace(0.3, 1.0, 15), np.linspace(1.05, 6.0, 25)]   # Sersic-index search grid
R_IN, R_OUT = 8.0, 60.0                               # inner / outer reference radii [kpc]


# %% ---- profile -> common grid ----------------------------------------------
def sigma_on_grid(prof_z):
    """Interpolate one epoch's isophote Sigma(R) onto RGRID (log-log); NaN outside
    the measured radial range. Returns (Sigma_grid, r_min, r_max)."""
    p = prof_z["prof"]
    r = np.asarray(p["r_kpc"], float)
    intens = np.asarray(p["intensity"], float)
    m = np.isfinite(r) & np.isfinite(intens) & (r > 0) & (intens > 0)
    if m.sum() < 5:
        return np.full(RGRID.shape, np.nan), np.nan, np.nan
    r, intens = r[m], intens[m]
    o = np.argsort(r)
    r, intens = r[o], intens[o]
    out = np.full(RGRID.shape, np.nan)
    inside = (RGRID >= r.min()) & (RGRID <= r.max())
    out[inside] = 10.0 ** np.interp(np.log10(RGRID[inside]), np.log10(r), np.log10(intens))
    return out, float(r.min()), float(r.max())


def build(sample):
    """Per galaxy: Sigma(R) at the 5 epochs and dSigma for the 4 adjacent pairs."""
    n = len(sample)
    sigma = np.full((n, 5, len(RGRID)), np.nan)
    dsigma = np.full((n, 4, len(RGRID)), np.nan)
    flags = np.zeros((n, 5), bool)
    for i, idx in enumerate(sample["index"]):
        pr = load_prof(int(idx))
        for k, zk in enumerate(ZKEYS):
            if zk in pr:
                flags[i, k] = bool(pr[zk].get("flag"))
                sigma[i, k] = sigma_on_grid(pr[zk])[0]
        for j, (lo, hi, _) in enumerate(PAIRS):
            dsigma[i, j] = sigma[i, lo] - sigma[i, hi]
    return dict(sigma=sigma, dsigma=dsigma, flags=flags,
                index=np.asarray(sample["index"]),
                logmh=np.asarray(sample["logmh_z0p4"], float),
                logmstar=np.asarray(sample["logmstar_cog"], float)[:, -1])


# %% ---- characterise the differential growth --------------------------------
# Two views of the same pair: (1) the LINEAR dSigma = Sigma_low - Sigma_high (what
# a deposition kernel adds); (2) the FRACTIONAL Dlog Sigma = log Sigma_low -
# log Sigma_high (robust to the steep, marginally-resolved core that dominates the
# linear difference). The fractional growth is the clean diagnostic: if it rises
# with radius, growth is inside-out and the profile flattens/extends -- i.e. the
# multiplicative law Sigma_low/Sigma_high ~ R^b, NOT a centred Gaussian deposit.
def best_sersic_n(R, y):
    """Best Sersic n for a positive declining profile y(R): n=0.5 Gaussian, 1 exp."""
    yl = np.log10(y)
    best = (np.nan, 1e9)
    for n in N_SERSIC:
        x = R ** (1.0 / n)
        c = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, yl, rcond=None)[0]
        if c[0] >= 0:
            continue
        rms = float(np.sqrt(np.mean((c[0] * x + c[1] - yl) ** 2)))
        if rms < best[1]:
            best = (float(n), rms)
    return best


def describe(sig_lo, sig_hi):
    """Fractional + linear descriptors of one differential profile over [FIT_LO,FIT_HI]."""
    fit = (RGRID >= FIT_LO) & (RGRID <= FIT_HI)
    R, slo, shi = RGRID[fit], sig_lo[fit], sig_hi[fit]
    ok = np.isfinite(slo) & np.isfinite(shi) & (slo > 0) & (shi > 0)
    if ok.sum() < 6:
        return dict(dlog_b=np.nan, dlog_in=np.nan, dlog_out=np.nan, R_peak=np.nan,
                    n_sersic=np.nan, n_rms=np.nan, klass="unusable")
    R, dlog, dS = R[ok], np.log10(slo[ok]) - np.log10(shi[ok]), slo[ok] - shi[ok]
    b = float(np.polyfit(np.log10(R), dlog, 1)[0])                  # Dlog Sigma ~ b*logR
    dlog_in = float(np.interp(R_IN, R, dlog))
    dlog_out = float(np.interp(R_OUT, R, dlog))
    R_peak = float(R[np.argmax(dS)]) if (dS > 0).any() else np.nan  # radius where added mass peaks
    pos = dS > 0
    n_sersic, n_rms = best_sersic_n(R[pos], dS[pos]) if pos.sum() >= 8 else (np.nan, np.nan)
    if dlog_in < -0.02:
        klass = "core_drop"        # central density fell (massive BCG dry-merger / expansion)
    elif b > 0.15:
        klass = "inside_out"       # outer grows faster -> profile flattens/extends
    elif b < -0.15:
        klass = "concentrating"    # inner grows faster
    else:
        klass = "self_similar"     # ~uniform fractional growth
    return dict(dlog_b=b, dlog_in=dlog_in, dlog_out=dlog_out, R_peak=R_peak,
                n_sersic=n_sersic, n_rms=n_rms, klass=klass)


def stacked_slope(sigma, lo, hi):
    """Fit Dlog Sigma ~ b*logR to the per-radius MEDIAN profile (robust to per-galaxy
    noise). Returns (b, dlogS at R_IN, dlogS at R_OUT) of the median profile."""
    with np.errstate(divide="ignore", invalid="ignore"):
        med = np.nanmedian(np.log10(sigma[:, lo, :]) - np.log10(sigma[:, hi, :]), axis=0)
    good = (RGRID >= FIT_LO) & (RGRID <= FIT_HI) & np.isfinite(med)
    b = float(np.polyfit(np.log10(RGRID[good]), med[good], 1)[0]) if good.sum() >= 4 else np.nan
    return b, float(np.interp(R_IN, RGRID, med)), float(np.interp(R_OUT, RGRID, med))


def analyze(data):
    """Characterise every (galaxy, pair) differential profile; flat results table."""
    rows = []
    for i in range(len(data["index"])):
        for j, (lo, hi, lab) in enumerate(PAIRS):
            d = describe(data["sigma"][i, lo], data["sigma"][i, hi])
            rows.append(dict(index=int(data["index"][i]), pair=lab, pair_idx=j,
                             logmh=data["logmh"][i], logmstar=data["logmstar"][i],
                             flag_ok=bool(data["flags"][i, lo] and data["flags"][i, hi]), **d))
    return Table(rows)


# %% ---- driver --------------------------------------------------------------
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    refit = "--refit" in sys.argv
    n = int(args[0]) if args else None

    # exp26 only differences MEASURED profiles -> it does NOT need the MAH-quality
    # cuts in `use` (mah_declined / logmh>=13 / latest_snap). Sample = every galaxy
    # with a valid z=0.4 profile (flag) and a finite CoG: the full profile-valid set.
    t = Table.read(TABLE)
    keep = np.asarray(t["flag"]) & np.isfinite(np.asarray(t["logmstar_cog"], float)).all(axis=1)
    t = t[keep]
    if n:
        t = t[:n]
    suffix = f"_n{n}" if n else ""
    npz = OUTDIR / f"differential_profiles{suffix}.npz"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if npz.exists() and not refit:
        d = np.load(npz, allow_pickle=True)
        data = {k: d[k] for k in d.files}
        print(f"loaded cached {npz.name}")
    else:
        print(f"building differential profiles for {len(t)} galaxies...")
        data = build(t)
        np.savez_compressed(npz, R=RGRID, **data)
        print(f"wrote {npz.name}")

    res = analyze(data)
    res.write(OUTDIR / f"differential_fits{suffix}.fits", overwrite=True)

    print(f"\nexp26 differential profiles: {len(data['index'])} galaxies x 4 pairs "
          f"= {len(res)} differential profiles  (resolved range {FIT_LO}-{FIT_HI} kpc)\n")
    print("  Robust population diagnostic: slope b of the median Dlog Sigma vs logR per epoch")
    print("  (Sigma_low/Sigma_high ~ R^b; b>0 = inside-out / profile flattens). Adjacent")
    print("  pairs have small Dt -> noisy per galaxy, so we fit the STACKED median profile.\n")
    print(f"  {'pair':12s} {'b_stack':>9s} {'dlogS(8kpc)':>12s} {'dlogS(60kpc)':>13s} "
          f"{'b(per-gal,noisy)':>18s}")
    for j, (lo, hi, lab) in enumerate(PAIRS):
        b, din, dout = stacked_slope(data["sigma"], lo, hi)
        bg = np.nanmedian(np.asarray(res["dlog_b"])[res["pair_idx"] == j])
        print(f"  {lab:12s} {b:>+9.2f} {din:>+12.3f} {dout:>+13.3f} {bg:>+18.2f}")
    b_long, din_long, dout_long = stacked_slope(data["sigma"], 0, 4)
    with np.errstate(divide="ignore", invalid="ignore"):
        dlog_long_in = (np.log10(data["sigma"][:, 0, :]) - np.log10(data["sigma"][:, 4, :]))
    iin = np.argmin(np.abs(RGRID - R_IN))
    f_coredrop = float(np.nanmean(dlog_long_in[:, iin] < -0.05))
    print(f"\n  long baseline z=2->0.4: b_stack={b_long:+.2f}  inner(8kpc) growth {din_long:+.2f} dex, "
          f"outer(60kpc) {dout_long:+.2f} dex")
    print(f"    -> outer grows ~{10**dout_long:.0f}x vs inner ~{10**din_long:.1f}x; "
          f"only {100*f_coredrop:.0f}% of galaxies show an inner-density DROP.")

    make_figures(data, res, suffix)
    print(f"\nwrote outputs -> {OUTDIR}\nwrote figures -> {FIGDIR}")

    # self-checks
    assert np.isfinite(res["dlog_b"]).mean() > 0.5, "too few usable differential profiles"
    assert b_long > 0.1, f"expected inside-out long-baseline growth, got b={b_long}"
    assert dout_long > din_long, "outer should grow more than inner (inside-out)"
    print(f"\n[verdict] differential growth is INSIDE-OUT and MULTIPLICATIVE: the profile evolves as "
          f"Sigma_low/Sigma_high ~ R^b (long-baseline b={b_long:+.2f}), the OUTER density growing "
          f"~{10**dout_long:.0f}x more than the inner. A centred Gaussian deposit (which adds most at "
          f"R=0) is the WRONG shape; the profile flattens and extends with time.")


# %% ---- figures -------------------------------------------------------------
def _stack_dlog(sigma, lo, hi):
    """Median and 16-84 band of Dlog Sigma(R) over galaxies for one epoch pair."""
    with np.errstate(divide="ignore", invalid="ignore"):
        dlog = np.log10(sigma[:, lo, :]) - np.log10(sigma[:, hi, :])
    return np.nanmedian(dlog, axis=0), np.nanpercentile(dlog, [16, 84], axis=0)


def make_figures(data, res, suffix):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    R = RGRID
    sigma = data["sigma"]
    fitm = (R >= FIT_LO) & (R <= FIT_HI)

    # FIG A: the robust diagnostic -- median fractional growth Dlog Sigma(R)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2))
    for j, (lo, hi, lab) in enumerate(PAIRS):
        med, _ = _stack_dlog(sigma, lo, hi)
        a1.plot(R, med, "-o", color=plt.cm.viridis(j / 4), ms=3, lw=1.8, label=f"pair {lab}")
    a1.axhline(0, color="k", lw=0.8); a1.axvspan(0, FIT_LO, color="0.9")
    a1.set_xscale("log"); a1.set_xlim(2, 130); a1.set_xlabel("R [kpc]")
    a1.set_ylabel(r"median $\Delta\log\Sigma_*$ (later $-$ earlier)")
    a1.legend(fontsize=8); a1.set_title("A. Fractional growth rises with R -> inside-out (each adjacent pair)")
    # long baseline z=2 -> z=0.4 with band + power-law fit
    med, (q16, q84) = _stack_dlog(sigma, 0, 4)
    a2.fill_between(R, q16, q84, color=OKABE_ITO[1], alpha=0.25, label="16-84%")
    a2.plot(R, med, "-o", color=OKABE_ITO[1], ms=4, lw=2.2, label="median (z=2$\\to$0.4)")
    good = fitm & np.isfinite(med)
    b, a0 = np.polyfit(np.log10(R[good]), med[good], 1)
    a2.plot(R[good], a0 + b * np.log10(R[good]), "k--", lw=1.8,
            label=f"$\\Delta\\log\\Sigma={a0:.2f}+{b:.2f}\\log R$\n($\\Sigma_{{0.4}}/\\Sigma_2\\propto R^{{{b:.2f}}}$)")
    a2.axhline(0, color="k", lw=0.8); a2.axvspan(0, FIT_LO, color="0.9")
    a2.set_xscale("log"); a2.set_xlim(2, 130); a2.set_xlabel("R [kpc]")
    a2.set_ylabel(r"$\Delta\log\Sigma_*$ ($z{=}2\to0.4$)")
    a2.legend(fontsize=8); a2.set_title("B. Profile evolves as a power-law: multiplicative, NOT a Gaussian")
    fig.suptitle("exp26 — differential stellar-density growth is inside-out and multiplicative "
                 "($\\Sigma\\propto R^{b}$ steepening), not a centred-Gaussian deposit", fontsize=11)
    fig.tight_layout(); save_fig(fig, FIGDIR / f"exp26_dlog_growth{suffix}")

    # FIG B: linear dSigma of three example galaxies + best Gaussian (Gaussian fails)
    i_of = {int(ix): i for i, ix in enumerate(data["index"])}
    sub0 = res[res["pair_idx"] == 0]
    pick = []
    for kl in ["inside_out", "core_drop", "self_similar"]:
        c = sub0[(sub0["klass"] == kl)]
        if len(c):
            pick.append((kl, int(c[np.argmax(c["logmstar"])]["index"])))
    fig2, axes2 = plt.subplots(1, len(pick), figsize=(6 * len(pick), 4.6), squeeze=False)
    for ax, (kl, gi) in zip(axes2[0], pick):
        i = i_of[gi]
        dS = sigma[i, 0] - sigma[i, 1]
        ax.plot(R, dS, "ko", ms=4, label=r"data $\Delta\Sigma$")
        m = fitm & np.isfinite(dS) & (dS > 0)
        if m.sum() >= 8:
            yl = np.log10(dS[m])
            for nfix, col, nm in [(0.5, OKABE_ITO[2], "Gaussian n=0.5"), (1.0, OKABE_ITO[1], "exp n=1")]:
                x = R[m] ** (1.0 / nfix)
                c = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, yl, rcond=None)[0]
                ax.plot(R[m], 10 ** (c[0] * x + c[1]), "-", color=col, lw=1.8, label=nm)
        ax.axhline(0, color="k", lw=0.8); ax.axvspan(0, FIT_LO, color="0.9")
        ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1e6); ax.set_xlim(2, 130)
        ax.set_xlabel("R [kpc]"); ax.set_ylabel(r"$\Delta\Sigma_*$ [$M_\odot$/kpc$^2$]")
        ax.set_title(f"{kl}: gal {gi}", fontsize=10); ax.legend(fontsize=7)
    fig2.suptitle("exp26 — example differential profiles (pair [0.4,0.7]); a centred Gaussian "
                  "over-predicts the centre and misses the extended growth", fontsize=11)
    fig2.tight_layout(); save_fig(fig2, FIGDIR / f"exp26_examples{suffix}")

    # FIG C: fraction with inside-out growth per pair + distribution of slope b
    fig3, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5))
    labels = [p[2] for p in PAIRS]
    f_io = [float(np.mean(np.asarray(res["dlog_b"])[res["pair_idx"] == j] > 0)) for j in range(4)]
    a1.bar(labels, f_io, color=OKABE_ITO[1], alpha=0.85)
    a1.axhline(0.5, color="0.6", lw=0.8, ls=":")
    a1.set_ylabel("fraction of galaxies with $b>0$ (inside-out)"); a1.set_ylim(0, 1)
    a1.set_title("A. Inside-out growth is the majority at every epoch")
    for j, (lo, hi, lab) in enumerate(PAIRS):
        bb = np.asarray(res["dlog_b"])[(res["pair_idx"] == j) & np.isfinite(res["dlog_b"])]
        if len(bb):
            a2.hist(bb, bins=np.linspace(-0.6, 1.2, 30), histtype="step", lw=2,
                    color=plt.cm.viridis(j / 4), label=f"{lab} (med {np.median(bb):+.2f})")
    a2.axvline(0, color="k", ls="--", lw=1.2, label="self-similar (b=0)")
    a2.set_xlabel(r"power-law slope $b$ ($\Sigma_{\rm low}/\Sigma_{\rm high}\propto R^b$)")
    a2.set_ylabel("count"); a2.legend(fontsize=8)
    a2.set_title("B. b>0 almost always -> profiles flatten/extend with time")
    fig3.suptitle("exp26 — inside-out multiplicative growth dominates across all epochs "
                  "(per-galaxy inner-sign is noisy for small-Dt pairs)", fontsize=11)
    fig3.tight_layout(); save_fig(fig3, FIGDIR / f"exp26_classes{suffix}")


if __name__ == "__main__":
    main()
