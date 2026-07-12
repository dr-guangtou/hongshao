"""Standardized tiered QA for multi-epoch CoG predictions (graduated from exp29
``mass_qa`` + exp30 ``profile_qa``; exp31).

HongShao's forward models are scored on a ladder, and every approach must be
scored identically. One entry point, ``evaluate(model_cogs, data_cogs, R,
anchor_z)``, reports three tiers:

1. **Aperture masses** — M*(<R) at fixed kpc and R_half-relative radii: the
   basic-goal scoreboard (per-epoch median bias + dex scatter).
2. **Annulus + outskirt masses** — M* in [10,30], [30,50], [50,100], [100,150]
   kpc shells (+ Re analogs) and M*(>R) envelopes: the sensitive shape tier.
   Plus the **observational planes**: joint 2-D distributions (e.g. M*(<30 kpc)
   vs M*[50,100 kpc]) — the predicted population must reproduce the truth's
   relation (slope/scatter), because that plane is what observations use.
3. **Profile** — worst-radius max|rel| quoted over ALL radii AND R>5 kpc (the
   inner 2-5 kpc is marginally resolved; exp07), with two visual products:
   median CoG by stellar-mass tercile with residual profiles, and a
   best/worst-case gallery with the worst radius marked.

Conventions: CoGs are linear masses, shape (n, nz, nr); R_half is measured on
the TRUTH CoG and shared by model and truth (isolates mass error from size
error); relative errors are (model-truth)/truth; dex scatter is the std of
log10(model/truth) over galaxies.

Use: ``from hongshao import qa; res = qa.evaluate(model, truth, R, ANCHOR_Z,
name="...", figdir=...)`` — returns a dict of all per-galaxy measurements and
summary tables for programmatic comparison (the exp31 scoreboard pattern).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from hongshao.plotting import set_style, save_fig

# --- the standard radial bins (edit here to change the standard) --------------
KPC_APER = [10.0, 30.0, 50.0, 100.0]          # M*(<R) [kpc]
KPC_ANN = [(10.0, 30.0), (30.0, 50.0), (50.0, 100.0), (100.0, 150.0)]
KPC_ENV = [50.0, 100.0]                        # M*(>R) [kpc]
RE_APER = [1.0, 2.0, 4.0]                      # M*(<k R_half)
RE_ANN = [(1.0, 2.0), (2.0, 4.0)]
RE_ENV = [2.0, 4.0]                            # M*(>k R_half)
PLANES = [("kpc:M(<30)", "kpc:M(30-50)"), ("kpc:M(<30)", "kpc:M(50-100)"),
          ("Re:M(<2Re)", "Re:M(2-4Re)")]
RMIN_KPC = 5.0                                 # inner 2-5 kpc marginally resolved


def half_mass_radius(cog, R):
    """Half-mass radius from a (monotonic) CoG; log-log extrapolation below the
    grid for compact high-z galaxies whose R_half < R[0]."""
    target = 0.5 * cog[-1]
    if target <= cog[0]:
        sl = (np.log(cog[1]) - np.log(cog[0])) / (np.log(R[1]) - np.log(R[0]))
        return float(max(R[0] * np.exp((np.log(target) - np.log(cog[0])) / max(sl, 1e-3)), 0.3))
    return float(np.interp(target, cog, R))


def measure(cog, R, rhalf):
    """Standard aperture/annulus/envelope masses for one CoG, keyed '<set>:label'."""
    mtot = cog[-1]
    m = {}
    for a in KPC_APER:
        m[f"kpc:M(<{int(a)})"] = float(np.interp(a, R, cog))
    for a, b in KPC_ANN:
        m[f"kpc:M({int(a)}-{int(b)})"] = float(np.interp(b, R, cog) - np.interp(a, R, cog))
    for a in KPC_ENV:
        m[f"kpc:M(>{int(a)})"] = float(mtot - np.interp(a, R, cog))
    for k in RE_APER:
        m[f"Re:M(<{k:g}Re)"] = float(np.interp(k * rhalf, R, cog))
    for a, b in RE_ANN:
        m[f"Re:M({a:g}-{b:g}Re)"] = float(np.interp(b * rhalf, R, cog)
                                          - np.interp(a * rhalf, R, cog))
    for k in RE_ENV:
        m[f"Re:M(>{k:g}Re)"] = float(mtot - np.interp(k * rhalf, R, cog))
    return m


def measure_all(model_cogs, data_cogs, R):
    """Truth/model mass arrays (n, nz) per quantity; R_half from TRUTH (shared)."""
    n, nz = data_cogs.shape[:2]
    keys = list(measure(data_cogs[0, 0], R, 1.0).keys())
    truth = {k: np.zeros((n, nz)) for k in keys}
    model = {k: np.zeros((n, nz)) for k in keys}
    rhalf = np.zeros((n, nz))
    for i in range(n):
        for j in range(nz):
            rh = half_mass_radius(data_cogs[i, j], R)
            rhalf[i, j] = rh
            md = measure(data_cogs[i, j], R, rh)
            mm = measure(model_cogs[i, j], R, rh)
            for k in keys:
                truth[k][i, j] = md[k]
                model[k][i, j] = mm[k]
    return truth, model, rhalf, keys


def relerr(model, truth):
    return (model - truth) / np.clip(truth, 1.0, None)


def dex_scatter(model, truth, axis=0):
    """Std over galaxies of log10(model/truth); NaN-safe, floors masses at 1 Msun."""
    d = np.log10(np.clip(model, 1.0, None)) - np.log10(np.clip(truth, 1.0, None))
    return np.nanstd(d, axis=axis)


def profile_maxrel(model_cogs, data_cogs, R, rmin=None):
    """Worst-radius |rel| per galaxy/epoch (n, nz), optionally over R > rmin."""
    m = R > (rmin if rmin is not None else 0.0)
    return np.abs((model_cogs[..., m] - data_cogs[..., m])
                  / data_cogs[..., m]).max(axis=-1)


def plane_stats(logx, logy):
    """OLS slope + vertical scatter + Spearman rho of a log-log relation."""
    from scipy.stats import spearmanr
    good = np.isfinite(logx) & np.isfinite(logy)
    if good.sum() < 3:
        return dict(slope=np.nan, scatter=np.nan, rho=np.nan)
    b = np.polyfit(logx[good], logy[good], 1)
    res = logy[good] - np.polyval(b, logx[good])
    return dict(slope=float(b[0]), scatter=float(np.std(res)),
                rho=float(spearmanr(logx[good], logy[good])[0]))


def energy_distance_2d(A, B):
    """Energy distance between two 2-D samples (Szekely & Rizzo): a true metric
    on distributions (0 iff equal), sensitive to location/spread/shape/correlation
    at once — unlike the 2-D K-S, it needs no per-case calibration and no kernel
    bandwidth. A (n,2), B (m,2); rows with non-finite entries are dropped."""
    A = A[np.isfinite(A).all(axis=1)]
    B = B[np.isfinite(B).all(axis=1)]
    if len(A) < 2 or len(B) < 2:
        return np.nan

    def mean_pdist(X, Y):
        d2 = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
        return np.sqrt(np.clip(d2, 0.0, None)).mean()

    e2 = 2.0 * mean_pdist(A, B) - mean_pdist(A, A) - mean_pdist(B, B)
    return float(np.sqrt(max(e2, 0.0)))


def plane_energy(truth_xy, model_xy, n_split=8, seed=0):
    """dict(energy, floor, energy_ratio, energy_ratio_centered).

    Both samples are standardized by the TRUTH's per-axis std (same transform),
    so values are dimensionless and comparable across planes/epochs. The floor
    is the median energy distance between two random halves of the truth (pure
    sampling noise): ratio ~ 1 means the predicted population is statistically
    indistinguishable from the real one. The CENTERED ratio shifts each sample
    to its own per-axis median first, isolating shape/spread mismatch (missing
    population diversity) from location mismatch (bias, fixable by
    recalibrating the amplitude model)."""
    T = truth_xy[np.isfinite(truth_xy).all(axis=1)]
    M = model_xy[np.isfinite(model_xy).all(axis=1)]
    sd = T.std(axis=0) + 1e-12
    T, M = T / sd, M / sd
    e = energy_distance_2d(T, M)
    e_c = energy_distance_2d(T - np.median(T, axis=0), M - np.median(M, axis=0))
    rng = np.random.default_rng(seed)
    floors = []
    for _ in range(n_split):
        perm = rng.permutation(len(T))
        h = len(T) // 2
        floors.append(energy_distance_2d(T[perm[:h]], T[perm[h:]]))
    floor = max(float(np.median(floors)), 1e-12)
    return dict(energy=e, floor=floor, energy_ratio=e / floor,
                energy_ratio_centered=e_c / floor)


# --- the standard entry point --------------------------------------------------
def evaluate(model_cogs, data_cogs, R, anchor_z, name="model", figdir=None,
             verbose=True, figures=True, bin_by=None, bin_label=None):
    """Tiered QA for one model. Prints the report, writes the standard figures
    (mass tables per bin set, observational planes, profile visual QA), and
    returns a dict with all measurements for programmatic comparison."""
    set_style()
    model_cogs, data_cogs = np.asarray(model_cogs), np.asarray(data_cogs)
    n, nz = data_cogs.shape[:2]
    truth, model, rhalf, keys = measure_all(model_cogs, data_cogs, R)
    mr_all = profile_maxrel(model_cogs, data_cogs, R)
    mr_out = profile_maxrel(model_cogs, data_cogs, R, rmin=RMIN_KPC)
    planes = {}
    for kx, ky in PLANES:
        lt = dict(x=np.log10(np.clip(truth[kx], 1.0, None)),
                  y=np.log10(np.clip(truth[ky], 1.0, None)))
        lm = dict(x=np.log10(np.clip(model[kx], 1.0, None)),
                  y=np.log10(np.clip(model[ky], 1.0, None)))
        per_epoch = []
        for j in range(nz):
            st = plane_stats(lt["x"][:, j], lt["y"][:, j])
            sm = plane_stats(lm["x"][:, j], lm["y"][:, j])
            sm.update(plane_energy(
                np.column_stack([lt["x"][:, j], lt["y"][:, j]]),
                np.column_stack([lm["x"][:, j], lm["y"][:, j]])))
            per_epoch.append((st, sm))
        planes[(kx, ky)] = per_epoch

    if verbose:
        print(f"\n=== QA [{name}]  (n={n}) ===")
        print("  tier 1+2 — masses, per epoch: median rel bias % / dex scatter")
        print(f"    {'quantity':>16s} | " + " | ".join(f"z={z}".rjust(13) for z in anchor_z))
        for k in keys:
            cells = []
            for j in range(nz):
                bias = 100 * np.nanmedian(relerr(model[k][:, j], truth[k][:, j]))
                sc = dex_scatter(model[k][:, j], truth[k][:, j])
                cells.append(f"{bias:+6.1f}%/{sc:.3f}")
            print(f"    {k:>16s} | " + " | ".join(cells))
        print("\n  tier 2b — observational planes (log-log): slope / scatter / rho, "
              "truth -> model | energy-distance ratio to sampling floor")
        for (kx, ky), st in planes.items():
            for j in range(nz):
                t, mo = st[j]
                print(f"    {kx} vs {ky}  z={anchor_z[j]}: "
                      f"{t['slope']:+.2f}/{t['scatter']:.3f}/{t['rho']:+.2f} -> "
                      f"{mo['slope']:+.2f}/{mo['scatter']:.3f}/{mo['rho']:+.2f} | "
                      f"E/floor {mo['energy_ratio']:.1f} "
                      f"(centered {mo['energy_ratio_centered']:.1f})")
        print("\n  tier 3 — profile max|rel| median per epoch (all R | R>5 kpc):")
        row_a = " | ".join(f"{100*np.nanmedian(mr_all[:, j]):5.1f}%" for j in range(nz))
        row_o = " | ".join(f"{100*np.nanmedian(mr_out[:, j]):5.1f}%" for j in range(nz))
        print(f"    all R   : {row_a} | avg {100*np.nanmean(np.nanmedian(mr_all, 0)):.1f}%")
        print(f"    R>5 kpc : {row_o} | avg {100*np.nanmean(np.nanmedian(mr_out, 0)):.1f}%")

    if figures and figdir is not None:
        figdir = Path(figdir)
        figdir.mkdir(parents=True, exist_ok=True)
        _mass_figure("kpc", [k for k in keys if k.startswith("kpc:")],
                     truth, model, anchor_z, name, figdir)
        _mass_figure("Re", [k for k in keys if k.startswith("Re:")],
                     truth, model, anchor_z, name, figdir)
        _plane_figure(truth, model, anchor_z, name, figdir)
        _bins_figure(model_cogs, data_cogs, R, anchor_z, name, figdir,
                     bin_by=bin_by, bin_label=bin_label)
        _cases_figure(model_cogs, data_cogs, R, anchor_z, mr_all, name, figdir)

    return dict(truth=truth, model=model, rhalf=rhalf, keys=keys,
                mr_all=mr_all, mr_out=mr_out, planes=planes)


# --- figures ---------------------------------------------------------------------
def _zcolors(nz):
    return [matplotlib.colormaps["cividis"](v) for v in np.linspace(0.0, 0.92, nz)]


def _mass_figure(tag, keys, truth, model, anchor_z, name, figdir):
    """Two figures per bin set: cumulative apertures ('aper') and differential
    annuli+envelopes ('diff'). Each quantity is a COLUMN: truth-vs-model on top,
    relative error below, vertically stacked with a shared per-quantity x-axis
    whose range follows that quantity's own truth values."""
    groups = {"aper": [k for k in keys if "(<" in k],
              "diff": [k for k in keys if "(<" not in k]}
    for gtag, gkeys in groups.items():
        if gkeys:
            _mass_group_figure(tag, gtag, gkeys, truth, model, anchor_z, name, figdir)


def _mass_group_figure(tag, gtag, keys, truth, model, anchor_z, name, figdir):
    nz = len(anchor_z)
    cols = _zcolors(nz)
    ncol = len(keys)
    fig, axes = plt.subplots(2, ncol, figsize=(3.1 * ncol, 6.4), squeeze=False,
                             sharex="col", height_ratios=[2, 1])
    for ci, key in enumerate(keys):
        X, Y = truth[key], model[key]
        av, ar = axes[0, ci], axes[1, ci]
        for j in range(nz):
            x, y = X[:, j], Y[:, j]
            good = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
            av.scatter(x[good], y[good], s=8, c=[cols[j]], alpha=0.4, edgecolors="none",
                       label=f"z={anchor_z[j]}" if ci == 0 else None)
            re = relerr(y, x)
            ar.scatter(x[good], re[good], s=8, c=[cols[j]], alpha=0.3, edgecolors="none")
            if good.any():
                ar.plot(np.median(x[good]), np.median(re[good]), "o", c=cols[j], ms=8,
                        mec="k", mew=0.6)
        pos = (X > 0) & np.isfinite(X)
        if pos.any():
            # robust per-quantity range: near-empty bins (~1 Msun) must not
            # stretch the axes; the residual panel still shows every galaxy
            lo = np.percentile(X[pos], 0.5) / 2.0
            hi = np.percentile(X[pos], 99.5) * 2.0
            av.plot([lo, hi], [lo, hi], "k--", lw=1, zorder=0)
            av.set_xlim(lo, hi)
            av.set_ylim(lo, hi)
        av.set(xscale="log", yscale="log", title=key)
        ar.axhline(0, c="0.5", lw=0.8)
        for gpm in (0.1, -0.1):
            ar.axhline(gpm, c="0.7", ls=":", lw=0.8)
        ar.set(xscale="log", ylim=(-0.6, 0.6), xlabel=r"truth $M_*$ [$M_\odot$]")
        if ci == 0:
            av.set_ylabel(r"model $M_*$ [$M_\odot$]")
            ar.set_ylabel("(model$-$truth)/truth")
            av.legend(fontsize=7, loc="upper left", markerscale=1.6)
    unit = "physical kpc" if tag == "kpc" else r"$R_{\rm half}$ units"
    kind = "cumulative apertures" if gtag == "aper" else "annuli + envelopes"
    fig.suptitle(f"QA [{name}] — {unit}: {kind}", fontsize=12)
    fig.tight_layout()
    print("wrote", save_fig(fig, figdir / f"qa_mass_{tag}_{gtag}_{name}")[0])


def _plane_figure(truth, model, anchor_z, name, figdir):
    """Observational planes: truth (filled) vs model (open), one row per plane."""
    nz = len(anchor_z)
    cols = _zcolors(nz)
    fig, axes = plt.subplots(len(PLANES), nz, figsize=(3.1 * nz, 3.2 * len(PLANES)),
                             squeeze=False)
    for ri, (kx, ky) in enumerate(PLANES):
        for j in range(nz):
            ax = axes[ri, j]
            lx, ly = (np.log10(np.clip(truth[kx][:, j], 1.0, None)),
                      np.log10(np.clip(truth[ky][:, j], 1.0, None)))
            mx, my = (np.log10(np.clip(model[kx][:, j], 1.0, None)),
                      np.log10(np.clip(model[ky][:, j], 1.0, None)))
            ax.scatter(lx, ly, s=14, c=[cols[j]], alpha=0.75, edgecolors="none",
                       label="truth")
            ax.scatter(mx, my, s=18, facecolors="none", edgecolors="0.25", lw=0.7,
                       label="model")
            st, sm = plane_stats(lx, ly), plane_stats(mx, my)
            ax.set_title(f"z={anchor_z[j]}  slope {st['slope']:+.2f}->{sm['slope']:+.2f}"
                         f"\nscatter {st['scatter']:.3f}->{sm['scatter']:.3f}",
                         fontsize=8)
            if j == 0:
                ax.set_ylabel(f"log {ky}")
            ax.set_xlabel(f"log {kx}")
            if ri == 0 and j == 0:
                ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(f"QA [{name}] — observational planes: truth (filled) vs model (open)",
                 fontsize=12)
    fig.tight_layout()
    print("wrote", save_fig(fig, figdir / f"qa_planes_{name}")[0])


def _bins_figure(model_cogs, data_cogs, R, anchor_z, name, figdir,
                 bin_by=None, bin_label=None):
    """Median data-vs-model CoG per tercile of ``bin_by`` + median residuals.

    Bin by a MODEL INPUT (halo mass) whenever available: binning by the truth
    stellar mass converts unexplained amplitude scatter into apparent bin-wise
    bias (regression to the mean) for any conditional-mean model. The truth-M*
    binning remains the fallback (and is the view an observed, M*-selected
    sample would give)."""
    nz = len(anchor_z)
    cols = _zcolors(nz)
    if bin_by is None:
        bin_by = np.log10(data_cogs[:, 0, -1])
        bin_label = bin_label or "logM* (z=0.4)"
    bin_label = bin_label or "bin quantity"
    edges = np.quantile(bin_by, [0, 1 / 3, 2 / 3, 1])
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.5), sharex=True,
                             height_ratios=[2, 1])
    for b in range(3):
        m = (bin_by >= edges[b]) & (bin_by <= edges[b + 1] + 1e-9)
        ax, rax = axes[0, b], axes[1, b]
        for k in range(nz):
            med_d = np.median(np.log10(data_cogs[m, k]), axis=0)
            med_m = np.median(np.log10(np.clip(model_cogs[m, k], 1.0, None)), axis=0)
            rel = 100 * np.median((model_cogs[m, k] - data_cogs[m, k])
                                  / data_cogs[m, k], axis=0)
            # amplitude-pinned residual: rescale each model CoG to the true
            # total, isolating SHAPE error from amplitude regression-to-the-
            # mean (binning by truth M* makes any conditional mean look
            # biased by ~ the unexplained amplitude scatter)
            pin = model_cogs[m, k] * (data_cogs[m, k][:, -1:]
                                      / np.clip(model_cogs[m, k][:, -1:], 1.0, None))
            rel_pin = 100 * np.median((pin - data_cogs[m, k]) / data_cogs[m, k],
                                      axis=0)
            ax.plot(R, med_d, "-", c=cols[k], lw=1.8,
                    label=f"z={anchor_z[k]}" if b == 0 else None)
            ax.plot(R, med_m, "--", c=cols[k], lw=1.4)
            rax.plot(R, rel, "-", c=cols[k], lw=1.4,
                     label="raw" if (b == 0 and k == 0) else None)
            rax.plot(R, rel_pin, ":", c=cols[k], lw=1.4,
                     label="amplitude-pinned (shape)" if (b == 0 and k == 0) else None)
        rax.axhline(0, c="0.6", lw=0.8)
        for y in (-20, 20):
            rax.axhline(y, c="0.8", lw=0.7, ls=":")
        if b == 0:
            rax.legend(fontsize=7, loc="upper right")
        ax.set(xscale="log",
               title=f"{bin_label} {edges[b]:.2f}-{edges[b+1]:.2f}  (n={m.sum()})")
        rax.set(xscale="log", xlabel="R [kpc]", ylim=(-60, 60))
    axes[0, 0].set(ylabel="median log$_{10}$ M$_*$(<R) [M$_\\odot$]")
    axes[1, 0].set(ylabel="median (model$-$data)/data [%]")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"QA [{name}] — CoGs by {bin_label} tercile "
                 "(data solid, model dashed)", fontsize=12)
    fig.tight_layout()
    print("wrote", save_fig(fig, figdir / f"qa_bins_{name}")[0])


def _cases_figure(model_cogs, data_cogs, R, anchor_z, mr, name, figdir):
    """Best/worst gallery by epoch-avg max|rel|, worst radius marked."""
    nz = len(anchor_z)
    cols = _zcolors(nz)
    logms = np.log10(data_cogs[:, 0, -1])
    order = np.argsort(np.nanmean(mr, axis=1))
    picks = list(order[:2]) + list(order[-2:])
    tags = ["best 1", "best 2", "worst 2", "worst 1"]
    fig, axes = plt.subplots(2, 4, figsize=(18.5, 7.5), sharex=True,
                             height_ratios=[2, 1])
    for c, (i, tag) in enumerate(zip(picks, tags)):
        ax, rax = axes[0, c], axes[1, c]
        for k in range(nz):
            rel = 100 * (model_cogs[i, k] - data_cogs[i, k]) / data_cogs[i, k]
            ax.plot(R, np.log10(data_cogs[i, k]), "-", c=cols[k], lw=1.8,
                    label=f"z={anchor_z[k]}" if c == 0 else None)
            ax.plot(R, np.log10(np.clip(model_cogs[i, k], 1.0, None)), "--",
                    c=cols[k], lw=1.4)
            rax.plot(R, rel, "-", c=cols[k], lw=1.4)
            j = int(np.argmax(np.abs(rel)))
            rax.plot(R[j], rel[j], "o", c=cols[k], ms=5, mec="0.2", mew=0.5)
        rax.axhline(0, c="0.6", lw=0.8)
        ax.set(xscale="log", title=f"{tag}: idx {i}, logM*={logms[i]:.2f}\n"
               f"epoch-avg max|rel| {100*np.nanmean(mr[i]):.0f}%")
        rax.set(xscale="log", xlabel="R [kpc]")
    axes[0, 0].set(ylabel="log$_{10}$ M$_*$(<R) [M$_\\odot$]")
    axes[1, 0].set(ylabel="(model$-$data)/data [%]")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"QA [{name}] — best/worst cases (data solid, model dashed; "
                 "dot = worst radius)", fontsize=12)
    fig.tight_layout()
    print("wrote", save_fig(fig, figdir / f"qa_cases_{name}")[0])


def demo():
    """Self-check on synthetic CoGs with known errors: identity -> zero bias /
    scatter / max|rel| and identical plane stats; a global +10% amplitude scaling
    -> +10% bias on every mass, ~0 dex scatter, 10% max|rel|."""
    rng = np.random.default_rng(0)
    R = np.geomspace(2.0, 150.0, 24)
    n, nz = 12, 3
    sig = rng.uniform(5.0, 40.0, (n, nz, 1))
    amp = 10.0 ** rng.uniform(10.5, 11.5, (n, 1, 1)) * np.ones((1, nz, 1))
    truth = amp * (1.0 - np.exp(-R[None, None, :] ** 2 / (2.0 * sig ** 2)))
    res = evaluate(truth, truth, R, [0.4, 1.0, 2.0], name="identity",
                   verbose=False, figures=False)
    for k in res["keys"]:
        assert np.nanmax(np.abs(relerr(res["model"][k], res["truth"][k]))) < 1e-12
        assert dex_scatter(res["model"][k], res["truth"][k]).max() < 1e-12
    assert res["mr_all"].max() < 1e-12 and res["mr_out"].max() < 1e-12
    for st in res["planes"].values():
        for t, m in st:
            assert abs(t["slope"] - m["slope"]) < 1e-9

    # energy distance: identical samples -> 0; a large shift -> far above floor;
    # a same-distribution resample -> ratio ~ 1
    rng2 = np.random.default_rng(1)
    A = rng2.normal(size=(400, 2))
    assert energy_distance_2d(A, A.copy()) < 1e-12
    shift = plane_energy(A, A + 3.0)
    same = plane_energy(A, rng2.normal(size=(400, 2)))
    tight = plane_energy(A, 0.3 * rng2.normal(size=(400, 2)))
    assert shift["energy_ratio"] > 10, shift
    assert shift["energy_ratio_centered"] < 3, shift      # pure shift: centered ~ floor
    assert same["energy_ratio"] < 3, same
    assert tight["energy_ratio_centered"] > 3, tight      # too tight: centered catches it

    res2 = evaluate(1.1 * truth, truth, R, [0.4, 1.0, 2.0], name="x1.1",
                    verbose=False, figures=False)
    for k in res2["keys"]:
        re = relerr(res2["model"][k], res2["truth"][k])
        nonzero = res2["truth"][k] > 1.0        # the truth-floor clip guards ~empty bins
        assert np.allclose(re[nonzero], 0.1, atol=1e-6), k
        d = np.log10(res2["model"][k][nonzero]) - np.log10(res2["truth"][k][nonzero])
        assert np.allclose(d, np.log10(1.1), atol=1e-9), k
    assert np.allclose(res2["mr_all"], 0.1, atol=1e-9)
    mr_rmin = profile_maxrel(1.1 * truth, truth, R, rmin=RMIN_KPC)
    assert np.allclose(mr_rmin, 0.1, atol=1e-9)
    print("qa.demo OK: identity exact, +10% scaling -> +10% bias everywhere, "
          "0 dex scatter, 10% max|rel| (all-R and R>5); energy distance: 0 on "
          f"identity; shift {shift['energy_ratio']:.0f} (centered "
          f"{shift['energy_ratio_centered']:.1f}); resample "
          f"{same['energy_ratio']:.1f}; over-tight centered "
          f"{tight['energy_ratio_centered']:.0f}")


if __name__ == "__main__":
    demo()
