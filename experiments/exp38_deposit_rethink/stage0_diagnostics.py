"""exp38 stage 0 — three cheap data diagnostics BEFORE any model building.

0.1 `similarity` — are profile SHAPES self-similar in R/R_half across epochs?
    Per galaxy, compare the normalized log CoG shape at z=2.0 vs z=0.4 in
    half-mass-radius units (x = R/R_half, 1..4) against the same comparison
    at fixed physical radii spanning the same band. Strong per-galaxy
    collapse in x → the evolving-Re template family (candidate 1j) is live.

0.2 `autopsy` — what deposit shape do the data demand? Stack the measured
    epoch-to-epoch surface-density growth (linear dSigma, median per logM*
    tercile x adjacent epoch pair), locate its peak radius (centred vs
    shell-like), and fit a Sersic index to its positive part (wing
    steepness). Saves the stacked kernels for the empirical-kernel
    candidate (1f).

0.3 `probe` — the rail-removal probe: refit the exp29 single-epoch deposit
    model on the massive tercile of the dev subsample at z=0.4 and z=2.0
    with the deposit swapped Gaussian -> Sersic-n free -> shell-p free.
    If massive galaxies prefer n>0.5 (or p>0) at a finite scale and better
    fit, the data want wings/shells, not a bigger Gaussian.

Run: PYTHONPATH=. uv run python experiments/exp38_deposit_rethink/\
stage0_diagnostics.py {all|similarity|autopsy|probe|demo}
"""
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/exp29_outer_deposit"))
sys.path.insert(0, str(HERE))

from shapes import gauss_cog, sersic_cog, shell_cog                   # noqa: E402
from deposit import width_t, eff_two_epoch, deposited                 # noqa: E402
from hongshao.qa import half_mass_radius, _tex                        # noqa: E402
from hongshao.tng_data import COG_RAD_KPC                             # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO          # noqa: E402

set_style()
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
R = COG_RAD_KPC
ANCHOR_Z = [0.4, 0.7, 1.0, 1.5, 2.0]
R_IN = 3.0
EVAL = R > R_IN
XGRID = np.geomspace(1.0, 4.0, 10)              # R/R_half band shared by all


def _load_pop():
    pop = np.load(POP_NPZ)
    return pop, pop["data"], pop["logms"]


def _rhalf_all(data):
    n = len(data)
    rh = np.empty((n, 5))
    for i in range(n):
        for k in range(5):
            rh[i, k] = half_mass_radius(data[i, k], R)
    return rh


# --------------------------------------------------------------------------- #
# 0.1 similarity                                                               #
# --------------------------------------------------------------------------- #
def _norm_shape_at(data_ik, radii):
    """log10 M(<r)/M(<148) interpolated in log-log at ``radii`` (clipped to
    the measured grid; the CoG is monotone so log-log interp is safe)."""
    y = np.log10(data_ik) - np.log10(data_ik[-1])
    return np.interp(np.log10(np.clip(radii, R[0], R[-1])), np.log10(R), y)


def cmd_similarity():
    pop, data, logms = _load_pop()
    n = len(data)
    rh = _rhalf_all(data)
    print(f"stage 0.1 — R/R_half self-similarity (n={n})")
    print("  median R_half [kpc] per epoch: "
          + " ".join(f"z{z}: {np.median(rh[:, k]):.1f}"
                     for k, z in enumerate(ANCHOR_Z)))

    # per-galaxy shape drift z=2.0 vs z=0.4, in Re units vs fixed kpc
    band_kpc = np.median(rh[:, 0]) * XGRID          # same band, physical kpc
    drift_re = np.empty(n)
    drift_kpc = np.empty(n)
    shape_re = np.empty((n, 5, len(XGRID)))
    for i in range(n):
        for k in range(5):
            shape_re[i, k] = _norm_shape_at(data[i, k], XGRID * rh[i, k])
        y04 = _norm_shape_at(data[i, 0], band_kpc)
        y20 = _norm_shape_at(data[i, 4], band_kpc)
        drift_kpc[i] = np.sqrt(np.mean((y20 - y04) ** 2))
        drift_re[i] = np.sqrt(np.mean((shape_re[i, 4] - shape_re[i, 0]) ** 2))

    edges = np.quantile(logms, [0, 1 / 3, 2 / 3, 1])
    print("\n  per-galaxy shape drift z=2.0 vs z=0.4 [dex rms over the "
          "1-4 R_half band] — R/R_half coords vs fixed-kpc coords:")
    for b in range(3):
        m = (logms >= edges[b]) & (logms <= edges[b + 1] + 1e-9)
        print(f"    logM* {edges[b]:.2f}-{edges[b+1]:.2f}: "
              f"Re-coords {np.median(drift_re[m]):.3f}  |  "
              f"kpc-coords {np.median(drift_kpc[m]):.3f}  |  "
              f"ratio {np.median(drift_re[m]) / np.median(drift_kpc[m]):.2f}")

    # population shape variance at each epoch, both coordinates
    print("\n  population shape scatter [dex, median over the band] per epoch"
          " (across-galaxy std of the normalized log CoG):")
    var_re, var_kpc = [], []
    for k in range(5):
        band_k = np.median(rh[:, k]) * XGRID
        raw = np.stack([_norm_shape_at(data[i, k], band_k) for i in range(n)])
        var_re.append(np.median(np.std(shape_re[:, k], axis=0)))
        var_kpc.append(np.median(np.std(raw, axis=0)))
        print(f"    z={ANCHOR_Z[k]}: Re-coords {var_re[-1]:.3f}  |  "
              f"kpc-coords {var_kpc[-1]:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))
    a, b, c = axes
    for k, z in enumerate(ANCHOR_Z):
        med_re = np.median(shape_re[:, k], axis=0)
        band_k = np.median(rh[:, k]) * XGRID
        med_kpc = np.median(np.stack(
            [_norm_shape_at(data[i, k], band_k) for i in range(n)]), axis=0)
        col = OKABE_ITO[k % len(OKABE_ITO)]
        a.plot(XGRID, med_re, "-o", ms=3, c=col, label=f"z={z}")
        b.plot(band_k, med_kpc, "-o", ms=3, c=col)
    a.set(xscale="log", xlabel=r"R / R$_{half}$",
          ylabel=_tex("median log M(<R)/M(<148)"),
          title="shape in half-mass units (epochs collapse?)")
    a.legend(fontsize=8)
    b.set(xscale="log", xlabel="R [kpc]",
          title="same band in physical kpc (epochs spread?)")
    c.scatter(drift_kpc, drift_re, s=5, alpha=0.3, c="#0072B2",
              edgecolors="none")
    lim = max(np.percentile(drift_kpc, 99), np.percentile(drift_re, 99))
    c.plot([0, lim], [0, lim], "k--", lw=1)
    c.set(xlabel="drift in kpc coords [dex]", ylabel="drift in Re coords [dex]",
          title="per-galaxy z=2 vs z=0.4 shape drift")
    fig.suptitle("exp38 stage 0.1 — is the profile shape self-similar with an "
                 "evolving R_half?", fontsize=12)
    fig.tight_layout()
    FIGDIR.mkdir(exist_ok=True)
    print("wrote", save_fig(fig, FIGDIR / "stage0_similarity")[0])
    OUTDIR.mkdir(exist_ok=True)
    np.savez(OUTDIR / "stage0_similarity.npz", rhalf=rh, drift_re=drift_re,
             drift_kpc=drift_kpc, var_re=var_re, var_kpc=var_kpc)


# --------------------------------------------------------------------------- #
# 0.2 autopsy                                                                  #
# --------------------------------------------------------------------------- #
def _sersic_gridfit(mid, y, n_grid=None, a_grid=None):
    """Fit y ~ A exp(-(R/a)^(1/n)) to a positive stacked kernel by grid
    search over (n, a) with the log amplitude analytic. Returns (n, a, rms).
    NOTE: (n, a) are degenerate over a wing-only band — read the kernel's
    half-mass radius (`_sersic_r50`), not the raw scale."""
    n_grid = np.linspace(0.3, 8.0, 40) if n_grid is None else n_grid
    a_grid = np.geomspace(0.2, 300.0, 60) if a_grid is None else a_grid
    ly = np.log10(y)
    best = None
    for nn in n_grid:
        shape = -(mid[:, None] / a_grid[None, :]) ** (1.0 / nn) / np.log(10.0)
        amp = np.mean(ly[:, None] - shape, axis=0)
        rms = np.sqrt(np.mean((ly[:, None] - shape - amp) ** 2, axis=0))
        j = int(np.argmin(rms))
        if best is None or rms[j] < best[2]:
            best = (float(nn), float(a_grid[j]), float(rms[j]))
    return best


def _sersic_r50(n, a):
    """Half-mass radius of a Sersic-n kernel (degeneracy-robust scale)."""
    from scipy.special import gammaincinv
    return float(a * gammaincinv(2.0 * n, 0.5) ** n)


def cmd_autopsy():
    pop, data, logms = _load_pop()
    dA = np.pi * (R[1:] ** 2 - R[:-1] ** 2)
    mid = np.sqrt(R[:-1] * R[1:])
    sig = np.diff(data, axis=-1) / dA                # (n, 5, 23) linear Sigma
    edges = np.quantile(logms, [0, 1 / 3, 2 / 3, 1])
    fit_band = (mid >= 4.0) & (mid <= 120.0)
    print("stage 0.2 — the measured deposit-shape autopsy (stacked median "
          "added Sigma per logM* tercile x adjacent epoch pair):")
    kern = np.full((3, 4, len(mid)), np.nan)
    rows = []
    fig, axes = plt.subplots(3, 4, figsize=(17.5, 10.5), sharex=True)
    for b in range(3):
        m = (logms >= edges[b]) & (logms <= edges[b + 1] + 1e-9)
        for k in range(4):
            dsig = np.median(sig[m, k, :] - sig[m, k + 1, :], axis=0)
            kern[b, k] = dsig
            pos = fit_band & (dsig > 0)
            nfit, afit, rms = _sersic_gridfit(mid[pos], dsig[pos])
            ipk = int(np.argmax(np.where(mid > R_IN, dsig, -np.inf)))
            core_drop = bool(dsig[mid <= 4.0].min() < 0)
            rows.append((b, k, nfit, afit, mid[ipk], core_drop))
            print(f"    T{b+1} z{ANCHOR_Z[k+1]}->z{ANCHOR_Z[k]}: "
                  f"Sersic n {nfit:4.1f}  kernel R50 "
                  f"{_sersic_r50(nfit, afit):6.1f} kpc  "
                  f"peak R {mid[ipk]:5.1f} kpc  "
                  f"core {'DROPS' if core_drop else 'grows'}  "
                  f"(fit rms {rms:.3f} dex)")
            ax = axes[b, k]
            ax.plot(mid, dsig, "ko-", ms=3, lw=1.2, label="stacked added Sigma")
            shape_mod = np.exp(-(mid / afit) ** (1.0 / nfit))
            scl = (dsig[pos] / shape_mod[pos]).mean() if pos.any() else 1.0
            ax.plot(mid, scl * shape_mod, "-", c="#D55E00", lw=1.6,
                    label=f"Sersic n={nfit:.1f}, a={afit:.0f}")
            ax.axhline(0, c="0.7", lw=0.8)
            ax.set(xscale="log", yscale="symlog")
            ax.legend(fontsize=6)
            if b == 0:
                ax.set_title(f"z{ANCHOR_Z[k+1]} to z{ANCHOR_Z[k]}", fontsize=9)
            if k == 0:
                ax.set_ylabel(f"T{b+1} {edges[b]:.1f}-{edges[b+1]:.1f}\n"
                              r"added $\Sigma$ [M$_\odot$ kpc$^{-2}$]",
                              fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("R [kpc]")
    fig.suptitle("exp38 stage 0.2 — stacked epoch-to-epoch added surface "
                 "density + best Sersic wing", fontsize=12)
    fig.tight_layout()
    FIGDIR.mkdir(exist_ok=True)
    print("wrote", save_fig(fig, FIGDIR / "stage0_autopsy")[0])
    OUTDIR.mkdir(exist_ok=True)
    np.savez(OUTDIR / "stage0_kernels.npz", mid=mid, kern=kern, edges=edges,
             rows=np.array([r[:5] for r in rows], float))
    print(f"wrote {OUTDIR / 'stage0_kernels.npz'} (the 1f empirical kernels)")


# --------------------------------------------------------------------------- #
# 0.3 the rail-removal probe                                                   #
# --------------------------------------------------------------------------- #
VARIANTS = ("gauss", "sersic", "shell")
BOX = {"log_s0": (0.3, 3.5), "n": (0.25, 8.0), "p": (0.0, 8.0)}


def _probe_model(q, variant, mah, k, anchor_snap):
    s0, g, be, bl, zc = 10.0 ** q[0], q[1], q[2], q[3], q[4]
    dM = deposited(eff_two_epoch(mah["z"], be, bl, zc), mah["dMh"], 1.0)
    s = width_t(s0, g, mah["t"], mah["t_obs"])
    m = mah["snap"] <= anchor_snap[k]
    if variant == "gauss":
        return gauss_cog(dM[m], s[m], R)
    if variant == "sersic":
        return sersic_cog(dM[m], s[m], q[5], R)
    return shell_cog(dM[m], s[m], q[5], R)


def _probe_penalty(q, variant):
    lo, hi = BOX["log_s0"]
    pen = np.clip(lo - q[0], 0, None) ** 2 + np.clip(q[0] - hi, 0, None) ** 2
    if variant != "gauss":
        lo6, hi6 = BOX["n" if variant == "sersic" else "p"]
        pen += (np.clip(lo6 - q[5], 0, None) ** 2
                + np.clip(q[5] - hi6, 0, None) ** 2)
    return 30.0 * float(pen)


def _fit_probe(mah, data_k, k, variant, anchor_snap):
    def loss(q):
        if q[4] <= -1.0:
            return 1e3
        mod = _probe_model(q, variant, mah, k, anchor_snap)
        if not np.isfinite(mod[-1]) or mod[-1] <= 0:
            return 1e3
        mod = mod * (data_k[-1] / mod[-1])
        v = float(np.sqrt(np.mean(
            ((mod[EVAL] - data_k[EVAL]) / data_k[EVAL]) ** 2)))
        return (v if np.isfinite(v) else 1e3) + _probe_penalty(q, variant)

    starts = []
    for zc0 in (1.0, 2.5):
        base = [np.log10(40.0), 1.5, 4.0, 1.5, zc0]
        if variant == "gauss":
            starts.append(base)
        elif variant == "sersic":
            starts += [base + [0.6], base + [2.0]]
        else:
            starts += [base + [0.5], base + [3.0]]
    best = None
    for q0 in starts:
        r = minimize(loss, q0, method="Nelder-Mead",
                     options=dict(maxiter=3000, xatol=1e-4, fatol=1e-9))
        if best is None or r.fun < best.fun:
            best = r
    mod = _probe_model(best.x, variant, mah, k, anchor_snap)
    mod = mod * (data_k[-1] / mod[-1])
    rel = np.abs((mod[EVAL] - data_k[EVAL]) / data_k[EVAL])
    return best.x, float(np.sqrt(np.mean(rel ** 2))), float(rel.max())


def _dep_r50(variant, q):
    """Degeneracy-robust derived scale: the half-mass radius [kpc] of ONE
    deposit at t = t_obs (raw scale parameters differ in meaning between
    families; R50 is comparable)."""
    rr = np.geomspace(0.1, 3e4, 600)
    one, s = np.array([1.0]), np.array([10.0 ** q[0]])
    if variant == "gauss":
        cog = gauss_cog(one, s, rr)
    elif variant == "sersic":
        cog = sersic_cog(one, s, q[5], rr)
    else:
        cog = shell_cog(one, s, q[5], rr)
    return float(np.interp(0.5, cog, rr))


def cmd_probe():
    from run import dipfree_mah, ANCHOR_SNAP                          # noqa
    from cog_extrapolate import measured_cog                          # noqa
    pop, data, logms = _load_pop()
    dev = pop["dev100"]
    dev_ms = logms[dev]
    massive = dev[dev_ms >= np.quantile(dev_ms, 2 / 3)]
    print(f"stage 0.3 — rail-removal probe (massive tercile of dev100, "
          f"n={len(massive)}, z=0.4 and z=2.0; deposit swapped "
          "gauss -> sersic-n -> shell-p):")
    res = {}
    t0 = time.time()
    for row in massive:
        gi = int(pop["index"][row])
        mah = dipfree_mah(gi)
        logC = measured_cog(gi)
        if mah is None or logC is None:
            continue
        for k in (0, 4):
            data_k = 10.0 ** logC[k]
            for v in VARIANTS:
                q, rms, mx = _fit_probe(mah, data_k, k, v, ANCHOR_SNAP)
                res.setdefault((v, k), []).append(
                    (q[0], q[5] if len(q) > 5 else np.nan, rms, mx,
                     _dep_r50(v, q)))
    print(f"  ({(time.time()-t0)/60:.1f} min)")
    print("\n  variant  epoch |  rel-RMS  max|rel| | log s0 (16/50/84)"
          "      | shape param       | deposit R50 [kpc] | at bound")
    summary = {}
    for k in (0, 4):
        for v in VARIANTS:
            arr = np.array(res[(v, k)])
            ls, sh, rms, mx, r50 = (arr[:, 0], arr[:, 1], arr[:, 2],
                                    arr[:, 3], arr[:, 4])
            lo, hi = BOX["log_s0"]
            nb = int(((ls > hi - 0.02) | (ls < lo + 0.02)).sum())
            if v != "gauss":
                blo, bhi = BOX["n" if v == "sersic" else "p"]
                nb += int(((sh > bhi - 0.02) | (sh < blo + 0.02)).sum())
            pct = np.percentile(ls, [16, 50, 84])
            shp = (" " * 17 if v == "gauss" else
                   f"{np.percentile(sh, 50):5.2f} "
                   f"({np.percentile(sh, 16):.2f}-{np.percentile(sh, 84):.2f})")
            print(f"  {v:>7s}  z={ANCHOR_Z[k]:<3} | {100*np.median(rms):7.2f}%"
                  f" {100*np.median(mx):7.1f}% | "
                  f"{pct[1]:5.2f} ({pct[0]:.2f}-{pct[2]:.2f}) | {shp} | "
                  f"{np.median(r50):8.1f} | {nb}/{len(arr)}")
            summary[(v, k)] = arr
    OUTDIR.mkdir(exist_ok=True)
    np.savez(OUTDIR / "stage0_probe.npz",
             **{f"{v}_k{k}": summary[(v, k)] for v, k in summary})
    # verdict helper: did wings improve the massive z=0.4 fit, off any bound?
    g0 = np.median(summary[("gauss", 0)][:, 2])
    s0_ = np.median(summary[("sersic", 0)][:, 2])
    p0 = np.median(summary[("shell", 0)][:, 2])
    print(f"\n  [read] massive z=0.4 median rel-RMS: gauss {100*g0:.2f}% -> "
          f"sersic {100*s0_:.2f}% -> shell {100*p0:.2f}%; fitted n median "
          f"{np.median(summary[('sersic', 0)][:, 1]):.2f} "
          "(0.5 = Gaussian, higher = heavier wing)")


def demo():
    # similarity math: synthetic self-similar profiles collapse in Re coords
    rng = np.random.default_rng(0)
    n = 30
    rh_true = np.column_stack([np.full(n, 12.0), np.full(n, 4.0)])
    drift = []
    for i in range(n):
        shapes_ = []
        for k, rh in enumerate(rh_true[i]):
            cog = 1.0 / (1.0 + (R / rh) ** -2)      # fixed shape f(R/rh)
            shapes_.append(_norm_shape_at(cog * 10 ** rng.normal(11, 0.1),
                                          XGRID * half_mass_radius(cog, R)))
        drift.append(np.sqrt(np.mean((shapes_[1] - shapes_[0]) ** 2)))
    assert np.median(drift) < 0.02, "self-similar synthetic must collapse"

    # autopsy: the Sersic grid fit recovers a known wing
    mid = np.sqrt(R[:-1] * R[1:])
    y = 3e7 * np.exp(-(mid / 30.0) ** (1.0 / 1.5))
    nfit, afit, rms = _sersic_gridfit(mid, y)
    assert abs(nfit - 1.5) < 0.35 and abs(np.log10(afit / 30.0)) < 0.25, \
        (nfit, afit)

    # probe: penalty box and nesting (sersic n=0.5 == gauss loss at same q)
    q5 = np.array([1.5, 1.2, 4.0, 1.5, 2.0])
    assert _probe_penalty(q5, "gauss") == 0.0
    assert _probe_penalty(np.append(q5, 9.0), "sersic") > 1.0
    mah = dict(z=np.array([0.5, 1.0, 3.0]), t=np.array([8.0, 5.0, 2.0]),
               t_obs=8.0, dMh=np.array([1e12, 5e11, 4e11]),
               snap=np.array([70, 50, 30]))
    a = _probe_model(q5, "gauss", mah, 0, [72] * 5)
    b = _probe_model(np.append(np.array(
        [q5[0] + 0.5 * np.log10(2.0), *q5[1:]]), 0.5), "sersic", mah, 0,
        [72] * 5)
    assert np.abs(a - b).max() < 1e-12, "sersic n=0.5 must nest gauss"
    print("stage0 demo OK: self-similar synthetic collapses in Re coords; "
          "Sersic grid fit recovers (n, a); probe penalty box sane; "
          "sersic n=0.5 nests the Gaussian probe exactly")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "demo":
        demo()
    elif cmd == "similarity":
        cmd_similarity()
    elif cmd == "autopsy":
        cmd_autopsy()
    elif cmd == "probe":
        cmd_probe()
    elif cmd == "all":
        cmd_similarity()
        cmd_autopsy()
        cmd_probe()
    else:
        sys.exit(f"unknown subcommand {cmd!r}")
