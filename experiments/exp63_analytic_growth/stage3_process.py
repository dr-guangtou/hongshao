"""exp63 Stage 3 — the stochastic deposition process: sweep, fit, gates,
predictions.

The mean is Stage 2's fitted two-channel model (frozen). Three noise sources
live INSIDE the history (`process3.py`): lumpy extended-channel accretion
(n_eff events per e-fold of halo growth), a correlated efficiency history
(sigma_eta, ell), and per-galaxy size offsets (sigma_sc, sigma_se). Each
draw is a whole galaxy at every epoch.

  1. TARGETS at z=0.4 from the truth: the scatter of the kpc planes
     M(30-50)|M(<30) and M(50-100)|M(<30), the R50|M* plane, and the widths
     of the tier-2e mass distributions.
  2. LEVERAGE SWEEP, before any fit (exp59's rule): each source alone at
     three amplitudes; a source that cannot widen either kpc plane by 0.03 dex
     without shifting a median by more than 2 per cent is dropped.
  3. FIT the surviving parameters to POPULATION statistics — the sum of the
     energy-distance ratios to the split-half floor on the two kpc planes and
     the R50 plane, plus the tier-2e W1 ratios of the four kpc masses — by
     Nelder-Mead with common random numbers, calibrated on one random half
     and scored on the other, both swaps; then re-fitted on the whole sample.
  4. GATES G3a (kpc-plane E/floor <= 1.5, held out) and G3b (tier-2e width
     ratios within 0.15 of 1, from draws — closes C16), with the mean alone
     as the null.
  5. PREDICTIONS at z=0.7-2.0 with the same draws: tier-2c size persistence,
     the declining fraction (zero by construction for this class — stated),
     the tier-2e widths per epoch.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp63_analytic_growth/stage3_process.py [--smoke] [--theta-default]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term", HERE):
    sys.path.insert(0, str(p))

import engine as E                                       # noqa: E402
import fit as F                                          # noqa: E402
import model2 as M2                                      # noqa: E402
import process3 as P3                                    # noqa: E402
import stage0_cost as S0                                 # noqa: E402
from hongshao import qa                                  # noqa: E402

RULE = "=" * 92
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
ANCHOR_Z = E.ANCHOR_Z
KPC_PLANES = (("kpc:M(<30)", "kpc:M(30-50)"), ("kpc:M(<30)", "kpc:M(50-100)"))
CDF_KEYS = ("kpc:M(<10)", "kpc:M(<30)", "kpc:M(30-50)", "kpc:M(50-100)")
I3 = 3
#: sweep configurations (each source alone)
SWEEP = [("lumpy", {"n_eff": v}) for v in (2.0, 5.0, 20.0)] + \
        [("efficiency", {"sigma_eta": s, "ell": l}) for s in (0.1, 0.2, 0.4) for l in (0.1, 0.5)] + \
        [("size_c", {"sigma_sc": v}) for v in (0.1, 0.2, 0.4)] + \
        [("size_e", {"sigma_se": v}) for v in (0.1, 0.2, 0.4)]
SWEEP_MIN_WIDEN, SWEEP_MAX_SHIFT = 0.03, 0.02
G3A_EFLOOR, G3B_WIDTH_TOL = 1.5, 0.15
N_DRAW_FIT, N_DRAW_SWEEP, N_DRAW_FINAL = 8, 4, 8


# --------------------------------------------------------------------------- #
def r50(cogs, R):
    """(n, nz) half-mass radii of the CoGs (aperture-relative, qa convention)."""
    n, nz = cogs.shape[:2]
    out = np.empty((n, nz))
    for i in range(n):
        for j in range(nz):
            out[i, j] = qa.enclosed_radius(cogs[i, j], R, 0.5)
    return out


def pop_stats(draws, data, R, truth=None, rh_t=None, j=0):
    """Population statistics at epoch j from pooled draws (S, n, nz, nR)."""
    d1 = data[:, j:j + 1]
    if truth is None:
        truth = qa.measure_all(d1, d1, R)[0]
        rh_t = r50(d1, R)[:, 0]
    S = draws.shape[0]
    pooled = {}
    rh_m = []
    for s in range(S):
        m = qa.measure_all(draws[s][:, j:j + 1], d1, R)[1]
        for k in m:
            pooled.setdefault(k, []).append(m[k][:, 0])
        rh_m.append(r50(draws[s][:, j:j + 1], R)[:, 0])
    pooled = {k: np.concatenate(v) for k, v in pooled.items()}
    rh_m = np.concatenate(rh_m)
    out = {}
    for kx, ky in KPC_PLANES:
        tx, ty = np.log10(truth[kx][:, 0]), np.log10(truth[ky][:, 0])
        mx, my = np.log10(np.clip(pooled[kx], 1, None)), np.log10(np.clip(pooled[ky], 1, None))
        ok = np.isfinite(mx) & np.isfinite(my)
        out[f"scatter {ky}|{kx}"] = (qa.plane_stats(tx, ty)["scatter"], qa.plane_stats(mx[ok], my[ok])["scatter"])
        out[f"E/floor {ky}|{kx}"] = qa.plane_energy(np.column_stack([tx, ty]), np.column_stack([mx[ok], my[ok]]))["energy_ratio"]
    tx, ty = np.log10(truth["kpc:M(<148)"][:, 0]) if "kpc:M(<148)" in truth else np.log10(d1[:, 0, -1]), np.log10(rh_t)
    mx = np.log10(np.clip(np.concatenate([draws[s][:, j, -1] for s in range(S)]), 1, None)); my = np.log10(rh_m)
    out["scatter R50|M*"] = (qa.plane_stats(tx, ty)["scatter"], qa.plane_stats(mx, my)["scatter"])
    out["E/floor R50|M*"] = qa.plane_energy(np.column_stack([tx, ty]), np.column_stack([mx, my]))["energy_ratio"]
    for k in CDF_KEYS + ("kpc:M(<100)",):
        tv, mv = np.log10(truth[k][:, 0]), np.log10(np.clip(pooled[k], 1, None))
        out[f"median shift {k}"] = float(np.median(mv) - np.median(tv))
        if k in CDF_KEYS:
            cd = qa.cdf_distances(tv, mv)
            out[f"W1/floor {k}"] = cd["w1_ratio"]
            out[f"width {k}"] = (np.nanpercentile(mv, 84) - np.nanpercentile(mv, 16)) / (np.nanpercentile(tv, 84) - np.nanpercentile(tv, 16))
    return out


def objective_value(st):
    """The fit objective from a pop_stats dict: plane E/floor + tier-2e W1 ratios."""
    return (sum(st[f"E/floor {ky}|{kx}"] for kx, ky in KPC_PLANES) + st["E/floor R50|M*"]
            + sum(st[f"W1/floor {k}"] for k in CDF_KEYS))


# --------------------------------------------------------------------------- #
def main(smoke=False, theta_default=False):
    tag = "_smoke" if smoke else ""
    print(f"{RULE}\nexp63 STAGE 3 — the stochastic deposition process{' (SMOKE)' if smoke else ''}\n{RULE}\n")
    recs, data, mask, lmh, spec_inc, th_inc, _ = S0.build(smoke)
    curves = E.build_curves(recs, verbose=False)
    R = F.R_GRID
    spec2 = M2.Spec2()
    if theta_default:
        theta = M2.default_theta(th_inc); src = "model2.default_theta (no Stage 2 fit yet)"
    else:
        fz = np.load(OUTDIR / f"stage2_fit{tag}.npz", allow_pickle=True)
        theta = np.asarray(fz["theta_best"], float); src = f"stage2_fit{tag}.npz"
    print(f"  mean model: {src}\n  " + "  ".join(f"{n}={v:+.3f}" for n, v in zip(spec2.theta_names, theta)))
    mean = M2.predict2(spec2, theta, curves, R)
    good = np.isfinite(data).all(axis=(1, 2)) & (data > 0).all(axis=(1, 2)) & np.isfinite(mean).all(axis=(1, 2)) & (mean > 0).all(axis=(1, 2))
    idx = np.where(good)[0]
    cv = [curves[i] for i in idx]; dd = data[idx]; mean = mean[idx]
    n = len(idx)
    print(f"  {n} galaxies\n")

    # 1. targets and the null
    truth = qa.measure_all(dd[:, :1], dd[:, :1], R)[0]
    rh_t = r50(dd[:, :1], R)[:, 0]
    st0 = pop_stats(mean[None], dd, R, truth, rh_t)
    print("  TARGETS at z=0.4 (truth) and the MEAN MODEL alone (the null):")
    for k, v in st0.items():
        if k.startswith("scatter"):
            print(f"    {k:<32} truth {v[0]:.3f} dex   mean model {v[1]:.3f} dex")
    for k, v in st0.items():
        if k.startswith("E/floor") or k.startswith("W1/floor") or k.startswith("width"):
            print(f"    {k:<32} {v:.2f}")
    null_obj = objective_value(st0)
    print(f"    objective (sum of E/floor + W1/floor): {null_obj:.2f}")

    # 2. the leverage sweep
    print(f"\n{RULE}\n  LEVERAGE SWEEP — each source alone, {N_DRAW_SWEEP} draws, z=0.4\n{RULE}")
    print(f"  {'source':<11}{'setting':<26}{'sc(30-50|<30)':>14}{'sc(50-100|<30)':>15}{'sc(R50|M*)':>11}{'E/fl 30-50':>11}{'width<10':>9}{'shift<10':>9}{'shift<100':>10}")
    print(f"  {'truth':<11}{'':<26}{st0['scatter kpc:M(30-50)|kpc:M(<30)'][0]:>14.3f}{st0['scatter kpc:M(50-100)|kpc:M(<30)'][0]:>15.3f}{st0['scatter R50|M*'][0]:>11.3f}")
    print(f"  {'mean only':<11}{'':<26}{st0['scatter kpc:M(30-50)|kpc:M(<30)'][1]:>14.3f}{st0['scatter kpc:M(50-100)|kpc:M(<30)'][1]:>15.3f}{st0['scatter R50|M*'][1]:>11.3f}{st0['E/floor kpc:M(30-50)|kpc:M(<30)']:>11.2f}{st0['width kpc:M(<10)']:>9.2f}{st0['median shift kpc:M(<10)']:>+9.3f}{st0['median shift kpc:M(<100)']:>+10.3f}")
    sweep = []
    keep = set()
    base_sc = st0["scatter kpc:M(30-50)|kpc:M(<30)"][1]
    base_sc2 = st0["scatter kpc:M(50-100)|kpc:M(<30)"][1]
    for src_name, setting in SWEEP:
        nz = P3.Noise3(**setting)
        dr = P3.draw(spec2, theta, nz, cv, R, np.random.default_rng(11), n_draw=N_DRAW_SWEEP, epochs=(0,))
        st = pop_stats(dr, dd, R, truth, rh_t)
        # leverage on EITHER kpc plane counts: the outer annulus is where the
        # mean is narrowest (exp61 Stage 4) and the extended-size source acts there
        widen = max(st["scatter kpc:M(30-50)|kpc:M(<30)"][1] - base_sc,
                    st["scatter kpc:M(50-100)|kpc:M(<30)"][1] - base_sc2)
        shift = max(abs(st["median shift kpc:M(<10)"] - st0["median shift kpc:M(<10)"]),
                    abs(st["median shift kpc:M(<100)"] - st0["median shift kpc:M(<100)"]))
        ok = widen >= SWEEP_MIN_WIDEN and shift <= np.log10(1 + SWEEP_MAX_SHIFT)
        if ok:
            keep.add(src_name)
        sweep.append((src_name, setting, st, widen, shift, ok))
        print(f"  {src_name:<11}{str(setting):<26}{st['scatter kpc:M(30-50)|kpc:M(<30)'][1]:>14.3f}"
              f"{st['scatter kpc:M(50-100)|kpc:M(<30)'][1]:>15.3f}{st['scatter R50|M*'][1]:>11.3f}"
              f"{st['E/floor kpc:M(30-50)|kpc:M(<30)']:>11.2f}{st['width kpc:M(<10)']:>9.2f}"
              f"{st['median shift kpc:M(<10)']:>+9.3f}{st['median shift kpc:M(<100)']:>+10.3f}  {'keep' if ok else 'no leverage'}")
    print(f"\n  sources with leverage (widen M(30-50)|M(<30) or M(50-100)|M(<30) by >= {SWEEP_MIN_WIDEN} dex, medians within "
          f"{100 * SWEEP_MAX_SHIFT:.0f}%): {sorted(keep) or 'NONE'}")
    free = []
    if "lumpy" in keep: free.append("n_eff")
    if "efficiency" in keep: free += ["sigma_eta", "ell"]
    if "size_c" in keep: free.append("sigma_sc")
    if "size_e" in keep: free.append("sigma_se")

    # 3. the fit, held out
    def fit_on(rows, seed, start):
        d_ = dd[rows]; c_ = [cv[i] for i in rows]
        t_ = qa.measure_all(d_[:, :1], d_[:, :1], R)[0]; r_ = r50(d_[:, :1], R)[:, 0]
        names = free
        lo = np.array([P3.NOISE_BOUNDS[k][0] for k in names]); hi = np.array([P3.NOISE_BOUNDS[k][1] for k in names])

        def to_noise(v):
            base = dict(zip(P3.NOISE_NAMES, P3.Noise3.off().vector()))
            for k, x in zip(names, v):
                base[k] = float(np.clip(x, P3.NOISE_BOUNDS[k][0], P3.NOISE_BOUNDS[k][1]))
            return P3.Noise3(**base)

        def obj(v):
            pen = float(np.sum(np.clip(lo - v, 0, None) ** 2 + np.clip(v - hi, 0, None) ** 2))
            dr = P3.draw(spec2, theta, to_noise(v), c_, R, np.random.default_rng(seed), n_draw=N_DRAW_FIT, epochs=(0,))
            return objective_value(pop_stats(dr, d_, R, t_, r_)) + 100 * pen

        x0 = np.array([start[k] for k in names])
        r = minimize(obj, x0, method="Nelder-Mead",
                     options=dict(maxiter=(15 if smoke else 250), xatol=1e-3, fatol=1e-3, adaptive=True))
        return to_noise(r.x), float(r.fun), r.nit

    def score_on(rows, noise, seed):
        d_ = dd[rows]; c_ = [cv[i] for i in rows]
        t_ = qa.measure_all(d_[:, :1], d_[:, :1], R)[0]; r_ = r50(d_[:, :1], R)[:, 0]
        dr = P3.draw(spec2, theta, noise, c_, R, np.random.default_rng(seed), n_draw=N_DRAW_FIT, epochs=(0,))
        st = pop_stats(dr, d_, R, t_, r_)
        st_null = pop_stats(mean[rows][None], d_, R, t_, r_)
        return st, st_null

    results = {}
    if free:
        start = {"n_eff": 5.0, "sigma_eta": 0.2, "ell": 0.3, "sigma_sc": 0.15, "sigma_se": 0.15}
        rng = np.random.default_rng(63)
        perm = rng.permutation(n); A, B = np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
        print(f"\n{RULE}\n  HELD-OUT FIT — free: {free}; calibrate on one half ({len(A)}), score on the other\n{RULE}")
        for lab, cal, sc in (("A->B", A, B), ("B->A", B, A)):
            t0 = time.time()
            nz, val, nit = fit_on(cal, 101, start)
            st, st_null = score_on(sc, nz, 202)
            print(f"  {lab}: fitted {dict(zip(P3.NOISE_NAMES, nz.vector().round(3)))} in {nit} iterations "
                  f"({(time.time() - t0) / 60:.1f} min); objective on the calibration half {val:.2f}")
            print(f"        held-out objective {objective_value(st):.2f} (mean alone {objective_value(st_null):.2f}); "
                  f"E/floor 30-50|<30 {st['E/floor kpc:M(30-50)|kpc:M(<30)']:.2f} (null {st_null['E/floor kpc:M(30-50)|kpc:M(<30)']:.2f}), "
                  f"50-100|<30 {st['E/floor kpc:M(50-100)|kpc:M(<30)']:.2f} (null {st_null['E/floor kpc:M(50-100)|kpc:M(<30)']:.2f}), "
                  f"R50|M* {st['E/floor R50|M*']:.2f} (null {st_null['E/floor R50|M*']:.2f})")
            results[lab] = dict(noise=nz.vector(), held_out=st, null=st_null)
        t0 = time.time()
        nz_all, val_all, nit = fit_on(np.arange(n), 303, start)
        print(f"  ALL: fitted {dict(zip(P3.NOISE_NAMES, nz_all.vector().round(3)))} in {nit} iterations "
              f"({(time.time() - t0) / 60:.1f} min); objective {val_all:.2f} (null {null_obj:.2f})")
    else:
        nz_all = P3.Noise3.off()
        print("\n  no source has leverage: the process is the mean; nothing fitted")

    # 4. gates, held out (both swaps) — from the held-out stats
    print(f"\n{RULE}\n  GATES (held out, z=0.4)\n{RULE}")
    g3 = {}
    for lab, r_ in results.items():
        st = r_["held_out"]
        e_ok = all(st[f"E/floor {ky}|{kx}"] <= G3A_EFLOOR for kx, ky in KPC_PLANES)
        w_ok = all(abs(st[f"width {k}"] - 1) <= G3B_WIDTH_TOL for k in CDF_KEYS)
        g3[lab] = (e_ok, w_ok)
        print(f"  {lab}: G3a kpc-plane E/floor <= {G3A_EFLOOR}: "
              + ", ".join(f"{st[f'E/floor {ky}|{kx}']:.2f}" for kx, ky in KPC_PLANES) + f" -> {'PASS' if e_ok else 'FAIL'};  "
              f"G3b tier-2e widths within {G3B_WIDTH_TOL} of 1: "
              + ", ".join(f"{st[f'width {k}']:.2f}" for k in CDF_KEYS) + f" -> {'PASS' if w_ok else 'FAIL'}")
        stn = r_["null"]
        print(f"        mean alone: E/floor " + ", ".join(f"{stn[f'E/floor {ky}|{kx}']:.2f}" for kx, ky in KPC_PLANES)
              + "; widths " + ", ".join(f"{stn[f'width {k}']:.2f}" for k in CDF_KEYS))

    # 5. predictions with the same draws at every epoch
    print(f"\n{RULE}\n  PREDICTIONS — the same draws propagated to z=0.7-2.0 (never fitted)\n{RULE}")
    dr = P3.draw(spec2, theta, nz_all, cv, R, np.random.default_rng(404), n_draw=N_DRAW_FINAL)
    rh_truth = r50(dd, R)
    rho_t = spearmanr(np.log10(rh_truth[:, 0]), np.log10(rh_truth[:, 4]))[0]
    rho_d = []
    for s in range(N_DRAW_FINAL):
        rh = r50(dr[s], R)
        rho_d.append(spearmanr(np.log10(rh[:, 0]), np.log10(rh[:, 4]))[0])
    print(f"  tier 2c: size-rank persistence z=0.4 <-> z=2, truth {rho_t:+.2f}; draws {np.mean(rho_d):+.2f} "
          f"+- {np.std(rho_d):.2f} (exp60's layer: +0.97)")
    dec_t = float(np.mean(dd[:, 4, I3] > dd[:, 0, I3]))
    dec_d = float(np.mean(dr[:, :, 4, I3] > dr[:, :, 0, I3]))
    print(f"  declining centres (M*(<4.92) at z=2 above z=0.4): truth {100 * dec_t:.1f}%, draws {100 * dec_d:.1f}% "
          f"— zero by construction: the process is deposition-only at every draw")
    widths = np.zeros((5, len(CDF_KEYS))); wnull = np.zeros_like(widths)
    for j in range(5):
        stj = pop_stats(dr, dd, R, j=j)
        s0j = pop_stats(mean[None], dd, R, j=j)
        widths[j] = [stj[f"width {k}"] for k in CDF_KEYS]; wnull[j] = [s0j[f"width {k}"] for k in CDF_KEYS]
    print("  tier-2e width ratios (draws / truth; mean alone in brackets):")
    print(f"  {'':<8}" + "".join(f"{k:>22}" for k in CDF_KEYS))
    for j, z in enumerate(ANCHOR_Z):
        print(f"  z={z:<5}" + "".join(f"{widths[j, i]:>14.2f} [{wnull[j, i]:.2f}]" for i in range(len(CDF_KEYS))))

    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUTDIR / f"stage3_process{tag}.npz", theta=theta, noise_all=nz_all.vector(),
             noise_names=np.array(P3.NOISE_NAMES), free=np.array(free), keep=np.array(sorted(keep)),
             **{f"noise_{k}": v["noise"] for k, v in results.items()},
             widths=widths, widths_null=wnull, rho_truth=rho_t, rho_draws=np.array(rho_d),
             sweep_widen=np.array([s[3] for s in sweep]), sweep_shift=np.array([s[4] for s in sweep]),
             sweep_labels=np.array([f"{s[0]} {s[1]}" for s in sweep]))
    print(f"\n  saved {(OUTDIR / f'stage3_process{tag}.npz').relative_to(ROOT)}")
    for p_ in figures(sweep, st0, dr, mean, dd, R, truth, rh_t, rh_truth, widths, wnull, tag):
        print(f"  wrote {p_.relative_to(ROOT)}")


def figures(sweep, st0, dr, mean, dd, R, truth, rh_t, rh_truth, widths, wnull, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hongshao.plotting import set_style, save_fig
    from hongshao.qa import _tex
    set_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    paths = []
    # sweep
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    labels = [f"{s[0]} {' '.join(f'{k}={v:g}' for k, v in s[1].items())}" for s in sweep]
    x = np.arange(len(sweep))
    ax[0].bar(x, [s[2]["scatter kpc:M(30-50)|kpc:M(<30)"][1] for s in sweep], color=["#0072B2" if s[5] else "#999999" for s in sweep])
    ax[0].axhline(st0["scatter kpc:M(30-50)|kpc:M(<30)"][0], color="#E69F00", lw=2, label="truth")
    ax[0].axhline(st0["scatter kpc:M(30-50)|kpc:M(<30)"][1], color="k", ls="--", label="mean alone")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, rotation=70, fontsize=6); ax[0].set_ylabel(_tex("scatter of M(30-50)|M(<30) [dex]"))
    ax[0].set_title("leverage sweep (blue = keeps the medians within 2%)"); ax[0].legend(fontsize=8)
    ax[1].bar(x, [s[2]["scatter R50|M*"][1] for s in sweep], color=["#0072B2" if s[5] else "#999999" for s in sweep])
    ax[1].axhline(st0["scatter R50|M*"][0], color="#E69F00", lw=2, label="truth")
    ax[1].axhline(st0["scatter R50|M*"][1], color="k", ls="--", label="mean alone")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, rotation=70, fontsize=6); ax[1].set_ylabel("scatter of R50|M* [dex]"); ax[1].legend(fontsize=8)
    fig.tight_layout(); save_fig(fig, FIGDIR / f"exp63_stage3_sweep{tag}"); paths.append(FIGDIR / f"exp63_stage3_sweep{tag}.png")
    # planes at z=0.4: truth vs one draw vs mean
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    m0 = qa.measure_all(mean[:, :1], dd[:, :1], R)[1]
    m1 = qa.measure_all(dr[0][:, :1], dd[:, :1], R)[1]
    for a, (kx, ky) in zip(ax[:2], KPC_PLANES):
        a.scatter(np.log10(truth[kx][:, 0]), np.log10(truth[ky][:, 0]), s=4, alpha=0.3, color="#E69F00", label="truth")
        a.scatter(np.log10(m0[kx][:, 0]), np.log10(m0[ky][:, 0]), s=4, alpha=0.3, color="#999999", label="mean")
        a.scatter(np.log10(np.clip(m1[kx][:, 0], 1, None)), np.log10(np.clip(m1[ky][:, 0], 1, None)), s=4, alpha=0.3, color="#0072B2", label="one draw")
        a.set_xlabel(_tex(f"log {kx}")); a.set_ylabel(_tex(f"log {ky}")); a.legend(fontsize=8)
    rh_m0 = r50(mean[:, :1], R)[:, 0]; rh_m1 = r50(dr[0][:, :1], R)[:, 0]
    ax[2].scatter(np.log10(dd[:, 0, -1]), np.log10(rh_t), s=4, alpha=0.3, color="#E69F00", label="truth")
    ax[2].scatter(np.log10(mean[:, 0, -1]), np.log10(rh_m0), s=4, alpha=0.3, color="#999999", label="mean")
    ax[2].scatter(np.log10(dr[0][:, 0, -1]), np.log10(rh_m1), s=4, alpha=0.3, color="#0072B2", label="one draw")
    ax[2].set_xlabel(_tex("log M*(<148 kpc)")); ax[2].set_ylabel("log R50 [kpc]"); ax[2].legend(fontsize=8)
    fig.suptitle("z=0.4 population planes: truth, the mean model, one draw of the process", fontsize=11)
    fig.tight_layout(); save_fig(fig, FIGDIR / f"exp63_stage3_planes{tag}"); paths.append(FIGDIR / f"exp63_stage3_planes{tag}.png")
    # predictions: tier 2c and widths vs epoch
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    rh1 = r50(dr[0], R)
    ax[0].scatter(np.log10(rh_truth[:, 0]), np.log10(rh_truth[:, 4]), s=4, alpha=0.3, color="#E69F00", label="truth")
    ax[0].scatter(np.log10(rh1[:, 0]), np.log10(rh1[:, 4]), s=4, alpha=0.3, color="#0072B2", label="one draw")
    ax[0].set_xlabel("log R50 at z=0.4 [kpc]"); ax[0].set_ylabel("log R50 at z=2 [kpc]"); ax[0].legend(fontsize=8)
    ax[0].set_title("tier 2c: the same galaxy at two epochs")
    for i, k in enumerate(CDF_KEYS):
        ax[1].plot(ANCHOR_Z, widths[:, i], "o-", label=_tex(f"{k} draws"))
        ax[1].plot(ANCHOR_Z, wnull[:, i], "x--", color=ax[1].lines[-1].get_color(), alpha=0.6)
    ax[1].axhline(1, color="k", lw=0.8); ax[1].set_xlabel("z"); ax[1].set_ylabel("16-84 width, model / truth")
    ax[1].set_title("tier 2e widths (solid: draws; dashed: mean alone)"); ax[1].legend(fontsize=7)
    fig.tight_layout(); save_fig(fig, FIGDIR / f"exp63_stage3_predictions{tag}"); paths.append(FIGDIR / f"exp63_stage3_predictions{tag}.png")
    return paths


if __name__ == "__main__":
    main("--smoke" in sys.argv[1:], "--theta-default" in sys.argv[1:])
