"""exp63 Stage 1 — read the deposit-size distribution, and the size law, from
the data.

Tech note 04 Eq. 3: under the deposition-only class a curve of growth is the
cumulative of W(v) — the stellar mass per logarithm of deposit size — smoothed
by the deposit kernel. Given a kernel, W can be RECOVERED from a measured curve
by non-negative least squares. This stage does that for every galaxy at z=0.4
under three kernels (the incumbent's cored gompertz_log with c = 0.80; a cuspy
Sersic n = 4; a Sersic n = 1, exp55's compact component), then

  1a  reports how well ANY deposition-only profile with that kernel can
      represent the measured curves (per-object freedom: the class's
      single-epoch representation floor), with a permuted control;
  1b  matches the cumulative of W against the cumulative stellar mass the
      incumbent's efficiency law deposits in TIME, quantile by quantile, to
      obtain the size-versus-time law each galaxy demands, s*(t), and reports
      s*/R200c(t) against redshift for the population;
  1c  measures which halo variables have leverage on W at fixed halo mass
      (decision D2: c200c is offered to Stage 2 only if |rho| >= 0.1).

Gate G1 (frozen in the Stage 1 plan): "two size scales warranted" iff, under
the incumbent's own kernel, the median log10(s*/R200c) at z* > 3 differs from
that at z* < 0.7 by more than 0.5 dex, OR more than half the galaxies have two
separated modes in W. The pre-registered prediction is both.

The time labels (1b) come from the incumbent's efficiency law: one epoch
fixes W but not when its parts were laid down, so s*(t) is conditional on that
law. Stage 2's earlier-epoch predictions are what test the labels.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp63_analytic_growth/stage1_deconvolve.py [--smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term", HERE):
    sys.path.insert(0, str(p))

import engine as E                                       # noqa: E402
import fit as F                                          # noqa: E402
import families                                          # noqa: E402
import stage0_cost as S0                                 # noqa: E402

RULE = "=" * 92
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
KERNELS = {"gompertz_c0.80": ("gompertz_log", (0.7965,)),
           "sersic_n4": ("sersic", (4.0,)),
           "sersic_n1": ("sersic", (1.0,))}
INCUMBENT_KERNEL = "gompertz_c0.80"
S_GRID = np.geomspace(0.3, 600.0, 20)                   # deposit half-mass radii [kpc]
Q_GRID = np.arange(0.05, 0.951, 0.05)
LAMBDA_SCAN = (0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
LAMBDA_DEGRADE = 0.10                                    # <10% worse median RMS than lambda=0
MODE_MIN_SHARE, MODE_MIN_SEP_DEX = 0.10, 0.5
G1_TREND_DEX, G1_TWO_MODE_FRAC = 0.5, 0.5
LEVERAGE_MIN = 0.1
#: the incumbent's size law, for the size_law figure
INC_LOG_F0, INC_B = -0.8433, -0.8957


# --------------------------------------------------------------------------- #
# 1a — the deconvolution                                                       #
# --------------------------------------------------------------------------- #
def design(kernel, R):
    fam, shape = KERNELS[kernel]
    return families.cog_unit(fam, S_GRID, shape, R)     # (nR, K)


def second_difference(K):
    D = np.zeros((K - 2, K))
    for i in range(K - 2):
        D[i, i:i + 3] = (1.0, -2.0, 1.0)
    return D


def deconvolve(A, d, lam):
    """Non-negative w with fractional residuals and a second-difference
    penalty of strength `lam` (relative to the data rows' norm)."""
    Aw = A / d[:, None]                                  # rows scaled: (Aw w - 1)
    rows = [Aw, ]
    rhs = [np.ones_like(d)]
    if lam > 0:
        K = A.shape[1]
        D = second_difference(K) * lam * np.linalg.norm(Aw) / np.sqrt(K)
        rows.append(D); rhs.append(np.zeros(K - 2))
    w, _ = nnls(np.vstack(rows), np.concatenate(rhs), maxiter=50 * A.shape[1])
    return w


def rms_dex(A, w, d):
    m = A @ w
    return float(np.sqrt(np.mean((np.log10(np.clip(m, 1.0, None)) - np.log10(d)) ** 2)))


def count_modes(w):
    """Separated modes of w on the size grid: local maxima of a 3-bin running
    mean, each holding >= MODE_MIN_SHARE of the total, >= MODE_MIN_SEP_DEX apart."""
    tot = w.sum()
    if tot <= 0:
        return 0
    sm = np.convolve(w, np.ones(3) / 3.0, mode="same")
    peaks = [i for i in range(len(sm))
             if sm[i] >= sm[max(i - 1, 0)] and sm[i] >= sm[min(i + 1, len(sm) - 1)]
             and sm[i] > 0]
    # share of each peak: mass in the bins closer to it than to any other peak
    kept = []
    for i in sorted(peaks, key=lambda j: -sm[j]):
        if all(abs(np.log10(S_GRID[i] / S_GRID[j])) >= MODE_MIN_SEP_DEX for j in kept):
            kept.append(i)
    if len(kept) < 2:
        return len(kept)
    kept = np.array(sorted(kept))
    owner = np.argmin(np.abs(np.log10(S_GRID)[:, None] - np.log10(S_GRID[kept])[None, :]), axis=1)
    shares = np.array([w[owner == m].sum() / tot for m in range(len(kept))])
    return int((shares >= MODE_MIN_SHARE).sum())


# --------------------------------------------------------------------------- #
# 1b — quantile matching                                                       #
# --------------------------------------------------------------------------- #
def cumulative_in_time(spec, theta, curves):
    """(n, N) normalised cumulative stellar mass deposited by the incumbent's
    efficiency law up to each node, and the nodes' z and analytic R200c."""
    lt, w, include, _ = E.nodes()
    lm, z, r200, dmstar, _, _, _ = E._deposits(spec, theta, curves, lt, "analytic")
    wd = dmstar * w[None, :] * include[0][None, :]
    ct = np.cumsum(wd, axis=1)
    return ct / ct[:, -1:], z[0], r200, lt


def size_law_from_w(w, ct, z_nodes, r200_nodes):
    """log10(s*/R200c) and z* at Q_GRID for one galaxy."""
    cw = np.cumsum(w); cw = cw / cw[-1]
    # cumulative attached to the upper edge of each bin; interpolate in ln s
    ls = np.log(S_GRID)
    edges = np.r_[ls[0] - 0.5 * (ls[1] - ls[0]), 0.5 * (ls[1:] + ls[:-1]), ls[-1] + 0.5 * (ls[1] - ls[0])]
    ls_star = np.interp(Q_GRID, np.r_[0.0, cw], edges)
    j = np.searchsorted(ct, Q_GRID)
    j = np.clip(j, 1, len(ct) - 1)
    frac = (Q_GRID - ct[j - 1]) / np.clip(ct[j] - ct[j - 1], 1e-30, None)
    z_star = z_nodes[j - 1] + frac * (z_nodes[j] - z_nodes[j - 1])
    lr = np.log10(r200_nodes[j - 1]) + frac * (np.log10(r200_nodes[j]) - np.log10(r200_nodes[j - 1]))
    return ls_star / np.log(10.0) - lr, z_star


# --------------------------------------------------------------------------- #
# 1c — leverage                                                                #
# --------------------------------------------------------------------------- #
def halo_variables(recs, curves):
    pop = np.load(POP_NPZ)
    rows = np.array([hc.row for hc in curves])
    out = {"early": np.array([hc.early for hc in curves]),
           "late": np.array([hc.late for hc in curves]),
           "logtc": np.array([hc.logtc for hc in curves]),
           "t50": pop["t50"][rows], "c200c": pop["c200c"][rows],
           "f_form": np.array([h.f_form[h.epoch_mask[0]][-1] for h in recs])}
    return pop["logmh"][rows], out


def partial_spearman(y, x, lmh):
    ok = np.isfinite(y) & np.isfinite(x) & np.isfinite(lmh)
    X = np.column_stack([np.ones(ok.sum()), lmh[ok], lmh[ok] ** 2])
    ry = y[ok] - X @ np.linalg.lstsq(X, y[ok], rcond=None)[0]
    rx = x[ok] - X @ np.linalg.lstsq(X, x[ok], rcond=None)[0]
    return float(spearmanr(ry, rx)[0])


# --------------------------------------------------------------------------- #
# figures                                                                      #
# --------------------------------------------------------------------------- #
def figures(R, data, res, lmh, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hongshao.plotting import set_style, save_fig
    from hongshao.qa import _tex, _pct
    set_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    ks = list(KERNELS)
    kc = {"gompertz_c0.80": "#0072B2", "sersic_n4": "#D55E00", "sersic_n1": "#009E73"}
    paths = []

    # representation: RMS histograms + three example galaxies under the incumbent kernel
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.4))
    for k in ks:
        ax[0].hist(res[k]["rms"], bins=np.linspace(0, 0.06, 40), histtype="step", lw=1.8,
                   color=kc[k], label=f"{k}: median {np.median(res[k]['rms']):.4f} dex")
        ax[0].hist(res[k]["rms_perm"], bins=np.linspace(0, 0.06, 40), histtype="step", lw=1.0,
                   ls="--", color=kc[k])
    ax[0].axvline(0.0044, color="k", ls=":", label="exp55 4-parameter form: 0.0044")
    ax[0].set_xlabel("CoG RMS of the deconvolution [dex] (dashed: permuted control)")
    ax[0].set_ylabel("galaxies"); ax[0].legend(fontsize=6.5); ax[0].set_title("(a) representation floor per kernel")
    k = INCUMBENT_KERNEL
    order = np.argsort(res[k]["rms"])
    for a, (lab, i) in zip(ax[1:], (("best", order[0]), ("typical", order[len(order) // 2]),
                                    ("worst", order[-1]))):
        A = design(k, R)
        a.loglog(R, data[i], "k-", lw=2, label="data")
        a.loglog(R, A @ res[k]["w"][i], "-", color=kc[k], lw=1.5, label=f"fit, RMS {res[k]['rms'][i]:.4f} dex")
        w = res[k]["w"][i]
        a2 = a.twinx()
        a2.bar(S_GRID, w / w.sum(), width=0.25 * S_GRID, color=kc[k], alpha=0.25)
        a2.set_ylim(0, 1); a2.set_ylabel("share of stellar mass per size bin", fontsize=7)
        a.set_xlabel(_tex("R or deposit size s [kpc]")); a.set_ylabel(_tex("M*(<R) [Msun]"))
        a.set_title(f"({'bcd'[('best', 'typical', 'worst').index(lab)]}) {lab}: idx {res['index'][i]}, {k}")
        a.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); save_fig(fig, FIGDIR / f"exp63_stage1_representation{tag}")
    paths.append(FIGDIR / f"exp63_stage1_representation{tag}.png")

    # W population: median share per bin, halo-mass terciles, per kernel
    edges = np.percentile(lmh, [0, 100 / 3, 200 / 3, 100])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for a, k in zip(ax, ks):
        for c, ls in zip(range(3), ("-", "--", ":")):
            sel = (lmh >= edges[c]) & (lmh <= edges[c + 1])
            share = res[k]["w"][sel] / res[k]["w"][sel].sum(1, keepdims=True)
            a.plot(S_GRID, np.median(share, 0), ls, color=kc[k], lw=1.8,
                   label=f"logMh {edges[c]:.2f}-{edges[c + 1]:.2f}")
            a.fill_between(S_GRID, np.percentile(share, 16, 0), np.percentile(share, 84, 0),
                           color=kc[k], alpha=0.08)
        a.set_xscale("log"); a.set_xlabel("deposit half-mass radius s [kpc]")
        a.set_title(f"{k}: {100 * np.mean(res[k]['n_modes'] >= 2):.0f}{_pct()} of galaxies have two modes")
        a.legend(fontsize=7)
    ax[0].set_ylabel("median share of stellar mass per size bin (16-84 band)")
    fig.suptitle("the deposit-size distribution W the data require, by halo mass", fontsize=11)
    fig.tight_layout(); save_fig(fig, FIGDIR / f"exp63_stage1_w_population{tag}")
    paths.append(FIGDIR / f"exp63_stage1_w_population{tag}.png")

    # size law: log(s*/R200c) vs z*
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    zz = np.linspace(0.4, 8, 100)
    for a, k in zip(ax, ks):
        ls_, zs_ = res[k]["log_s_over_r200"], res["z_star"]
        zb = np.array([0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0, 15.0])
        med, lo, hi, zc = [], [], [], []
        for a_, b_ in zip(zb[:-1], zb[1:]):
            sel = (zs_ >= a_) & (zs_ < b_) & np.isfinite(ls_)
            if sel.sum() < 20:
                continue
            med.append(np.median(ls_[sel])); lo.append(np.percentile(ls_[sel], 16))
            hi.append(np.percentile(ls_[sel], 84)); zc.append(np.median(zs_[sel]))
        a.plot(zc, med, "o-", color=kc[k], lw=2, label="required: median, 16-84 band")
        a.fill_between(zc, lo, hi, color=kc[k], alpha=0.2)
        a.plot(zz, INC_LOG_F0 + INC_B * np.log10(1 + zz), "k--", lw=1.2,
               label=r"incumbent: $10^{-0.84}(1+z)^{-0.90}$")
        a.axhline(np.log10(0.015), color="gray", ls=":", label=r"$R_{1/2}=0.015\,R_{200c}$ (Kravtsov 2013)")
        a.set_xscale("log"); a.set_xlabel(r"redshift of the deposit, $z^\ast$")
        a.set_title(k); a.legend(fontsize=7, loc="lower left")
    ax[0].set_ylabel(r"$\log_{10}\,[\,s^\ast/R_{200c}(t^\ast)\,]$")
    fig.suptitle("the size-versus-time law the data demand (quantile-matched to the incumbent's deposition in time)", fontsize=11)
    fig.tight_layout(); save_fig(fig, FIGDIR / f"exp63_stage1_size_law{tag}")
    paths.append(FIGDIR / f"exp63_stage1_size_law{tag}.png")

    # leverage
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for a, k in zip(ax, ks):
        L = res[k]["leverage"]                           # dict stat -> dict var -> rho
        vars_ = list(next(iter(L.values())))
        x = np.arange(len(vars_))
        for i, (stat, rs) in enumerate(L.items()):
            a.bar(x + 0.27 * (i - 1), [rs[v] for v in vars_], width=0.25, label=_tex(stat))
        a.axhline(LEVERAGE_MIN, color="k", ls=":"); a.axhline(-LEVERAGE_MIN, color="k", ls=":")
        a.set_xticks(x); a.set_xticklabels(vars_, rotation=30); a.set_title(k)
        a.legend(fontsize=7)
    ax[0].set_ylabel(r"partial Spearman $\rho$ at fixed $\log M_h$")
    fig.suptitle("which halo variables have leverage on the required W (dotted: the 0.1 rule)", fontsize=11)
    fig.tight_layout(); save_fig(fig, FIGDIR / f"exp63_stage1_leverage{tag}")
    paths.append(FIGDIR / f"exp63_stage1_leverage{tag}.png")
    return paths


# --------------------------------------------------------------------------- #
def main(smoke=False):
    tag = "_smoke" if smoke else ""
    print(f"{RULE}\nexp63 STAGE 1 — the deposit-size distribution and the size law, read from "
          f"the data{' (SMOKE)' if smoke else ''}\n{RULE}\n")
    recs, data_all, mask, lmh_all, spec, theta, _ = S0.build(smoke)
    R = F.R_GRID
    good = np.isfinite(data_all[:, 0]).all(1) & (data_all[:, 0] > 0).all(1)
    recs = [h for h, g in zip(recs, good) if g]
    data = data_all[good, 0]                              # z=0.4 curves
    curves = E.build_curves(recs, verbose=False)
    lmh, hv = halo_variables(recs, curves)
    n = len(recs)
    print(f"  {n} galaxies with finite positive z=0.4 curves; kernels {list(KERNELS)}; "
          f"size grid {S_GRID[0]:.2f}-{S_GRID[-1]:.0f} kpc, {len(S_GRID)} bins\n")

    # the penalty, chosen on the incumbent kernel and frozen
    A0 = design(INCUMBENT_KERNEL, R)
    sub = np.arange(0, n, max(1, n // 100))
    med0 = None
    lam_use = 0.0
    print("  penalty scan (incumbent kernel, every ~100th galaxy): lambda -> median RMS [dex]")
    for lam in LAMBDA_SCAN:
        r = np.array([rms_dex(A0, deconvolve(A0, data[i], lam), data[i]) for i in sub])
        m = float(np.median(r))
        med0 = m if med0 is None else med0
        ok = m <= med0 * (1 + LAMBDA_DEGRADE)
        print(f"    {lam:<6g} {m:.5f} {'ok' if ok else 'too smooth'}")
        if ok:
            lam_use = lam
    print(f"  frozen: lambda = {lam_use:g}\n")

    ct, z_nodes, r200_nodes, _ = cumulative_in_time(spec, theta, curves)
    rng = np.random.default_rng(63)
    perm = rng.permutation(n)
    res = {"index": np.array([hc.index for hc in curves]), "lambda": lam_use}
    for k in KERNELS:
        A = design(k, R)
        w = np.array([deconvolve(A, data[i], lam_use) for i in range(n)])
        rms = np.array([rms_dex(A, w[i], data[i]) for i in range(n)])
        # permuted control: galaxy i's curve against galaxy perm[i]'s fit, amplitude-matched at 100 kpc
        rms_perm = np.array([rms_dex(A, w[perm[i]] * data[i, F.I100] / (A @ w[perm[i]])[F.I100], data[i])
                             for i in range(n)])
        n_modes = np.array([count_modes(w[i]) for i in range(n)])
        ls_r200 = np.full((n, len(Q_GRID)), np.nan); z_star = np.full((n, len(Q_GRID)), np.nan)
        for i in range(n):
            if w[i].sum() > 0:
                ls_r200[i], z_star[i] = size_law_from_w(w[i], ct[i], z_nodes, r200_nodes[i])
        lw = np.log10(S_GRID)
        stats = {"mean log s": (w * lw).sum(1) / np.clip(w.sum(1), 1e-30, None),
                 "inner share s<5": w[:, S_GRID < 5].sum(1) / np.clip(w.sum(1), 1e-30, None),
                 "outer share s>50": w[:, S_GRID > 50].sum(1) / np.clip(w.sum(1), 1e-30, None)}
        lev = {s: {v: partial_spearman(y, x, lmh) for v, x in hv.items()} for s, y in stats.items()}
        res[k] = dict(w=w, rms=rms, rms_perm=rms_perm, n_modes=n_modes,
                      log_s_over_r200=ls_r200, leverage=lev, stats=stats)
        res["z_star"] = z_star

        print(f"{RULE}\n  kernel {k}\n{RULE}")
        print(f"  1a representation: CoG RMS median {np.median(rms):.4f} dex, 16-84% "
              f"{np.percentile(rms, 16):.4f}-{np.percentile(rms, 84):.4f}; permuted control "
              f"median {np.median(rms_perm):.4f} dex  (exp55's 4-parameter analytic form: 0.0044; PCA-3: 0.0034)")
        print(f"     two or more separated modes in W: {100 * np.mean(n_modes >= 2):.1f}% of galaxies "
              f"(one mode {100 * np.mean(n_modes == 1):.1f}%)")
        early = (z_star > 3) & np.isfinite(ls_r200); late = (z_star < 0.7) & np.isfinite(ls_r200)
        m_e, m_l = np.median(ls_r200[early]), np.median(ls_r200[late])
        print(f"  1b size law: median log10(s*/R200c) at z*>3: {m_e:+.2f} (16-84% "
              f"{np.percentile(ls_r200[early], 16):+.2f}..{np.percentile(ls_r200[early], 84):+.2f}), "
              f"at z*<0.7: {m_l:+.2f} ({np.percentile(ls_r200[late], 16):+.2f}..{np.percentile(ls_r200[late], 84):+.2f}); "
              f"trend late-early {m_l - m_e:+.2f} dex; incumbent's law at z=4: {INC_LOG_F0 + INC_B * np.log10(5):+.2f}, at z=0.5: {INC_LOG_F0 + INC_B * np.log10(1.5):+.2f}")
        res[k]["trend"] = m_l - m_e
        print("  1c leverage, partial Spearman at fixed log Mh:")
        for s, rs in lev.items():
            print(f"     {s:<18}" + "  ".join(f"{v} {r:+.2f}" for v, r in rs.items()))

    k = INCUMBENT_KERNEL
    two = float(np.mean(res[k]["n_modes"] >= 2)); tr = float(res[k]["trend"])
    warranted = (abs(tr) > G1_TREND_DEX) or (two > G1_TWO_MODE_FRAC)
    print(f"\n{RULE}\n  GATE G1 (incumbent kernel): trend {tr:+.2f} dex (rule: |trend| > {G1_TREND_DEX}), "
          f"two-mode fraction {100 * two:.0f}% (rule: > {100 * G1_TWO_MODE_FRAC:.0f}%) -> "
          f"{'TWO SCALES WARRANTED' if warranted else 'one scale suffices'}\n{RULE}")
    offered = sorted({v for kk in KERNELS for s in ("mean log s", "inner share s<5")
                      for v, r in res[kk]["leverage"][s].items() if abs(r) >= LEVERAGE_MIN})
    print(f"  D2/1c: variables with |rho| >= {LEVERAGE_MIN} on the mean size or the inner share under "
          f"any kernel: {offered}; c200c {'OFFERED' if 'c200c' in offered else 'NOT offered'} to Stage 2")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUTDIR / f"stage1_deconvolve{tag}.npz", s_grid=S_GRID, q_grid=Q_GRID,
             index=res["index"], z_star=res["z_star"], lam=lam_use, lmh=lmh,
             g1_warranted=warranted, g1_trend=tr, g1_two_mode=two, offered=np.array(offered),
             **{f"{key}_{k}": res[k][key] for k in KERNELS
                for key in ("w", "rms", "rms_perm", "n_modes", "log_s_over_r200")},
             **{f"leverage_{k}": np.array([[res[k]["leverage"][s][v] for v in hv] for s in res[k]["leverage"]])
                for k in KERNELS},
             leverage_vars=np.array(list(hv)), leverage_stats=np.array(list(res[k]["leverage"])))
    print(f"  saved {(OUTDIR / f'stage1_deconvolve{tag}.npz').relative_to(ROOT)}")
    for p in figures(R, data, res, lmh, tag):
        print(f"  wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main("--smoke" in sys.argv[1:])
