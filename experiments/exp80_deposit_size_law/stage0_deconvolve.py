"""exp80 Stage 0, part A/B — NO FIT. What deposit sizes does the data demand at
EVERY epoch, and what sizes does the adopted baseline lay down?

exp63 Stage 1 read the deposit-size distribution W(s) — the stellar mass per
logarithm of deposit half-mass radius s — from each z = 0.4 curve of growth by
non-negative least squares under a fixed deposit kernel. This stage runs that
same operator on every epoch's curves (the fitting sample, on exp73's merged
0.673–148 kpc grid so the compact mode is resolved below 2 kpc), and, so that
the comparison is like for like, on the BASELINE MODEL's own predicted curves
at the same epochs under the same kernels. It also reads the baseline's
deposit sizes analytically from `model2._deposits2` (the size law in its own
units: s_c in kpc, s_e as a fraction of R200c at the deposit time).

For each epoch, kernel and halo-mass tercile (the binned term's terciles, by
the measured halo mass at that epoch) the table gives:
  compact size s_c    the mass-weighted mean log s below the split
  extended size s_e   the mass-weighted mean log s above the split
  extended share      the stellar mass above the split
  s_e / R200c(z_k)    the extended size against the halo's radius AT THAT EPOCH
for the data and for the model (deconvolved the same way), with the tercile
median, its bootstrap standard error (the "tercile scatter of the
deconvolution" the plan's gate refers to) and the 16–84 per cent spread; and
the model's analytic deposit sizes and share. The split between the modes is
read per epoch and kernel from the DATA's stacked W (the minimum between its
two largest modes) and applied unchanged to the model.

Item 4 of the plan: the slope of each mode's size against the galaxy's
stellar mass at z = 2 (data vs model), next to the R50–M* slope the size
gate reads.

Nothing here touches the truncation boundary (exp79's) or fits anything.

Run:
    HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 PYTHONPATH=. \\
        nohup uv run python -u experiments/exp80_deposit_size_law/stage0_deconvolve.py [--smoke] \\
        > experiments/exp80_deposit_size_law/outputs/stage0_deconvolve.log 2>&1 &
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term",
          ROOT / "experiments/exp63_analytic_growth",
          ROOT / "experiments/exp73_size_relative",
          ROOT / "experiments/exp74_c19_history_leak", HERE):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import engine as E                                       # noqa: E402
import model2 as M2                                      # noqa: E402
import families                                          # noqa: E402
import selection as SEL                                  # noqa: E402
import coordinate as C                                   # noqa: E402


def _by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# exp54 shadows `rebaseline`, exp76/exp78 shadow `stage0_terms`: import by path
S0T = _by_path("exp78_stage0_terms", ROOT / "experiments/exp78_size_aware_objective/stage0_terms.py")
RB = S0T.RB

RULE = "=" * 110
THIN = "-" * 110
ANCHOR_Z = list(E.ANCHOR_Z)
EPOCHS = (0, 1, 2, 3, 4)
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
SEL_NPZ = ROOT / "experiments/exp54_unpinned_amplitude/outputs/selection.npz"
S_GRID = np.geomspace(0.3, 600.0, 20)                    # deposit half-mass radii [kpc], exp63's grid
LOG_S = np.log10(S_GRID)
N_BOOT = 200
MODE_MIN_SHARE = 0.10                                    # a mode must hold a tenth of the stacked mass
TERCILE_LABELS = ("low", "mid", "high")
#: exp63 Stage 1's z = 0.4 reading on the standard grid, reproduced as a check
EXP63_STAGE1 = {"compact_kpc": (6.2, 6.0, 6.3), "extended_kpc": (43.0, 53.0, 74.0),
                "share_ge15": (0.35, 0.45, 0.57)}


# --------------------------------------------------------------------------- #
# the operator (exp63 Stage 1, lambda = 0 frozen there)                        #
# --------------------------------------------------------------------------- #
def kernels_of(spec2, theta):
    """The three kernels: the model's extended kernel, the model's compact
    kernel, and exp63's Sersic n = 1 reference."""
    p = spec2.unpack(theta)
    return {"extended kernel": (spec2.extended_family, (float(p["c_e"]),)),
            "compact kernel": (M2.COMPACT_FAMILY, (float(p["n_c"]),)),
            "sersic n=1": ("sersic", (1.0,))}


def design(kernel, R):
    fam, shape = kernel
    return families.cog_unit(fam, S_GRID, shape, R)     # (nR, K)


def deconvolve(A, d):
    """Non-negative W with fractional residuals (rows scaled by the data)."""
    Aw = A / d[:, None]
    w, _ = nnls(Aw, np.ones_like(d), maxiter=50 * A.shape[1])
    return w


def deconvolve_many(A, curves):
    """(n, K) W for (n, nR) curves; rows with a non-finite or non-positive
    curve get NaN."""
    n = len(curves)
    W = np.full((n, len(S_GRID)), np.nan)
    rms = np.full(n, np.nan)
    for i in range(n):
        d = curves[i]
        if not (np.isfinite(d).all() and (d > 0).all()):
            continue
        W[i] = deconvolve(A, d)
        m = A @ W[i]
        rms[i] = np.sqrt(np.mean((np.log10(np.clip(m, 1.0, None)) - np.log10(d)) ** 2))
    return W, rms


# --------------------------------------------------------------------------- #
# reading the modes                                                            #
# --------------------------------------------------------------------------- #
def stack(W):
    """The population's deposit-size distribution: the mean over galaxies of
    each galaxy's share per bin (NaN rows ignored)."""
    ok = np.isfinite(W).all(1) & (np.nansum(W, 1) > 0)
    sh = W[ok] / W[ok].sum(1, keepdims=True)
    return sh.mean(0), int(ok.sum())


def modes_of(st):
    """Two largest separated modes of a stacked share vector on S_GRID: the
    peaks of the 3-bin running mean, each owning the bins nearer to it in log
    s than to any other peak, ranked by the mass they own. Returns dict with
    the peak sizes, owned shares, the split (the bin minimum between the two
    largest modes) and the gap depth (stack at the split / lower peak)."""
    sm = np.convolve(st, np.ones(3) / 3.0, mode="same")
    K = len(st)
    peaks = [i for i in range(K) if sm[i] > 0 and sm[i] >= sm[max(i - 1, 0)] and sm[i] >= sm[min(i + 1, K - 1)]]
    if not peaks:
        return dict(n_modes=0, peaks_kpc=(), shares=(), split_kpc=np.nan, gap_depth=np.nan)
    owner = np.argmin(np.abs(LOG_S[:, None] - LOG_S[peaks][None, :]), axis=1)
    shares = np.array([st[owner == m].sum() for m in range(len(peaks))])
    order = np.argsort(-shares)
    big = [peaks[j] for j in order if shares[j] >= MODE_MIN_SHARE][:2]
    if len(big) < 2:
        j = order[0]
        return dict(n_modes=1, peaks_kpc=(float(S_GRID[peaks[j]]),), shares=(float(shares[j]),),
                    split_kpc=np.nan, gap_depth=np.nan)
    lo, hi = sorted(big)
    j_split = lo + int(np.argmin(sm[lo:hi + 1]))
    # the split at the bin EDGE below the minimum bin's upper neighbour: use the
    # geometric mean of the minimum bin and the next, so the minimum bin's mass
    # (almost nothing) goes with the compact side
    split = float(np.sqrt(S_GRID[j_split] * S_GRID[min(j_split + 1, K - 1)]))
    depth = float(sm[j_split] / min(sm[lo], sm[hi]))
    return dict(n_modes=2, peaks_kpc=(float(S_GRID[lo]), float(S_GRID[hi])),
                shares=(float(st[LOG_S < np.log10(split)].sum()), float(st[LOG_S >= np.log10(split)].sum())),
                split_kpc=split, gap_depth=depth)


def per_galaxy(W, split_kpc):
    """(log s_c, log s_e, extended share) per galaxy for a split; NaN where a
    side holds no mass."""
    below = LOG_S < np.log10(split_kpc)
    tot = np.nansum(W, 1)
    mc, me = np.nansum(W[:, below], 1), np.nansum(W[:, ~below], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ls_c = np.where(mc > 0, (W[:, below] * LOG_S[below]).sum(1) / mc, np.nan)
        ls_e = np.where(me > 0, (W[:, ~below] * LOG_S[~below]).sum(1) / me, np.nan)
        share_e = np.where(tot > 0, me / tot, np.nan)
    ls_c[~np.isfinite(W).all(1)] = np.nan
    ls_e[~np.isfinite(W).all(1)] = np.nan
    return ls_c, ls_e, share_e


def med_se(x, rng, n_boot=N_BOOT):
    """(median, bootstrap s.e. of the median, p16, p84, n) of the finite values."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan, np.nan, np.nan, np.nan, len(x)
    meds = np.array([np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)])
    return float(np.median(x)), float(np.std(meds)), float(np.percentile(x, 16)), float(np.percentile(x, 84)), len(x)


def slope(y, x, min_n=30):
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() >= min_n else np.nan


# --------------------------------------------------------------------------- #
# the model's analytic deposits                                                #
# --------------------------------------------------------------------------- #
def analytic_deposits(spec2, theta, curves, block=300):
    """The model's OWN deposit-size distribution, kernel-free: per galaxy and
    epoch, the stellar mass deposited (by both channels, up to that epoch) in
    each S_GRID bin of deposit half-mass radius — `W_ana` (n, 5, K), the
    quantity the deconvolution estimates. Also the per-CHANNEL summaries: the
    mass-weighted mean log s_c [kpc], log s_e [kpc], log(s_e / R200c(t')),
    the extended channel's share, and log R200c(z_k) [kpc] of the halo at the
    epoch. Arrays (n, 5)."""
    lt, w, include, _ = E.nodes(**M2.FULL_NODES)
    n, K = len(curves), len(S_GRID)
    edges = np.r_[LOG_S[0] - 0.5 * (LOG_S[1] - LOG_S[0]), 0.5 * (LOG_S[1:] + LOG_S[:-1]), LOG_S[-1] + 0.5 * (LOG_S[1] - LOG_S[0])]
    out = {k: np.full((n, 5), np.nan) for k in ("ls_c", "ls_e", "ls_e_r200", "share_e", "lr200_k")}
    W_ana = np.zeros((n, 5, K))
    lt_k = np.log10(E.T_ANCHOR)
    for lo in range(0, n, block):
        cv = curves[lo:lo + block]
        dm_c, dm_e, s_c, s_e, _ = M2._deposits2(spec2, theta, cv, lt)
        r200 = np.array([E.r200c_of(hc, lt) for hc in cv])
        ls_c, ls_e, lse_r = np.log10(s_c), np.log10(s_e), np.log10(s_e / r200)
        j_c = np.clip(np.digitize(ls_c, edges) - 1, 0, K - 1)
        j_e = np.clip(np.digitize(ls_e, edges) - 1, 0, K - 1)
        gi = np.arange(len(cv))[:, None] * np.ones_like(j_c)
        for k in EPOCHS:
            wk = (w * include[k])[None, :]
            wc, we = dm_c * wk, dm_e * wk
            Wk = np.zeros((len(cv), K))
            np.add.at(Wk, (gi.ravel(), j_c.ravel()), wc.ravel())
            np.add.at(Wk, (gi.ravel(), j_e.ravel()), we.ravel())
            W_ana[lo:lo + len(cv), k] = Wk
            sc, se = wc.sum(1), we.sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                out["ls_c"][lo:lo + len(cv), k] = np.where(sc > 0, (wc * ls_c).sum(1) / sc, np.nan)
                out["ls_e"][lo:lo + len(cv), k] = np.where(se > 0, (we * ls_e).sum(1) / se, np.nan)
                out["ls_e_r200"][lo:lo + len(cv), k] = np.where(se > 0, (we * lse_r).sum(1) / se, np.nan)
                out["share_e"][lo:lo + len(cv), k] = np.where(sc + se > 0, se / (sc + se), np.nan)
        for i, hc in enumerate(cv):
            out["lr200_k"][lo + i] = np.log10(E.r200c_of(hc, lt_k))
    out["W_ana"] = W_ana
    return out


# --------------------------------------------------------------------------- #
def exp63_check(data, rows0, lmh0):
    """exp63 Stage 1's z = 0.4 reading on the STANDARD grid under the incumbent
    kernel (gompertz c = 0.7965), with its statistics (median share per bin,
    share at s >= 15 kpc), so the generalised operator is seen to reproduce it."""
    A = families.cog_unit("gompertz_log", S_GRID, (0.7965,), F.R_GRID)
    W, _ = deconvolve_many(A, data[rows0, 0])
    lm = lmh0[rows0]
    e = np.quantile(lm, [0, 1 / 3, 2 / 3, 1])
    print(f"\n  CHECK against exp63 Stage 1 (standard grid, incumbent kernel, z = 0.4, {len(rows0)} galaxies; "
          f"exp63 had 2397 = every galaxy with a finite curve):")
    print(f"    {'tercile':<14}{'n':>6}{'compact peak':>14}{'extended peak':>15}{'share s>=15':>13}   exp63: "
          f"{'compact':>8}{'extended':>10}{'share':>7}")
    for b in range(3):
        sel = (lm >= e[b]) & (lm <= e[b + 1] + 1e-9)
        ok = sel & np.isfinite(W).all(1) & (np.nansum(W, 1) > 0)
        sh = W[ok] / W[ok].sum(1, keepdims=True)
        med = np.median(sh, 0)
        below, above = LOG_S < np.log10(15.0), LOG_S >= np.log10(15.0)
        pc = S_GRID[below][np.argmax(med[below])]; pe = S_GRID[above][np.argmax(med[above])]
        share15 = float(np.median(sh[:, above].sum(1)))
        print(f"    {TERCILE_LABELS[b]:<14}{int(ok.sum()):>6}{pc:>14.1f}{pe:>15.1f}{share15:>13.2f}   "
              f"       {EXP63_STAGE1['compact_kpc'][b]:>8.1f}{EXP63_STAGE1['extended_kpc'][b]:>10.1f}"
              f"{EXP63_STAGE1['share_ge15'][b]:>7.2f}")


def main(smoke=False):
    tag = "_smoke" if smoke else ""
    t0 = time.time()
    print(f"{RULE}\nexp80 Stage 0 A/B — the deposit sizes the data demand at every epoch vs the baseline's; NO FIT"
          f"{' (SMOKE)' if smoke else ''}\n{RULE}\n")
    recs, data, mask, lmh_dm, lmh_bins, meas, fz, spec2, th_inc, pr, Rm, truth_m, n_inner, rows_m = S0T.build(smoke)
    spec_b, th_b = RB.adopted_baseline()
    assert spec_b.theta_names == spec2.theta_names
    p = spec2.unpack(th_b)
    print(f"\n  THE BASELINE MEAN (exp74's measured optimum, {spec2.n_theta} parameters): compact channel Sersic "
          f"n_c = {p['n_c']:.3f}, s_c = 10^{p['log_f_c']:.3f} (1+z')^{p['b_c']:+.3f} kpc "
          f"= {10 ** p['log_f_c']:.1f} kpc at z' = 0, {10 ** p['log_f_c'] * 3 ** p['b_c']:.1f} kpc at z' = 2, "
          f"{10 ** p['log_f_c'] * 6 ** p['b_c']:.1f} kpc at z' = 5;\n    extended channel {spec2.extended_family} "
          f"c_e = {p['c_e']:.3f}, s_e = 10^{p['log_f_e']:.3f} (1+z')^{p['b_e']:+.3f} R200c(t') = "
          f"{10 ** p['log_f_e']:.3f} R200c at z' = 0, {10 ** p['log_f_e'] * 3 ** p['b_e']:.3f} at z' = 2, "
          f"{10 ** p['log_f_e'] * 6 ** p['b_e']:.3f} at z' = 5; split m_half {p['m_half']:.2f}, d_split {p['d_split']:.2f}")
    kernels = kernels_of(spec2, th_b)
    print(f"  kernels for the deconvolution: " + "; ".join(f"{k}: {v[0]} {v[1]}" for k, v in kernels.items()))
    print(f"  size grid {S_GRID[0]:.2f}-{S_GRID[-1]:.0f} kpc, {len(S_GRID)} bins ({LOG_S[1] - LOG_S[0]:.3f} dex); "
          f"merged radius grid {Rm[0]:.3f}-{Rm[-1]:.1f} kpc, {len(Rm)} radii; NNLS with fractional residuals, no penalty (exp63)")

    # the model on the merged grid, every fitting row
    index_of = np.full(len(recs), -1)
    index_of[pr.all_rows] = np.arange(len(pr.all_rows))
    cv = [meas[i] for i in pr.all_rows]
    t1 = time.time()
    pred_m = M2.predict2(spec2, th_b, cv, Rm, epochs=EPOCHS, nodes=M2.FULL_NODES)
    ana = analytic_deposits(spec2, th_b, cv)
    print(f"  baseline predicted on the merged grid and its deposits read analytically: {time.time() - t1:.0f} s")

    # the samples
    sn = np.load(SEL_NPZ, allow_pickle=True)
    hs = np.load(SEL.HS_NPZ, allow_pickle=True)
    lmh_cat = SEL.sample_masses(hs)[np.array([h.row for h in recs])]
    complete = np.isfinite(lmh_cat) & (lmh_cat >= sn["cuts"][None, :])
    print(f"  sample per epoch (fitting rows with a usable merged truth): "
          + "/".join(str(len(rows_m[k])) for k in EPOCHS) + "; mh-complete among them: "
          + "/".join(str(int(complete[rows_m[k], k].sum())) for k in EPOCHS))
    exp63_check(data, pr.rows[0], lmh_bins[:, 0])

    rng = np.random.default_rng(80)
    res = {}                                             # (kernel, k) -> dict
    logms = np.log10(np.clip(data[:, :, -1], 1.0, None))  # M*(<148) per epoch, the size gate's x
    for kname, kern in kernels.items():
        A = design(kern, Rm)
        print(f"\n{RULE}\n  KERNEL: {kname} — {kern[0]} {kern[1]}\n{RULE}")
        for k in EPOCHS:
            rows = rows_m[k]
            W_d, rms_d = deconvolve_many(A, truth_m[rows, k])
            W_m, rms_m = deconvolve_many(A, pred_m[index_of[rows], k])
            st_d, n_d = stack(W_d)
            st_m, n_m = stack(W_m)
            md, mm = modes_of(st_d), modes_of(st_m)
            split = md["split_kpc"] if md["n_modes"] == 2 else (mm["split_kpc"] if mm["n_modes"] == 2 else 15.0)
            W_a = ana["W_ana"][index_of[rows], k]
            g_d, g_m, g_a = per_galaxy(W_d, split), per_galaxy(W_m, split), per_galaxy(W_a, split)
            st_a, _ = stack(W_a)
            lm = lmh_bins[rows, k]
            e = np.quantile(lm, [0, 1 / 3, 2 / 3, 1])
            terc = [(lm >= e[b]) & (lm <= e[b + 1] + 1e-9) for b in range(3)]
            comp = complete[rows, k]
            lr200 = ana["lr200_k"][index_of[rows], k]
            r = dict(rows=rows, split=split, modes_data=md, modes_model=mm, stack_data=st_d, stack_model=st_m,
                     rms_data=rms_d, rms_model=rms_m, W_data=W_d, W_model=W_m, edges=e,
                     data=dict(ls_c=g_d[0], ls_e=g_d[1], share_e=g_d[2], ls_e_r200=g_d[1] - lr200),
                     model=dict(ls_c=g_m[0], ls_e=g_m[1], share_e=g_m[2], ls_e_r200=g_m[1] - lr200),
                     analytic=dict(ls_c=g_a[0], ls_e=g_a[1], share_e=g_a[2], ls_e_r200=g_a[1] - lr200),
                     channel={q: ana[q][index_of[rows], k] for q in ("ls_c", "ls_e", "ls_e_r200", "share_e")},
                     W_ana=W_a, stack_ana=st_a,
                     lr200_k=lr200, complete=comp, lm=lm, logms=logms[rows, k])
            # tercile stacks and their modes
            r["stack_terc_data"] = [stack(W_d[t])[0] for t in terc]
            r["stack_terc_model"] = [stack(W_m[t])[0] for t in terc]
            r["stack_terc_ana"] = [stack(W_a[t])[0] for t in terc]
            r["modes_terc_data"] = [modes_of(s) for s in r["stack_terc_data"]]
            r["modes_terc_model"] = [modes_of(s) for s in r["stack_terc_model"]]
            res[(kname, k)] = r

            print(f"\n  z = {ANCHOR_Z[k]}  ({len(rows)} galaxies; deconvolution rms median data {np.nanmedian(rms_d):.4f} dex, "
                  f"model {np.nanmedian(rms_m):.4f}; tercile edges logMh {e[0]:.2f}/{e[1]:.2f}/{e[2]:.2f}/{e[3]:.2f})")
            print(f"    stacked W, whole sample — data: {md['n_modes']} mode(s) at "
                  + "/".join(f"{v:.1f}" for v in md["peaks_kpc"]) + " kpc, shares "
                  + "/".join(f"{v:.2f}" for v in md["shares"]) + f", gap depth {md['gap_depth']:.2f}; model: {mm['n_modes']} at "
                  + "/".join(f"{v:.1f}" for v in mm["peaks_kpc"]) + " kpc, shares "
                  + "/".join(f"{v:.2f}" for v in mm["shares"]) + f", gap depth {mm['gap_depth']:.2f};  SPLIT {split:.1f} kpc"
                  + (" (the data's gap)" if md["n_modes"] == 2 else (" (the model's gap; the data show one mode)" if mm["n_modes"] == 2
                                                                    else " (FALLBACK 15 kpc: one mode in both)")))
            ma = modes_of(st_a)
            print(f"    the model's TRUE deposit-size distribution (analytic, no kernel): {ma['n_modes']} mode(s) at "
                  + "/".join(f"{v:.1f}" for v in ma["peaks_kpc"]) + " kpc, shares " + "/".join(f"{v:.2f}" for v in ma["shares"])
                  + f"; mass below/above the split {1 - st_a[LOG_S >= np.log10(split)].sum():.2f}/{st_a[LOG_S >= np.log10(split)].sum():.2f}")
            print(f"    {'tercile':<8}{'n':>5}  {'quantity':<20}{'DATA median':>12}{'s.e.':>7}{'16-84':>16}   "
                  f"{'MODEL deconv.':>13}{'s.e.':>7}{'16-84':>16}   {'MODEL true W':>14}{'16-84':>16}   {'data-model':>10}")
            for b in range(3):
                t = terc[b]
                rows_lab = [("compact size [kpc]", "ls_c", True), ("extended size [kpc]", "ls_e", True),
                            ("extended share", "share_e", False), ("log s_e/R200c(z_k)", "ls_e_r200", False)]
                for j, (lab, q, in_kpc) in enumerate(rows_lab):
                    d = med_se(r["data"][q][t], rng); m = med_se(r["model"][q][t], rng); a = med_se(r["analytic"][q][t], rng)
                    if in_kpc:
                        fmt = lambda v: f"{10 ** v:.1f}" if np.isfinite(v) else "nan"
                    else:
                        fmt = lambda v: f"{v:+.2f}" if np.isfinite(v) else "nan"
                    diff = d[0] - m[0]
                    sig = abs(diff) / np.hypot(d[1], m[1]) if np.isfinite(diff) and np.hypot(d[1], m[1]) > 0 else np.nan
                    print(f"    {TERCILE_LABELS[b] if j == 0 else '':<8}{int(t.sum()) if j == 0 else '':>5}  {lab:<20}"
                          f"{fmt(d[0]):>12}{d[1]:>7.3f}{fmt(d[2]) + '..' + fmt(d[3]):>16}   "
                          f"{fmt(m[0]):>13}{m[1]:>7.3f}{fmt(m[2]) + '..' + fmt(m[3]):>16}   "
                          f"{fmt(a[0]):>14}{fmt(a[2]) + '..' + fmt(a[3]):>16}   "
                          f"{diff:>+7.3f}{'' if not in_kpc else ' dex'} ({sig:.1f} s.e.)")
                ch = r["channel"]
                cc = med_se(ch["ls_c"][t], rng); ce = med_se(ch["ls_e"][t], rng); cr = med_se(ch["ls_e_r200"][t], rng); cs = med_se(ch["share_e"][t], rng)
                print(f"    {'':<8}{'':>5}  {'by channel (law)':<20}{'':>12}{'':>7}{'':>16}   {'':>13}{'':>7}{'':>16}   "
                      f"compact ch. {10 ** cc[0]:.1f} kpc, extended ch. {10 ** ce[0]:.1f} kpc = 10^{cr[0]:+.2f} R200c(t'), extended-channel share {cs[0]:.2f}")
                mtd, mtm = r["modes_terc_data"][b], r["modes_terc_model"][b]
                print(f"    {'':<8}{'':>5}  {'stacked peaks [kpc]':<20}{'/'.join(f'{v:.1f}' for v in mtd['peaks_kpc']):>12}"
                      f"{'':>7}{'shares ' + '/'.join(f'{v:.2f}' for v in mtd['shares']):>16}   "
                      f"{'/'.join(f'{v:.1f}' for v in mtm['peaks_kpc']):>13}{'':>7}{'shares ' + '/'.join(f'{v:.2f}' for v in mtm['shares']):>16}")
            # the mh-complete subset, compactly: extended size and share per tercile
            line = []
            for b in range(3):
                t = terc[b] & comp
                d = med_se(r["data"]["ls_e"][t], rng); m = med_se(r["model"]["ls_e"][t], rng)
                ds = med_se(r["data"]["share_e"][t], rng); ms = med_se(r["model"]["share_e"][t], rng)
                line.append(f"{TERCILE_LABELS[b]} n={int(t.sum())}: s_e {10 ** d[0] if np.isfinite(d[0]) else np.nan:.1f} vs "
                            f"{10 ** m[0] if np.isfinite(m[0]) else np.nan:.1f} kpc, share {ds[0]:.2f} vs {ms[0]:.2f}")
            print(f"    mh-complete subset (data vs model deconvolved): " + "; ".join(line))
            # halo radius at the epoch, per tercile
            print(f"    median R200c(z_k) per tercile [kpc]: " + "/".join(f"{10 ** np.median(lr200[t]):.0f}" for t in terc)
                  + f";  median M*(<148) [log Msun]: " + "/".join(f"{np.median(r['logms'][t]):.2f}" for t in terc))

        # item 4: the slopes at z = 2 (and 0.4) of each mode's size against stellar mass
        print(f"\n  slopes d log size / d log M*(<148) [dex per dex] — data | model deconvolved | model analytic:")
        for k in (0, 4):
            r = res[(kname, k)]
            cells = []
            for q, lab in (("ls_c", "compact"), ("ls_e", "extended")):
                cells.append(f"{lab} {slope(r['data'][q], r['logms']):+.3f} | {slope(r['model'][q], r['logms']):+.3f} | "
                             f"{slope(r['analytic'][q], r['logms']):+.3f}")
            r50_t = np.log10(C.size_radius(truth_m[r["rows"], k][:, None, :], Rm, 0.5)[:, 0])
            r50_m = np.log10(C.size_radius(pred_m[index_of[r["rows"]], k][:, None, :], Rm, 0.5)[:, 0])
            cells.append(f"R50 (merged grid) {slope(r50_t, r['logms']):+.3f} | {slope(r50_m, r['logms']):+.3f}")
            print(f"    z = {ANCHOR_Z[k]}: " + ";  ".join(cells))

    # the time trend of the extended size vs the halo radius, under the extended kernel
    kname = "extended kernel"
    print(f"\n{RULE}\n  THE TIME TREND ({kname}): tercile-median log s_e [kpc] per epoch against log R200c(z_k); "
          f"slopes vs log(1+z)\n{RULE}")
    lz = np.log10(1 + np.array(ANCHOR_Z))
    print(f"    {'tercile':<8}{'quantity':<26}" + "".join(f"{f'z={z}':>9}" for z in ANCHOR_Z) + f"{'slope':>9}")
    trend = {}
    for b in range(3):
        for lab, src, q in (("data s_e", "data", "ls_e"), ("model deconv. s_e", "model", "ls_e"),
                            ("model analytic s_e", "analytic", "ls_e"), ("R200c(z_k)", None, None),
                            ("data s_c", "data", "ls_c"), ("model deconv. s_c", "model", "ls_c"),
                            ("model analytic s_c", "analytic", "ls_c"),
                            ("data share_e", "data", "share_e"), ("model analytic share_e", "analytic", "share_e")):
            vals = []
            for k in EPOCHS:
                r = res[(kname, k)]
                lm = r["lm"]; e = r["edges"]
                t = (lm >= e[b]) & (lm <= e[b + 1] + 1e-9)
                x = r["lr200_k"][t] if src is None else r[src][q][t]
                vals.append(float(np.nanmedian(x)))
            vals = np.array(vals)
            trend[(b, lab)] = vals
            print(f"    {TERCILE_LABELS[b] if lab == 'data s_e' else '':<8}{lab:<26}" + "".join(f"{v:>9.3f}" for v in vals)
                  + f"{slope(vals, lz, min_n=3):>+9.2f}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    save = dict(s_grid=S_GRID, Rm=Rm, anchor_z=np.array(ANCHOR_Z), kernels=np.array(list(kernels)),
                theta_baseline=th_b, theta_names=np.array(spec2.theta_names))
    for (kname, k), r in res.items():
        key = f"{kname.replace(' ', '_').replace('=', '')}_z{k}"
        save[f"{key}_rows"] = r["rows"]; save[f"{key}_split"] = r["split"]
        save[f"{key}_stack_data"] = r["stack_data"]; save[f"{key}_stack_model"] = r["stack_model"]; save[f"{key}_stack_ana"] = r["stack_ana"]
        save[f"{key}_stack_terc_ana"] = np.array(r["stack_terc_ana"]); save[f"{key}_W_ana"] = r["W_ana"]
        for q, v in r["channel"].items():
            save[f"{key}_channel_{q}"] = v
        save[f"{key}_stack_terc_data"] = np.array(r["stack_terc_data"]); save[f"{key}_stack_terc_model"] = np.array(r["stack_terc_model"])
        save[f"{key}_W_data"] = r["W_data"]; save[f"{key}_W_model"] = r["W_model"]
        save[f"{key}_rms_data"] = r["rms_data"]; save[f"{key}_rms_model"] = r["rms_model"]
        for src in ("data", "model", "analytic"):
            for q, v in r[src].items():
                save[f"{key}_{src}_{q}"] = v
        save[f"{key}_lr200_k"] = r["lr200_k"]; save[f"{key}_lm"] = r["lm"]; save[f"{key}_logms"] = r["logms"]
        save[f"{key}_complete"] = r["complete"]; save[f"{key}_edges"] = r["edges"]
    save["trend_keys"] = np.array([f"{b}|{lab}" for (b, lab) in trend]); save["trend"] = np.array(list(trend.values()))
    np.savez(OUTDIR / f"stage0_deconvolve{tag}.npz", **save)
    print(f"\n  saved {OUTDIR / f'stage0_deconvolve{tag}.npz'}")
    for pth in figures(res, list(kernels), tag):
        print(f"  wrote {pth.relative_to(ROOT)}")
    print(f"  total {(time.time() - t0) / 60:.1f} min")


# --------------------------------------------------------------------------- #
def figures(res, kernel_names, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hongshao.plotting import set_style, save_fig
    from hongshao.qa import _tex
    set_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    paths = []
    tc = ("#0072B2", "#009E73", "#D55E00")
    # 1. the stacked W per kernel (rows) and epoch (columns), terciles as colours, data solid / model dashed
    fig, ax = plt.subplots(len(kernel_names), 5, figsize=(20, 3.6 * len(kernel_names)), sharex=True, sharey="row")
    ax = np.atleast_2d(ax)
    for i, kn in enumerate(kernel_names):
        for k in EPOCHS:
            r = res[(kn, k)]
            a = ax[i, k]
            for b in range(3):
                a.plot(S_GRID, r["stack_terc_data"][b], "-", color=tc[b], lw=1.8,
                       label=f"data, {TERCILE_LABELS[b]} logMh" if k == 0 else None)
                a.plot(S_GRID, r["stack_terc_model"][b], "--", color=tc[b], lw=1.4,
                       label=f"baseline deconvolved, {TERCILE_LABELS[b]}" if k == 0 else None)
                a.plot(S_GRID, r["stack_terc_ana"][b], ":", color=tc[b], lw=1.2,
                       label=f"baseline true deposits, {TERCILE_LABELS[b]}" if k == 0 else None)
            a.axvline(r["split"], color="k", ls=":", lw=1)
            a.set_xscale("log")
            a.set_title(_tex(f"{kn}, z = {ANCHOR_Z[k]}: split {r['split']:.0f} kpc"), fontsize=9)
            if i == len(kernel_names) - 1:
                a.set_xlabel("deposit half-mass radius s [kpc]")
            if k == 0:
                a.set_ylabel("stacked share of stellar mass per size bin")
                a.legend(fontsize=6.5)
    fig.suptitle("the deposit-size distribution the data require at each epoch (solid) vs the baseline's curves deconvolved the same way (dashed) and its true deposits (dotted)", fontsize=11)
    fig.tight_layout()
    save_fig(fig, FIGDIR / f"exp80_stage0_w_stack{tag}")
    paths.append(FIGDIR / f"exp80_stage0_w_stack{tag}.png")

    # 2. sizes and share vs redshift per tercile, extended kernel: data vs model (deconvolved and analytic)
    kn = "extended kernel"
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))
    zz = np.array(ANCHOR_Z)
    for b in range(3):
        for j, (q, lab) in enumerate((("ls_c", "compact size s_c [kpc]"), ("ls_e", "extended size s_e [kpc]"),
                                      ("share_e", "extended share"), ("ls_e_r200", "log s_e / R200c(z_k)"))):
            a = ax[j]
            med = lambda src: np.array([np.nanmedian(res[(kn, k)][src][q][
                (res[(kn, k)]["lm"] >= res[(kn, k)]["edges"][b]) & (res[(kn, k)]["lm"] <= res[(kn, k)]["edges"][b + 1] + 1e-9)])
                for k in EPOCHS])
            d, m, an = med("data"), med("model"), med("analytic")
            if q in ("ls_e", "ls_e_r200"):
                # an extended "size" where the extended mode holds no mass is the
                # operator's edge bin, not a measurement (the user, 2026-09-10):
                # blank the epochs whose median extended share is below 0.05
                sh_d, sh_m, sh_a = (np.array([np.nanmedian(res[(kn, k)][src]["share_e"][
                    (res[(kn, k)]["lm"] >= res[(kn, k)]["edges"][b]) & (res[(kn, k)]["lm"] <= res[(kn, k)]["edges"][b + 1] + 1e-9)])
                    for k in EPOCHS]) for src in ("data", "model", "analytic"))
                d, m, an = np.where(sh_d >= 0.05, d, np.nan), np.where(sh_m >= 0.05, m, np.nan), np.where(sh_a >= 0.05, an, np.nan)
            f = (lambda v: 10 ** v) if q in ("ls_c", "ls_e") else (lambda v: v)
            a.plot(zz, f(d), "o-", color=tc[b], lw=2, label=f"data, {TERCILE_LABELS[b]} logMh" if j == 0 else None)
            a.plot(zz, f(m), "s--", color=tc[b], lw=1.4, label="baseline, deconvolved" if (j == 0 and b == 0) else None)
            a.plot(zz, f(an), "^:", color=tc[b], lw=1.4, label="baseline, true deposits (same split)" if (j == 0 and b == 0) else None)
            if q == "ls_e":
                r200 = np.array([np.nanmedian(res[(kn, k)]["lr200_k"][
                    (res[(kn, k)]["lm"] >= res[(kn, k)]["edges"][b]) & (res[(kn, k)]["lm"] <= res[(kn, k)]["edges"][b + 1] + 1e-9)])
                    for k in EPOCHS])
                a.plot(zz, 0.1 * 10 ** r200, "-", color=tc[b], lw=0.8, alpha=0.5,
                       label=_tex("0.1 R200c(z_k), same tercile") if b == 0 else None)
            a.set_xlabel("epoch z"); a.set_ylabel(_tex(lab))
            if q in ("ls_c", "ls_e"):
                a.set_yscale("log")
    for a in ax:
        a.legend(fontsize=7)
    fig.suptitle("tercile medians vs epoch under the extended kernel: the data's demanded sizes against the baseline's "
                 "(extended-mode panels blank where that mode holds < 5% of the mass)", fontsize=11)
    fig.tight_layout()
    save_fig(fig, FIGDIR / f"exp80_stage0_sizes_vs_z{tag}")
    paths.append(FIGDIR / f"exp80_stage0_sizes_vs_z{tag}.png")
    return paths


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
