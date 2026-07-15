"""exp38 stage 2a — the promoted deposit primitives in the FULL multi-epoch
harness (the exp35/36 structure; only the deposit shape differs from the
incumbent).

Variants:
  2ch-exp  the exp36 2ch-fa architecture with the WIDE channel's Gaussian
           swapped to an EXPONENTIAL profile (Sersic n=1) — the
           single-variable "the split was right, the wide shape was wrong"
           test. Same 16-parameter layout as 2ch-fa: base5 [log_s0, g, q,
           mu, sig] + prune conditioning (log_s0 and sig rows on the
           standardized [logMh, c200c, fz2]) + tail [log_s0_ex, fa + 3
           split slopes]. f_ex -> 0 reduces EXACTLY to the exp35
           single-width model (asserted).
  1ch-mof  a SINGLE channel with a power-law-tail (Moffat) profile inside
           the same dyntrans transport structure (core at rc(t_i)
           migrating to rc*(t_k/t_i)^q, alpha = 1 clock) — the "a heavy
           tail replaces the split entirely" test. 12 parameters: base6
           [log_rc, g, q, mu, sig, gamma] + the two conditioning rows.

Everything else is inherited unchanged: lognormal efficiency, M(<500)
normalization, the physical box (+ gamma in (1.05, 6)), joint 5-epoch fit,
10-fold CV, the differential-deposition test, the overshoot terciles, and
the bounds-stress protocol.

Run: PYTHONPATH=. uv run python experiments/exp38_deposit_rethink/\
stage2_multiepoch.py {demo|fit|cv|differential|overshoot|stress|report} \
[--variant 2ch-exp] [--dev]
"""
import importlib.util
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
E35_DIR = ROOT / "experiments/exp35_total_norm"
E36_DIR = ROOT / "experiments/exp36_two_channel"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
VARIANTS = ("2ch-exp", "1ch-mof")
NFOLD = 10
KS = [0, 1, 2, 3, 4]
S0EX_HI = 3.0                    # stress overrides (parent-side penalty only)
GAMMA_BOX = (1.05, 6.0)
_W = {}


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _w_init(rows):
    e = _load_by_path("exp35_run", E35_DIR / "run.py")
    mh_scale = tuple(np.load(ROOT / "experiments/exp32_full_population/"
                             "outputs/um_slope_diffmah.npz")["mh_scale"])
    e._init("diffmah", mh_scale, None if rows is None else np.asarray(rows))
    pop = np.load(POP_NPZ)
    row_of = {g["row"]: g for g in e._G["gals"]}
    z_all = np.column_stack([pop["logmh"], pop["c200c"], pop["fz2"]])
    mu = np.nanmean(z_all, axis=0)
    sd = np.nanstd(z_all, axis=0)
    for r, g in row_of.items():
        z = (z_all[r] - mu) / sd
        g["cond"] = np.where(np.isfinite(z), z, 0.0)
    _W["e"] = e
    _W["gals"] = e._G["gals"]


def _w_init_cached(rows=None):
    if "e" not in _W:
        _w_init(rows)


# --------------------------------------------------------------------------- #
# the two swapped bases (mirror exp35 basis_ext; only the profile differs)     #
# --------------------------------------------------------------------------- #
def basis_exp(th4, ti, t_obs, tk, r):
    """Exponential-profile dyntrans basis: CoG(x) = 1 - (1+x)exp(-x),
    x = R/a — same transport structure as exp35 basis_ext, Gaussian ->
    exponential."""
    a0 = np.clip(10.0 ** th4[0] * (ti / t_obs) ** th4[1], 1e-4, 1e5)
    dt = np.clip(tk - ti, 0.0, None)
    fc = np.exp(-dt / (10.0 ** th4[2] * ti))
    aw = np.clip(a0 * (tk / ti) ** max(th4[3], 0.0), 1e-4, 1e5)

    def cog(a):
        x = r[:, None] / a[None, :]
        return 1.0 - (1.0 + x) * np.exp(-x)
    return fc[None, :] * cog(a0) + (1.0 - fc)[None, :] * cog(aw)


def basis_mof(th4, gam, ti, t_obs, tk, r):
    """Moffat (power-law-tail) dyntrans basis: CoG(x) = 1 - (1+x^2)^(1-gam),
    x = R/rc; finite mass needs gam > 1 (the box enforces it)."""
    rc0 = np.clip(10.0 ** th4[0] * (ti / t_obs) ** th4[1], 1e-4, 1e5)
    dt = np.clip(tk - ti, 0.0, None)
    fc = np.exp(-dt / (10.0 ** th4[2] * ti))
    rcw = np.clip(rc0 * (tk / ti) ** max(th4[3], 0.0), 1e-4, 1e5)

    def cog(rc):
        u = 1.0 + (r[:, None] / rc[None, :]) ** 2
        return 1.0 - u ** (1.0 - gam)
    return fc[None, :] * cog(rc0) + (1.0 - fc)[None, :] * cog(rcw)


# --------------------------------------------------------------------------- #
# theta layer + model                                                          #
# --------------------------------------------------------------------------- #
def theta_of(p, g, variant):
    """Per-galaxy conditioned parameters (prune-style: log-scale and sig
    rows on the standardized [logMh, c200c, fz2])."""
    p = np.asarray(p, float)
    if variant == "2ch-exp":
        S = np.zeros((5, 3))
        S[0] = p[5:8]
        S[4] = p[8:11]
        base5 = p[:5] + S @ g["cond"]
        tail = p[11:16]
        return base5, tail[0], float(expit(tail[1] + tail[2:] @ g["cond"]))
    # 1ch-mof: base6 [log_rc, g, q, mu, sig, gamma]
    S = np.zeros((6, 3))
    S[0] = p[6:9]
    S[4] = p[9:12]
    base6 = p[:6] + S @ g["cond"]
    return base6, None, None


def model_cogs(p, g, ks, variant):
    """M(<500)-normalized model CoGs at epochs ks for one galaxy."""
    e = _W["e"]
    mah = g["mah"]
    if variant == "2ch-exp":
        base5, ls_ex, f_ex = theta_of(p, g, variant)
        w = e.weights(mah["z"], base5[3], base5[4])
    else:
        base6, _, _ = theta_of(p, g, variant)
        w = e.weights(mah["z"], base6[3], base6[4])
    dM = w * mah["dMh"]
    if not np.isfinite(dM).all() or dM.sum() <= 0:
        return None
    dM = dM / dM.sum()
    out = []
    for k in ks:
        mask = mah["snap"] <= e.ANCHOR_SNAP[k]
        if variant == "2ch-exp":
            th_in = [base5[0], base5[1], 0.0, base5[2]]
            th_ex = [ls_ex, base5[1], 0.0, base5[2]]
            B_in = e.basis_ext(th_in, mah["t"], mah["t_obs"], e.pe.AT[k],
                               e.R_EXT)
            m = (1.0 - f_ex) * (B_in @ (dM * mask))
            if f_ex > 0.0:
                B_ex = basis_exp(th_ex, mah["t"], mah["t_obs"], e.pe.AT[k],
                                 e.R_EXT)
                m = m + f_ex * (B_ex @ (dM * mask))
        else:
            th4 = [base6[0], base6[1], 0.0, base6[2]]
            gam = float(np.clip(base6[5], GAMMA_BOX[0] + 1e-6, GAMMA_BOX[1]))
            B = basis_mof(th4, gam, mah["t"], mah["t_obs"], e.pe.AT[k],
                          e.R_EXT)
            m = B @ (dM * mask)
        if not np.isfinite(m[-1]) or m[-1] <= 0 or m[-2] <= 0:
            return None
        out.append(m[:-1] * (g["m500"][k] / m[-1]))
    return out


def penalty(p, variant):
    # parent-side only (fit_pop adds it in the parent, so the stress
    # override of S0EX_HI reaches the fit; _cv_fold children see defaults)
    e = _W["e"]
    if variant == "2ch-exp":
        base = e.penalty(p[:5])
        tail = p[11:16]
        lo_t = np.array([1.0, -6.0, -4.0, -4.0, -4.0])
        hi_t = np.array([S0EX_HI, 6.0, 4.0, 4.0, 4.0])
        return base + 30.0 * float(np.sum(np.clip(lo_t - tail, 0, None) ** 2
                                          + np.clip(tail - hi_t, 0, None) ** 2))
    base = e.penalty(p[:5])                     # [log_rc, g, q, mu, sig]
    lo_g, hi_g = GAMMA_BOX
    return base + 30.0 * float(np.clip(lo_g - p[5], 0, None) ** 2
                               + np.clip(p[5] - hi_g, 0, None) ** 2)


def gal_loss(p, g, ks, variant):
    cogs = model_cogs(p, g, ks, variant)
    if cogs is None:
        return 4.0
    return float(np.mean([np.sqrt(np.mean(((c - g["data"][k]) / g["data"][k])
                                          ** 2))
                          for c, k in zip(cogs, ks)]))


def gal_eval(p, g, ks, variant, rmin=5.0):
    """[all-R max|rel|, R>rmin, 148-pinned shape R>rmin, dlog f148]/epoch."""
    e = _W["e"]
    cogs = model_cogs(p, g, ks, variant)
    if cogs is None:
        return (np.full((len(ks), 4), np.nan),
                np.full((len(ks), len(e.R)), np.nan))
    m = e.R > rmin
    met = []
    for c, k in zip(cogs, ks):
        rel = (c - g["data"][k]) / g["data"][k]
        cs = c * (g["data"][k][-1] / c[-1])
        rel_s = (cs - g["data"][k]) / g["data"][k]
        met.append([np.abs(rel).max(), np.abs(rel[m]).max(),
                    np.abs(rel_s[m]).max(),
                    np.log10(c[-1] / g["data"][k][-1])])
    return np.array(met), np.array(cogs)


def _chunk(args):
    p, variant, ks, lo, hi = args
    return sum(gal_loss(p, g, ks, variant) for g in _W["gals"][lo:hi])


def _cv_fold(args):
    variant, ks, fold, warm, maxiter = args
    gals = _W["gals"]
    n = len(gals)
    train = [i for i in range(n) if i % NFOLD != fold]
    r = minimize(lambda p: np.mean([gal_loss(p, gals[i], ks, variant)
                                    for i in train]) + penalty(p, variant),
                 warm, method="Nelder-Mead",
                 options=dict(maxiter=maxiter, xatol=3e-4, fatol=1e-8))
    return [(gals[i]["row"], *gal_eval(r.x, gals[i], ks, variant))
            for i in range(n) if i % NFOLD == fold]


def fit_pop(variant, ks, starts, maxiter, pool, nchunk, label):
    n = len(_W["gals"])
    edges = np.linspace(0, n, nchunk + 1).astype(int)

    def loss(p):
        parts = pool.map(_chunk, [(p, variant, ks, edges[i], edges[i + 1])
                                  for i in range(nchunk)])
        return sum(parts) / n + penalty(p, variant)

    best = None
    for p0 in starts:
        t0 = time.time()
        r = minimize(loss, p0, method="Nelder-Mead",
                     options=dict(maxiter=maxiter, xatol=3e-4, fatol=1e-8))
        print(f"  [{label}] start: loss {r.fun:.4f} "
              f"({(time.time()-t0)/60:.1f} min, {r.nit} iters)", flush=True)
        if best is None or r.fun < best.fun:
            best = r
    return best.x, best.fun


def _npz(tag=""):
    return OUTDIR / f"stage2_multiepoch{tag}.npz"


def _at_bound(p, variant):
    """Names of parameters within 2% of their box edges."""
    e = _W["e"]
    names, lo, hi = [], [], []
    names += ["log_s0", "g", "q", "mu", "sig"]
    lo += list(e.LO)
    hi += list(e.HI)
    if variant == "2ch-exp":
        names += ["log_s0_ex", "fa", "fb_mh", "fb_c", "fb_f"]
        lo += [1.0, -6.0, -4.0, -4.0, -4.0]
        hi += [S0EX_HI, 6.0, 4.0, 4.0, 4.0]
        vals = np.concatenate([p[:5], p[11:16]])
    else:
        names[0] = "log_rc"
        names += ["gamma"]
        lo += [GAMMA_BOX[0]]
        hi += [GAMMA_BOX[1]]
        vals = np.concatenate([p[:5], [p[5]]])
    out = []
    for v, nm, lo_, hi_ in zip(vals, names, lo, hi):
        w = hi_ - lo_
        if v < lo_ + 0.02 * w or v > hi_ - 0.02 * w:
            out.append(f"{nm}={v:.2f}")
    return out


def cmd_fit(dev=False, variants=VARIANTS):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    tag = "_dev" if dev else ""
    scale = 0.1 if dev else 1.0
    workers = max(os.cpu_count() - 2, 2)
    e36 = np.load(E36_DIR / "outputs/two_channel.npz")
    fitted = dict(np.load(_npz(tag))) if _npz(tag).exists() else {}
    print(f"exp38 stage 2a fit (n={len(_W['gals'])}{', DEV' if dev else ''}, "
          "joint 5-epoch)")
    with Pool(workers, initializer=_w_init, initargs=(rows,)) as pool:
        for variant in variants:
            if variant == "2ch-exp":
                # warm from the incumbent's basin (identical layout) + a
                # lower-ls_ex foothold (the exponential reaches further at
                # a given scale)
                t = e36["theta_multi_2ch-fa"]
                t2 = t.copy()
                t2[11] = 2.2
                starts = [t, t2]
            else:
                s1 = np.load(OUTDIR / "stage1_deposit_dev.npz")
                base = np.median(np.stack(
                    [s1[f"theta_moffat_k{k}"] for k in range(5)]), axis=0)
                starts = [np.concatenate([[base[0], base[1], 0.8, base[2],
                                           base[3], base[4]], np.zeros(6)]),
                          np.concatenate([[1.8, 1.5, 1.2, 1.6, 0.35, 1.3],
                                          np.zeros(6)])]
            th, lo = fit_pop(variant, KS, starts,
                             max(int(5000 * scale), 80), pool, workers,
                             variant)
            fitted[f"theta_{variant}"] = th
            fitted[f"loss_{variant}"] = lo
            ab = _at_bound(th, variant)
            print(f"    theta {np.round(th, 2)}")
            print("    at bound: " + (", ".join(ab) if ab else "NONE"))
            OUTDIR.mkdir(exist_ok=True)
            np.savez(_npz(tag), **fitted)
    print(f"wrote {_npz(tag)}")


def cmd_cv(dev=False, variants=VARIANTS):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    gals = _W["gals"]
    n = len(gals)
    e = _W["e"]
    tag = "_dev" if dev else ""
    fitted = np.load(_npz(tag))
    workers = min(max(os.cpu_count() - 2, 2), NFOLD)
    maxiter = 150 if dev else 1500
    row_to_i = {g["row"]: i for i, g in enumerate(gals)}
    out = dict(np.load(_npz(tag)))
    print(f"exp38 stage 2a {NFOLD}-fold CV (joint 5-epoch; metrics: all-R / "
          "R>5 | pinned shape R>5 | dlog f148; 2ch-fa marks "
          "17.7/17.1/16.3/16.0/14.7):")
    for variant in variants:
        t0 = time.time()
        met = np.full((n, len(KS), 4), np.nan)
        cogs = np.full((n, len(KS), len(e.R)), np.nan)
        with Pool(workers, initializer=_w_init, initargs=(rows,)) as pool:
            jobs = [(variant, KS, f, fitted[f"theta_{variant}"], maxiter)
                    for f in range(NFOLD)]
            for part in pool.map(_cv_fold, jobs):
                for row, m_, c_ in part:
                    met[row_to_i[row]] = m_
                    cogs[row_to_i[row]] = c_
        line = " ".join(
            f"z{e.ANCHOR_Z[k]}: {100*np.nanmedian(met[:, j, 0]):.1f}/"
            f"{100*np.nanmedian(met[:, j, 1]):.1f} | "
            f"{100*np.nanmedian(met[:, j, 2]):.1f} | "
            f"{np.nanmedian(met[:, j, 3]):+.4f}"
            for j, k in enumerate(KS))
        print(f"  {variant}: {line}  ({(time.time()-t0)/60:.1f} min)",
              flush=True)
        out[f"met_{variant}"] = met
        out[f"cogs_{variant}"] = cogs
    np.savez(_npz(tag), **out)
    print(f"wrote {_npz(tag)}")


def cmd_differential(dev=False):
    tag = "_dev" if dev else ""
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    gals = _W["gals"]
    e = _W["e"]
    d = np.load(_npz(tag))
    data = np.stack([g["data"] for g in gals])
    logms = np.array([g["logms"] for g in gals])
    print("exp38 stage 2a differential deposition (data -> model, "
          "f>50/f>100; marks: data 0.37/0.11, exp36 2ch-fa 0.40/0.14 "
          "massive z0.7->0.4):")
    for variant in VARIANTS:
        key = f"theta_{variant}"
        if key not in d.files:
            continue
        cogs = np.full((len(gals), 5, len(e.R)), np.nan)
        for i, g in enumerate(gals):
            out = model_cogs(d[key], g, KS, variant)
            if out is not None:
                cogs[i] = out
        ed3, rows_d = e.differential(cogs, data, logms)
        for b in range(3):
            cells = [f"z{e.ANCHOR_Z[k+1]}->z{e.ANCHOR_Z[k]}: "
                     f"{rows_d[('data', b, k)][0]:.2f}/"
                     f"{rows_d[('data', b, k)][1]:.2f} -> "
                     f"{rows_d[('model', b, k)][0]:.2f}/"
                     f"{rows_d[('model', b, k)][1]:.2f}"
                     for k in range(4)]
            print(f"  [{variant}] logM* {ed3[b]:.2f}-{ed3[b+1]:.2f}: "
                  + "  ".join(cells))


def cmd_overshoot(dev=False):
    from hongshao.profile_emulator import density_from_cog
    tag = "_dev" if dev else ""
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    gals = _W["gals"]
    e = _W["e"]
    d = np.load(_npz(tag))
    pop = np.load(POP_NPZ)
    logms = np.array([pop["logms"][g["row"]] for g in gals])
    data0 = np.stack([g["data"][0] for g in gals])
    ls_d, mid = density_from_cog(np.log10(data0), e.R)
    bands = ((mid >= 30.0) & (mid < 60.0), (mid >= 60.0) & (mid <= 148.0))
    edges = np.quantile(logms, [0, 1 / 3, 2 / 3, 1])
    print("exp38 stage 2a outskirt overshoot — held-out tercile dlog Sigma "
          "(model - data), z=0.4 [30-60 / 60-148 kpc]; exp36 2ch-fa: "
          "T1 +0.029/+0.070:")
    for variant in VARIANTS:
        key = f"cogs_{variant}"
        if key not in d.files:
            continue
        cogs = d[key][:, 0]
        ok = np.isfinite(cogs).all(1) & (cogs > 0).all(1)
        ls_m = np.full_like(ls_d, np.nan)
        ls_m[ok] = density_from_cog(np.log10(cogs[ok]), e.R)[0]
        dl = ls_m - ls_d
        cells = []
        for b in range(3):
            sel = ok & (logms >= edges[b]) & (logms <= edges[b + 1] + 1e-9)
            cells.append(" ".join(f"{np.nanmedian(dl[np.ix_(sel, bm)]):+.3f}"
                                  for bm in bands))
        print(f"  {variant}: " + "  |  ".join(
            f"T{b+1} {c}" for b, c in enumerate(cells)))


def cmd_stress(dev=False, variants=VARIANTS):
    """Loosen each variant's watched bound and ask HOW the freedom is spent
    (the exp35/36 protocol): 2ch-exp -> log_s0_ex 3.0 -> 3.5 (horizon
    escape?); 1ch-mof -> mu 3.0 -> 4.5 (is the efficiency-peak rail a flat
    direction?)."""
    global S0EX_HI
    tag = "_dev" if dev else ""
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    gals = _W["gals"]
    e = _W["e"]
    d = np.load(_npz(tag))
    data = np.stack([g["data"] for g in gals])
    logms = np.array([g["logms"] for g in gals])
    m500 = np.stack([g["m500"] for g in gals])
    massive = logms >= np.quantile(logms, 2 / 3)
    workers = max(os.cpu_count() - 2, 2)
    scale = 0.1 if dev else 1.0
    res = {}
    for variant in variants:
        base_key = f"theta_{variant}"
        if base_key not in d.files:
            continue
        warm = d[base_key]
        nudge = warm.copy()
        if variant == "2ch-exp":
            S0EX_HI = 3.5
            nudge[11] = 3.3
            label = "log_s0_ex 3.0->3.5"
        else:
            e.HI = np.array([3.0, 4.0, 3.0, 4.5, 2.0])   # mu box loosened
            nudge[3] = 3.5
            label = "mu 3.0->4.5"
        with Pool(workers, initializer=_w_init, initargs=(rows,)) as pool:
            th, lo = fit_pop(variant, KS, [warm, nudge],
                             max(int(4000 * scale), 80), pool, workers,
                             f"stress-{variant}")
        print(f"\n  [{variant}] stress ({label}): loss {lo:.4f} vs baseline "
              f"{float(d[f'loss_{variant}']):.4f}")
        print(f"    theta {np.round(th, 2)}; at bound: "
              + (", ".join(_at_bound(th, variant)) or "NONE"))
        cogs = np.full((len(gals), 5, len(e.R)), np.nan)
        for i, g in enumerate(gals):
            out = model_cogs(th, g, KS, variant)
            if out is not None:
                cogs[i] = out
        _, rows_d = e.differential(cogs, data, logms)
        print(f"    massive f148 (z=0.4) "
              f"{np.nanmedian(cogs[massive, 0, -1] / m500[massive, 0]):.3f} "
              f"(data {np.median(data[massive, 0, -1] / m500[massive, 0]):.3f})"
              "; differential z0.7->z0.4 massive "
              + "/".join(f"{v:.2f}" for v in rows_d[("model", 2, 0)]))
        res[f"theta_stress_{variant}"] = th
        res[f"loss_stress_{variant}"] = lo
        if variant == "2ch-exp":
            S0EX_HI = 3.0
        else:
            e.HI = np.array([3.0, 4.0, 3.0, 3.0, 2.0])
    np.savez(OUTDIR / f"stage2_stress{tag}.npz", **res)
    print(f"wrote {OUTDIR / f'stage2_stress{tag}.npz'}")


def cmd_report(dev=False):
    from hongshao import qa
    tag = "_dev" if dev else ""
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    gals = _W["gals"]
    e = _W["e"]
    d = np.load(_npz(tag))
    data = np.stack([g["data"] for g in gals])
    logmh = np.array([g["logmh"] for g in gals])
    FIGDIR.mkdir(exist_ok=True)
    for variant in VARIANTS:
        if f"cogs_{variant}" not in d.files:
            continue
        qa.evaluate(d[f"cogs_{variant}"], data[:, KS], e.R,
                    [e.ANCHOR_Z[k] for k in KS],
                    name=f"exp38_{variant}{tag}", figdir=FIGDIR,
                    figures=True, bin_by=logmh, bin_label="logMh")


def demo():
    rows = np.load(POP_NPZ)["dev100"][:12]
    _w_init(rows)
    e = _W["e"]
    gals = _W["gals"]
    p5 = np.array([2.4, 2.0, 0.9, 1.3, 0.4])

    # (1) 2ch-exp with f_ex -> 0 reduces EXACTLY to the exp35 base model
    p_exp = np.concatenate([p5, np.zeros(6), [2.5, -30.0, 0.0, 0.0, 0.0]])
    for g in gals[:6]:
        ref = e.model_cogs_total(p5, g["mah"], g["m500"], [0, 4])
        got = model_cogs(p_exp, g, [0, 4], "2ch-exp")
        err = max(np.abs(np.asarray(a) / np.asarray(b) - 1.0).max()
                  for a, b in zip(got, ref))
        assert err < 1e-9, f"f_ex=0 nesting broken: {err:.2e}"

    # (2) the exponential wide channel moves mass outward as f_ex rises
    g = gals[0]
    fr = []
    for fa in (-30.0, 0.0, 3.0):
        p_ = p_exp.copy()
        p_[12] = fa
        c = model_cogs(p_, g, [0], "2ch-exp")[0]
        fr.append(c[int(np.searchsorted(e.R, 50.0))] / c[-1])
    assert fr[0] > fr[1] > fr[2], "inner fraction must fall with f_ex"

    # (3) exponential vs Gaussian wide channel at the SAME scale: the
    # exponential has heavier wings (less mass inside 50 kpc)
    from shapes import sersic_cog
    rr = e.R
    a = np.array([50.0])
    one = np.array([1.0])
    cog_e = sersic_cog(one, a, 1.0, rr)
    cog_g = 1.0 - np.exp(-rr ** 2 / (2.0 * a[0] ** 2))
    i50 = int(np.searchsorted(rr, 50.0))
    assert cog_e[i50] < cog_g[i50], "exponential must be heavier-winged"

    # (4) 1ch-mof: finite, monotone, M(<500)-normalized, gamma box enforced
    p_mof = np.concatenate([[2.0, 1.5, 0.9, 1.3, 0.4, 1.3], np.zeros(6)])
    for g in gals[:6]:
        c = model_cogs(p_mof, g, KS, "1ch-mof")
        assert c is not None
        for k, ck in zip(KS, c):
            assert np.isfinite(ck).all() and np.all(np.diff(ck) > -1e-9)
            assert ck[-1] < g["m500"][k]
    assert penalty(np.concatenate([[2.0, 1.5, 0.9, 1.3, 0.4, 0.5],
                                   np.zeros(6)]), "1ch-mof") > 1.0

    # (5) conditioning: zero slopes leave theta at the population values;
    # a c200c slope moves the scale only through cond[1]
    g = gals[3]
    b6, _, _ = theta_of(p_mof, g, "1ch-mof")
    assert np.allclose(b6, p_mof[:6])
    p2 = p_mof.copy()
    p2[7] = 0.3
    b6b, _, _ = theta_of(p2, g, "1ch-mof")
    assert np.isclose(b6b[0] - b6[0], 0.3 * g["cond"][1])

    print("stage2 demo OK: f_ex=0 nests exp35 exactly; exponential wide "
          "channel is heavier-winged and moves mass outward; 1ch-mof "
          "finite/monotone/normalized with the gamma box enforced; the "
          "conditioning layer acts through the standardized halo vector")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    dev = "--dev" in sys.argv
    variants = ([sys.argv[sys.argv.index("--variant") + 1]]
                if "--variant" in sys.argv else VARIANTS)
    if cmd == "demo":
        demo()
    elif cmd == "fit":
        cmd_fit(dev, variants)
    elif cmd == "cv":
        cmd_cv(dev, variants)
    elif cmd == "differential":
        cmd_differential(dev)
    elif cmd == "overshoot":
        cmd_overshoot(dev)
    elif cmd == "stress":
        cmd_stress(dev, variants)
    elif cmd == "report":
        cmd_report(dev)
    else:
        sys.exit(f"unknown subcommand {cmd!r}")
