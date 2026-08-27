"""Probe behind tech note 04: is the incumbent deposition sum an analytic integral?

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python \
     doc/tech_note/scripts/04_analytic_check.py 60        # the numbers
     doc/tech_note/scripts/04_analytic_check.py 80 fig    # the figure

A throwaway probe, not an experiment: no gates, no manifest. exp63 Stage 0
rebuilds the engine properly.

Part A  the 72-step sum of exp54 `model.forward` versus the continuum limit
        M*(<R,t) = int_{t_min}^{t} eps(M(t'),z(t')) dM/dt' F(R; R50(t'), 3 R200c(t')) dt'
        with M(t') the analytic DiffMAH curve, z(t') the closed-form flat-LCDM
        relation, and R200c(t') either the catalog value interpolated in log t
        (to isolate the discretisation error) or the analytic (M, rho_c(z))
        value (fully analytic, no catalog at all).
Part B  the closed form: a power-law MAH in an Einstein-de Sitter universe with
        the incumbent's ingredients gives a deposit-size distribution
        W(s) ds ~ s^beta ds/s and, for the gompertz_log kernel, the EXACT CoG
        M*(<R) = A R^beta (ln2)^(-beta/c) c^-1 [gamma(beta/c, ln2 (s_max/R)^c)
                                                 - gamma(beta/c, ln2 (s_min/R)^c)]
        (lower incomplete gamma). Checked against brute-force quadrature.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
from scipy.special import gammainc, gammaincc, gamma as Gamma, expit

def upper_gamma(q, x):
    """Gamma(q, x) for any real q (recurrence lifts q above 0)."""
    x = np.asarray(x, float)
    if q > 0:
        return gammaincc(q, x) * Gamma(q)
    return (upper_gamma(q + 1, x) - x ** q * np.exp(-x)) / q

ROOT = Path(__file__).resolve().parents[3]
EXP54 = ROOT / "experiments/exp54_unpinned_amplitude"
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink", EXP54):
    sys.path.insert(0, str(p))
import fit as F            # noqa  (must precede halo/model)
import halo as H           # noqa
import model as M          # noqa  (puts exp53 on the path)
import stage33 as S33      # noqa
import stage37_size_epoch as S37   # noqa
import families            # noqa
from astropy.table import Table
from hongshao.tng_data import TNG_COSMO, load_cosmic_time
from hongshao.diffmah import log_mah, MAH_K

R = np.geomspace(2.0, 148.22, 24)
H0 = 67.74; OM = 0.3089; OL = 1 - OM
H0_GYR = H0 / 977.792                      # km/s/Mpc -> 1/Gyr
RHO_C0 = 2.775e11 * (H0 / 100) ** 2        # Msun / Mpc^3, h-free

# ---------- closed-form cosmology --------------------------------------------
def a_of_t(t):                             # flat LCDM, exact
    return (OM / OL) ** (1 / 3) * np.sinh(1.5 * H0_GYR * np.sqrt(OL) * t) ** (2 / 3)
def z_of_t(t):
    return 1 / a_of_t(t) - 1
def rho_c(z):
    return RHO_C0 * (OM * (1 + z) ** 3 + OL)
def r200c_analytic(logm, z):               # physical kpc
    return (3 * 10.0 ** logm / (4 * np.pi * 200 * rho_c(z))) ** (1 / 3) * 1e3

# ---------- the incumbent's pieces, as continuous functions ------------------
def log_eps(eff, logmh, z):
    a0, aM, az, aMz = eff
    m, x = logmh - M.M_PIV, np.log1p(z)
    return a0 + aM * m + az * x + aMz * m * x
def r50_law(size, z, r200):
    return 10.0 ** (size[0] + size[1] * np.log10(1 + z)) * r200

def mah_params(indices):
    t = Table.read(ROOT / "experiments/exp27_tng_api_crossmatch/outputs/diffmah_combined.fits")
    row = {int(i): k for k, i in enumerate(t["index"])}
    out = {}
    for i in indices:
        r = t[row[int(i)]]
        out[int(i)] = dict(logmp=float(r["official_logmp_z0"]), logtc=float(r["official_logtc"]),
                           early=float(r["official_early"]), late=float(r["official_late"]),
                           src=str(r["diffmah_source"]))
    return out

def main(n_gal=40, fully_analytic=False):
    tsnap = load_cosmic_time()
    logt0 = np.log10(tsnap[99])   # official DiffMAH anchor: z=0
    rows = np.arange(0, 2397, max(1, 2397 // n_gal))
    recs = H.build_records(rows=rows, verbose=False)
    spec = S33.spec_from_label("gompertz_log-E2-S2")
    theta = S37.incumbent()
    eff, size, shape = spec.unpack(theta)
    prm = mah_params([h.index for h in recs])
    data = np.load(ROOT / "experiments/exp32_full_population/outputs/population.npz")["data"]
    print(f"spec {spec.label}, theta {np.round(theta, 4)}; {len(recs)} galaxies")

    # (0) does the analytic DiffMAH curve reproduce the record's own masses?
    worst = 0.0
    for h in recs:
        p = prm[h.index]
        lm = np.atleast_1d(log_mah(np.log10(h.t), p["logmp"], p["logtc"], p["early"], p["late"], logt0))
        worst = max(worst, float(np.max(np.abs(lm - h.logmh))))
    print(f"(0) analytic DiffMAH vs record logmh: max |dlogM| = {worst:.2e} dex")
    zz = z_of_t(recs[0].t)
    print(f"    closed-form z(t) vs astropy: max |dz| = {np.max(np.abs(zz - recs[0].z)):.2e}")

    # (A) sum vs integral
    def integral(h, k, n_panel=12, n_node=24, analytic_r200=False):
        """continuum limit of the deposition sum up to epoch k, Gauss-Legendre in ln t."""
        p = prm[h.index]
        ok = np.isfinite(h.r200c) & (h.r200c > 0)
        t_hi = h.t[h.epoch_mask[k]][-1]
        t_lo = h.t[0] / (h.t[1] / h.t[0])      # the record's first step starts one snapshot before h.t[0]
        # the discrete sum's first deposit is M(t[0]) - M(t[-1]) where t[-1] is the
        # snapshot before; the record dropped that time, so recover it from the
        # DiffMAH curve: dmh[0] = 10^lm(t0) - 10^lm(t_prev)
        j0 = int(np.argmax(ok))                     # first deposit forward() keeps
        from scipy.optimize import brentq
        target = np.log10(10 ** h.logmh[j0] - h.dmh[j0])
        f = lambda lt: float(np.atleast_1d(log_mah(lt, p["logmp"], p["logtc"], p["early"], p["late"], logt0))[0]) - target
        t_lo = 10 ** brentq(f, np.log10(h.t[j0]) - 1.0, np.log10(h.t[j0]))
        xg, wg = np.polynomial.legendre.leggauss(n_node)
        edges = np.linspace(np.log(t_lo), np.log(t_hi), n_panel + 1)
        tot = np.zeros(len(R))
        lt_r, lr_r = np.log10(h.t[ok]), np.log10(h.r200c[ok])
        for a, b in zip(edges[:-1], edges[1:]):
            tau = 0.5 * (b - a) * xg + 0.5 * (b + a); w = 0.5 * (b - a) * wg
            t = np.exp(tau); lt = np.log10(t)
            lm = np.atleast_1d(log_mah(lt, p["logmp"], p["logtc"], p["early"], p["late"], logt0))
            z = z_of_t(t)
            # dM/dtau = M ln10 dlog10M/dlog10t / ln10 ... : dM/d ln t = M * dlnM/dlnt = M * dlog10M/dlog10t
            s = expit(MAH_K * (lt - p["logtc"]))
            alpha = p["early"] + (p["late"] - p["early"]) * s
            dalpha = (p["late"] - p["early"]) * MAH_K * s * (1 - s)
            dlogm_dlogt = alpha + dalpha * (lt - logt0)
            dM_dtau = 10 ** lm * dlogm_dlogt
            r200 = r200c_analytic(lm, z) if analytic_r200 else 10 ** np.interp(lt, lt_r, lr_r)
            dm = 10 ** log_eps(eff, lm, z) * dM_dtau
            r50 = r50_law(size, z, r200)
            B = M.cog_truncated(spec.family, shape, r50, spec.trunc_C * r200, R)
            tot += B @ (dm * w)
        return tot

    def refined_sum(h, k, sub, analytic_r200=False):
        """the 72-step sum re-done with each snapshot step split into `sub` pieces"""
        p = prm[h.index]
        ok = np.isfinite(h.r200c) & (h.r200c > 0)
        lt_r, lr_r = np.log10(h.t[ok]), np.log10(h.r200c[ok])
        j0 = int(np.argmax(ok))
        from scipy.optimize import brentq
        target = np.log10(10 ** h.logmh[j0] - h.dmh[j0])
        f = lambda lt: float(np.atleast_1d(log_mah(lt, p["logmp"], p["logtc"], p["early"], p["late"], logt0))[0]) - target
        t_prev = 10 ** brentq(f, np.log10(h.t[j0]) - 1.0, np.log10(h.t[j0]))
        tt = h.t[h.epoch_mask[k] & ok]
        lt_edges = np.log10(np.r_[t_prev, tt])
        lt_all = np.concatenate([np.linspace(a, b, sub + 1)[1:] for a, b in zip(lt_edges[:-1], lt_edges[1:])])
        lt_e = np.r_[lt_edges[0], lt_all]
        lm = np.atleast_1d(log_mah(lt_e, p["logmp"], p["logtc"], p["early"], p["late"], logt0))
        dm_h = np.diff(10 ** lm)
        t = 10 ** lt_all; z = z_of_t(t)
        r200 = r200c_analytic(lm[1:], z) if analytic_r200 else 10 ** np.interp(lt_all, lt_r, lr_r)
        dm = 10 ** log_eps(eff, lm[1:], z) * dm_h
        r50 = r50_law(size, z, r200)
        B = M.cog_truncated(spec.family, shape, r50, spec.trunc_C * r200, R)
        return B @ dm

    print("\n(A) the 72-step sum versus its continuum limit (catalog R200c interpolated in log t)")
    print("    per galaxy, z=0.4 and z=2: max |log10(sum/integral)| over 2-148 kpc; and the refined sums")
    t0 = time.time()
    dev = {1: [], 2: [], 4: [], 8: [], "int": []}
    devA = {"int_analytic_r200": []}
    gap_dev, nogap_dev, dropped_frac = [], [], []
    slope_model, slope_data, prof_dev = [], [], []
    i5 = 4
    for h in recs:
        ok = np.isfinite(h.r200c) & (h.r200c > 0)
        j0 = int(np.argmax(ok))
        midgap = int((~ok[j0:]).sum())
        for k in (0, 4):
            ref = M.forward(spec, theta, h, [k], R)[0]
            if not (ref > 0).all():
                continue
            I = integral(h, k)
            # same-convention check: does forward() equal refined_sum(sub=1)?
            s1 = refined_sum(h, k, 1)
            d1 = np.max(np.abs(np.log10(s1 / ref)))
            dev[1].append(d1)
            (gap_dev if midgap else nogap_dev).append(d1)
            if k == 0:
                em = h.epoch_mask[0]
                dm_all = 10 ** log_eps(eff, h.logmh, h.z) * h.dmh
                dropped_frac.append(dm_all[em & ~ok & (np.arange(len(ok)) >= j0)].sum() / dm_all[em & ok].sum())
                slope_model.append(np.log10(ref[i5] / ref[0]) / np.log10(R[i5] / R[0]))
                slope_data.append(np.log10(data[h.row][0, i5] / data[h.row][0, 0]) / np.log10(R[i5] / R[0]))
                prof_dev.append(np.log10(ref / I))
            for sub in (2, 4, 8):
                dev[sub].append(np.max(np.abs(np.log10(refined_sum(h, k, sub) / I))))
            dev["int"].append(np.max(np.abs(np.log10(ref / I))))
            devA["int_analytic_r200"].append(np.max(np.abs(np.log10(integral(h, k, analytic_r200=True) / ref))))
    print(f"    forward() vs my re-implementation of the same 72-step sum (sub=1): max {np.max(dev[1]):.2e} dex  (the convention check)")
    for sub in (2, 4, 8):
        print(f"    sum with each step split {sub}x  vs integral: median {np.median(dev[sub]):.2e}, max {np.max(dev[sub]):.2e} dex")
    print(f"    THE 72-STEP SUM vs the integral            : median {np.median(dev['int']):.3f}, max {np.max(dev['int']):.3f} dex  <- discretisation error of the incumbent")
    print(f"    fully analytic (R200c from M(t), rho_c(z)) vs the 72-step sum: median {np.median(devA['int_analytic_r200']):.3f}, max {np.max(devA['int_analytic_r200']):.3f} dex  <- catalog-vs-analytic R200c")
    print(f"    convention gap split: galaxies WITHOUT a mid-history R200c gap: max {np.max(nogap_dev) if nogap_dev else float('nan'):.2e} dex "
          f"({len(nogap_dev)} cases); WITH a gap: max {np.max(gap_dev) if gap_dev else float('nan'):.2e} dex ({len(gap_dev)} cases)")
    print(f"    stellar mass forward() silently DROPS at mid-history gaps, as a fraction of what it keeps (z=0.4): median {100*np.median(dropped_frac):.2f}%, max {100*np.max(dropped_frac):.2f}%")
    pd = np.array(prof_dev)
    print(f"    72-step sum minus integral, log10, z=0.4, by radius: 2 kpc {np.median(pd[:,0]):+.3f}, 5 kpc {np.median(pd[:,4]):+.3f}, 10 kpc {np.median(pd[:,8]):+.3f}, 30 kpc {np.median(pd[:,14]):+.3f}, 100 kpc {np.median(pd[:,21]):+.3f}, 148 kpc {np.median(pd[:,23]):+.3f}")
    print(f"    inner CoG slope dlogM/dlogR over 2-5 kpc at z=0.4: DATA median {np.median(slope_data):.3f}, INCUMBENT median {np.median(slope_model):.3f}")
    print(f"    ({time.time() - t0:.0f} s)")

    # (B) the closed form on a synthetic power-law MAH in EdS, untruncated kernel
    print("\n(B) closed form: power-law MAH, EdS, a_Mz = 0, untruncated gompertz_log kernel")
    c = shape[0]
    a0, aM, az, _ = eff
    lf0, b = size
    tU = 13.8
    for alpha in (0.5, 1.0, 2.0, 3.0):
        beta = (3 * (1 + aM) * alpha - 2 * az * np.log(10)) / (alpha + 2 * (1 - b))
        # brute force: integrate over ln t from t_lo to t_hi
        t_lo, t_hi = 0.3, 9.39
        tau = np.linspace(np.log(t_lo), np.log(t_hi), 40001)
        t = np.exp(tau)
        Mt = 10 ** 13.5 * (t / t_hi) ** alpha
        opz = (tU / t) ** (2 / 3)
        rc = RHO_C0 * opz ** 3
        r200 = (3 * Mt / (4 * np.pi * 200 * rc)) ** (1 / 3) * 1e3
        s = 10 ** (lf0 + b * np.log10(opz)) * r200
        eps = 10 ** (a0 + aM * (np.log10(Mt) - 13.5) + az * np.log(opz))
        Wtau = eps * alpha * Mt
        Fk = families.cog_unit("gompertz_log", s, (c,), R)      # (24, n)
        brute = np.trapezoid(Fk * Wtau[None, :], tau, axis=1)
        # closed form: W_v = Wtau / (dv/dtau); dv/dtau = gamma_r = alpha/3 + 2(1-b)/3
        gr = alpha / 3 + 2 * (1 - b) / 3
        Wv_ref = Wtau[-1] / gr; s_ref = s[-1]
        A = Wv_ref / s_ref ** beta
        smax, smin = s[-1], s[0]
        q = beta / c
        cf = A * R ** beta * np.log(2) ** (-q) / c * (
            upper_gamma(q, np.log(2) * (smin / R) ** c) - upper_gamma(q, np.log(2) * (smax / R) ** c))
        slope_meas = np.gradient(np.log(brute), np.log(R))
        print(f"    alpha={alpha:.1f}: beta(formula)={beta:+.3f}; local CoG slope at 2-5 kpc (brute) = {slope_meas[:5].mean():+.3f}; "
              f"max |log10(closed/brute)| = {np.max(np.abs(np.log10(cf / brute))):.2e} dex; s range {smin:.2f}-{smax:.1f} kpc")

def part_c():
    """Does the MAH know about the core? Partial Spearman of the measured inner
    CoG shape with the DiffMAH parameters at fixed halo mass, z=0.4."""
    from scipy.stats import spearmanr
    pop = np.load(ROOT / "experiments/exp32_full_population/outputs/population.npz")
    t = Table.read(ROOT / "experiments/exp27_tng_api_crossmatch/outputs/diffmah_combined.fits")
    row = {int(i): k for k, i in enumerate(t["index"])}
    rr = np.array([row[int(i)] for i in pop["index"]])
    early = np.asarray(t["official_early"], float)[rr]; late = np.asarray(t["official_late"], float)[rr]
    logtc = np.asarray(t["official_logtc"], float)[rr]
    d = pop["data"][:, 0, :]                       # z=0.4 CoGs (2397, 24)
    i5, i10, i30 = np.argmin(np.abs(R - 5)), np.argmin(np.abs(R - 10)), np.argmin(np.abs(R - 30))
    f5 = np.log10(d[:, i5] / d[:, i30]); f10 = np.log10(d[:, i10] / d[:, i30])
    slope_in = np.log10(d[:, 4] / d[:, 0]) / np.log10(R[4] / R[0])       # local CoG slope 2-5 kpc
    out30 = np.log10(d[:, -1] / d[:, i30])
    lmh, ms = pop["logmh"], np.log10(d[:, i30])
    ok = np.isfinite(early) & np.isfinite(f5) & np.isfinite(slope_in)
    def resid(y, X):
        X1 = np.column_stack([np.ones(ok.sum()), X])
        return y - X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]
    print("\n(C) DATA: does the halo history know about the core? partial Spearman at fixed log Mh(z=0.4)")
    print(f"    measured inner CoG slope dlogM/dlogR (2-5 kpc): median {np.median(slope_in[ok]):.3f}, 16-84% {np.percentile(slope_in[ok],[16,84]).round(3)}")
    for name, y in (("log M(<5)/M(<30)", f5), ("log M(<10)/M(<30)", f10), ("CoG slope 2-5 kpc", slope_in), ("log M(<148)/M(<30)", out30)):
        ry = resid(y[ok], np.column_stack([lmh[ok], lmh[ok] ** 2]))
        line = f"    {name:<20s} sd {np.std(ry):.3f}: "
        for pn, pv in (("early", early), ("late", late), ("logtc", logtc), ("t50", pop["t50"]), ("c200c", pop["c200c"])):
            m = ok & np.isfinite(pv)
            rp = resid(pv[m], np.column_stack([lmh[m], lmh[m] ** 2]))
            r_, _ = spearmanr(resid(y[m], np.column_stack([lmh[m], lmh[m] ** 2])), rp)
            line += f"{pn} {r_:+.2f}  "
        print(line)

if __name__ == "__main__":
    part_c()
    main(n_gal=int(sys.argv[1]) if len(sys.argv) > 1 else 40)


def figure(n_gal=60):
    """One figure: (a) the 72-step sum's error by radius, (b) closed form vs
    brute force, (c) the deposit-size distribution picture for one galaxy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hongshao.plotting import set_style, save_fig
    set_style()
    tsnap = load_cosmic_time(); logt0 = np.log10(tsnap[99])
    rows = np.arange(0, 2397, max(1, 2397 // n_gal))
    recs = H.build_records(rows=rows, verbose=False)
    spec = S33.spec_from_label("gompertz_log-E2-S2"); theta = S37.incumbent()
    eff, size, shape = spec.unpack(theta)
    prm = mah_params([h.index for h in recs])
    data = np.load(ROOT / "experiments/exp32_full_population/outputs/population.npz")["data"]
    # reuse the closures by re-running main's inner functions: simplest is to copy them
    import types
    g = {}
    exec(open(__file__).read().split("def main(")[0], g)   # module-level helpers
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    # (a) per-galaxy log10(sum/integral) by radius at z=0.4
    devs, slopes_m, slopes_d = [], [], []
    for h in recs:
        ok = np.isfinite(h.r200c) & (h.r200c > 0)
        if (~ok[int(np.argmax(ok)):]).any():
            continue
        ref = M.forward(spec, theta, h, [0], R)[0]
        if not (ref > 0).all():
            continue
        # continuum limit (same code path as main.integral)
        p = prm[h.index]
        from scipy.optimize import brentq
        j0 = int(np.argmax(ok))
        target = np.log10(10 ** h.logmh[j0] - h.dmh[j0])
        f = lambda lt: float(np.atleast_1d(log_mah(lt, p["logmp"], p["logtc"], p["early"], p["late"], logt0))[0]) - target
        t_lo = 10 ** brentq(f, np.log10(h.t[j0]) - 1.0, np.log10(h.t[j0]))
        t_hi = h.t[h.epoch_mask[0]][-1]
        xg, wg = np.polynomial.legendre.leggauss(24)
        edges = np.linspace(np.log(t_lo), np.log(t_hi), 13)
        tot = np.zeros(len(R)); lt_r, lr_r = np.log10(h.t[ok]), np.log10(h.r200c[ok])
        for a, b in zip(edges[:-1], edges[1:]):
            tau = 0.5 * (b - a) * xg + 0.5 * (b + a); w = 0.5 * (b - a) * wg
            t = np.exp(tau); lt = np.log10(t)
            lm = np.atleast_1d(log_mah(lt, p["logmp"], p["logtc"], p["early"], p["late"], logt0)); z = z_of_t(t)
            s = expit(MAH_K * (lt - p["logtc"])); alpha = p["early"] + (p["late"] - p["early"]) * s
            dalpha = (p["late"] - p["early"]) * MAH_K * s * (1 - s)
            dM_dtau = 10 ** lm * (alpha + dalpha * (lt - logt0))
            r200 = 10 ** np.interp(lt, lt_r, lr_r)
            dm = 10 ** log_eps(eff, lm, z) * dM_dtau
            B = M.cog_truncated(spec.family, shape, r50_law(size, z, r200), spec.trunc_C * r200, R)
            tot += B @ (dm * w)
        devs.append(np.log10(ref / tot))
        slopes_m.append(np.log10(ref[4] / ref[0]) / np.log10(R[4] / R[0]))
        slopes_d.append(np.log10(data[h.row][0, 4] / data[h.row][0, 0]) / np.log10(R[4] / R[0]))
    devs = np.array(devs)
    ax[0].fill_between(R, np.percentile(devs, 16, 0), np.percentile(devs, 84, 0), alpha=0.3, label="16-84%")
    ax[0].plot(R, np.median(devs, 0), lw=2, label="median")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xscale("log"); ax[0].set_xlabel("R [kpc]")
    ax[0].set_ylabel(r"$\log_{10}$ (72-step sum / exact integral)")
    ax[0].set_title(f"(a) incumbent's discretisation error, z=0.4 ({len(devs)} galaxies)")
    ax[0].legend(loc="lower right")
    # (b) closed form vs brute force
    c = shape[0]; a0, aM, az, _ = eff; lf0, b = size; tU = 13.8
    for alpha, col in zip((0.5, 1.0, 2.0, 3.0), ("C0", "C1", "C2", "C3")):
        beta = (3 * (1 + aM) * alpha - 2 * az * np.log(10)) / (alpha + 2 * (1 - b))
        t_lo, t_hi = 0.3, 9.39
        tau = np.linspace(np.log(t_lo), np.log(t_hi), 40001); t = np.exp(tau)
        Mt = 10 ** 13.5 * (t / t_hi) ** alpha; opz = (tU / t) ** (2 / 3)
        r200 = (3 * Mt / (4 * np.pi * 200 * RHO_C0 * opz ** 3)) ** (1 / 3) * 1e3
        s = 10 ** (lf0 + b * np.log10(opz)) * r200
        eps = 10 ** (a0 + aM * (np.log10(Mt) - 13.5) + az * np.log(opz))
        Wtau = eps * alpha * Mt
        Rg = np.geomspace(0.3, 300, 80)
        Fk = families.cog_unit("gompertz_log", s, (c,), Rg)
        brute = np.trapezoid(Fk * Wtau[None, :], tau, axis=1)
        gr = alpha / 3 + 2 * (1 - b) / 3
        A = (Wtau[-1] / gr) / s[-1] ** beta; smax, smin = s[-1], s[0]; q = beta / c
        cf = A * Rg ** beta * np.log(2) ** (-q) / c * (upper_gamma(q, np.log(2) * (smin / Rg) ** c) - upper_gamma(q, np.log(2) * (smax / Rg) ** c))
        ax[1].plot(Rg, brute / brute[-1], color=col, lw=3, alpha=0.35, label=rf"$\alpha$={alpha:g}, $\beta$={beta:+.2f} (quadrature)")
        ax[1].plot(Rg, cf / brute[-1], color=col, lw=1.0, ls="--", label="closed form" if alpha == 0.5 else None)
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("R [kpc]")
    ax[1].set_ylabel(r"$M_*(<R)\,/\,M_*(<300\,\mathrm{kpc})$")
    ax[1].set_title("(b) power-law MAH: incomplete-gamma CoG is exact")
    ax[1].legend(fontsize=7, loc="lower right")
    # (c) inner slope histogram
    ax[2].hist(slopes_d, bins=25, range=(0, 2.5), alpha=0.5, label=f"TNG, median {np.median(slopes_d):.2f}")
    ax[2].hist(slopes_m, bins=25, range=(0, 2.5), alpha=0.5, label=f"incumbent, median {np.median(slopes_m):.2f}")
    ax[2].set_xlabel(r"CoG slope $d\log M_*/d\log R$ over 2-5 kpc, z=0.4")
    ax[2].set_ylabel("galaxies"); ax[2].legend()
    ax[2].set_title("(c) the model's centre rises too slowly")
    fig.tight_layout()
    save_fig(fig, Path(__file__).parents[1] / "figures" / "04_analytic_check")
    print("wrote", Path(__file__).parents[1] / "figures" / "04_analytic_check.png")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "fig":
    figure(int(sys.argv[1]))
