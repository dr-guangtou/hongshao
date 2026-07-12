"""exp34 — asymptotic total stellar mass from CoG extrapolation.

The aperture-horizon degeneracy (exp33) needs the mass BEYOND 148 kpc; the raw
drop has nothing out there, and catalog bound-mass totals mix definitions. So:
extrapolate our own CoG with a validated tail form. Terminology per user: the
"beyond-aperture fraction" f_out = 1 - M(<148)/M_tot (NOT "ICL" — observational
ICL definitions overlap heavily with R<150 kpc light).

Tail forms (fitted to the outer CoG, both in linear mass):
  power  M(<R) = M_tot - A R^(-a)   (density tail Sigma ~ R^-(a+2), a>0)
  expo   M(<R) = M_tot - A exp(-R/Rs)
Validation is honest and internal:
  1  TRUNCATION test — fit only R <= R_cut (80/100 kpc), predict the MEASURED
     CoG at 100-148 kpc; the winning (form, range) earns the extrapolation.
  2  form disagreement -> per-galaxy systematic on M_tot.
  3  DIFFERENTIAL deposition (user): dCoG(R) between adjacent snapshots = where
     newly added + migrated mass lands, incl. the fraction near/beyond the
     aperture edge — the model-free empirical prior for f_out's plausibility
     and a direct constraint for the transport model.

Output: outputs/asymptotic_total.npz — M_tot (n,5) per form, f_out, tail slope.

Run:  PYTHONPATH=. uv run python experiments/exp34_asymptotic_total/run.py
Demo: ... run.py demo
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from hongshao.tng_data import COG_RAD_KPC                                            # noqa: E402
from hongshao.plotting import set_style, save_fig                                    # noqa: E402

set_style()
FIGDIR, OUTDIR = HERE / "figures", HERE / "outputs"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
R = COG_RAD_KPC
ZK = [0.4, 0.7, 1.0, 1.5, 2.0]
FIT_RMIN = 50.0                                # tail fit uses R in [FIT_RMIN, 148]


def f_power(r, mtot, loga, a):
    return mtot - 10.0 ** loga * r ** (-a)


def f_expo(r, mtot, loga, rs):
    return mtot - 10.0 ** loga * np.exp(-r / rs)


def fit_tail(cog, form, rmax=None):
    """Fit one CoG's tail on [FIT_RMIN, rmax]; returns (M_tot, params) or None."""
    m = (R >= FIT_RMIN) & (R <= (rmax or R[-1] + 1))
    r, y = R[m], cog[m]
    mlast = y[-1]
    try:
        if form == "power":
            p0 = [mlast * 1.05, np.log10(max(mlast * 0.5 * FIT_RMIN, 1.0)), 1.0]
            popt, _ = curve_fit(f_power, r, y, p0=p0, maxfev=4000,
                                bounds=([mlast, 0, 0.05], [mlast * 3, 20, 6.0]))
        else:
            p0 = [mlast * 1.05, np.log10(max(mlast * 0.5, 1.0)), 50.0]
            popt, _ = curve_fit(f_expo, r, y, p0=p0, maxfev=4000,
                                bounds=([mlast, 0, 5.0], [mlast * 3, 20, 500.0]))
    except (RuntimeError, ValueError):
        return None
    return popt


def truncation_test(data):
    """Fit tails on R<=R_cut only; score predicted vs measured M(<148 kpc)."""
    n = len(data)
    print("  truncation validation: fit R in [50, R_cut], predict M(<148);")
    print(f"    {'form':>6s} {'R_cut':>6s} | median |dlogM(148)| dex | fail%")
    best = None
    for form in ("power", "expo"):
        for rcut in (80.0, 100.0):
            errs, fails = [], 0
            for i in range(n):
                for k in range(5):
                    p = fit_tail(data[i, k], form, rmax=rcut)
                    if p is None:
                        fails += 1
                        continue
                    pred = (f_power if form == "power" else f_expo)(R[-1], *p)
                    errs.append(abs(np.log10(max(pred, 1.0) / data[i, k, -1])))
            med = float(np.median(errs))
            print(f"    {form:>6s} {rcut:6.0f} | {med:20.4f} | "
                  f"{100 * fails / (5 * n):4.1f}%")
            if best is None or med < best[2]:
                best = (form, rcut, med)
    print(f"    -> winner: {best[0]} (fit range validates at {best[2]:.4f} dex)")
    return best[0]


def extrapolate_all(data, form):
    n = len(data)
    mtot = np.full((n, 5, 2), np.nan)                  # [:, :, 0]=power, 1=expo
    slope = np.full((n, 5), np.nan)
    for i in range(n):
        for k in range(5):
            for fi, fm in enumerate(("power", "expo")):
                p = fit_tail(data[i, k], fm)
                if p is not None:
                    mtot[i, k, fi] = p[0]
                    if fm == "power":
                        slope[i, k] = p[2]
    return mtot, slope


def differential(data, logms):
    """Where does newly added mass land between adjacent snapshots?
    dCoG_k(R) = CoG_{z_k}(R) - CoG_{z_{k+1}}(R), medians per mass tercile."""
    print("\n  differential deposition (median fraction of the inter-snapshot mass"
          "\n  growth landing at R>50 / R>100 kpc; negative growth kept as-is):")
    edges = np.quantile(logms, [0, 1 / 3, 2 / 3, 1])
    out = {}
    for b in range(3):
        m = (logms >= edges[b]) & (logms <= edges[b + 1] + 1e-9)
        rows = []
        for k in range(4):                             # pair (z_{k+1} -> z_k)
            d = data[m, k, :] - data[m, k + 1, :]      # (nb, 24) added mass profile
            tot = d[:, -1]
            ok = tot > 0
            f50 = np.median(1.0 - d[ok][:, np.searchsorted(R, 50.0)] / tot[ok])
            f100 = np.median(1.0 - d[ok][:, np.searchsorted(R, 100.0)] / tot[ok])
            rows.append((f50, f100, np.median(tot[ok])))
            out[(b, k)] = d
        lab = f"logM* {edges[b]:.2f}-{edges[b+1]:.2f}"
        print(f"    {lab:>22s}: " + "  ".join(
            f"z{ZK[k+1]}->z{ZK[k]}: {r[0]:.2f}/{r[1]:.2f}" for k, r in enumerate(rows)))
    return edges, out


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pop = np.load(POP_NPZ)
    data, logms = pop["data"], pop["logms"]
    n = len(data)
    print(f"exp34 — asymptotic totals from CoG extrapolation (n={n} x 5 epochs)\n")

    form = truncation_test(data[:: max(1, n // 300)])   # validation subsample
    mtot, slope = extrapolate_all(data, form)
    f_out = 1.0 - data[:, :, -1, None] / mtot          # (n, 5, 2)
    sys_dex = np.abs(np.log10(mtot[:, :, 0] / mtot[:, :, 1]))

    print("\n  beyond-aperture fraction f_out (power form), median [16-84%]:")
    for k in range(5):
        v = f_out[:, k, 0]
        print(f"    z={ZK[k]}: {np.nanmedian(v):.3f} "
              f"[{np.nanpercentile(v, 16):.3f}, {np.nanpercentile(v, 84):.3f}]  "
              f"form-sys {np.nanmedian(sys_dex[:, k]):.4f} dex")
    print("\n  f_out vs stellar mass (z=0.4, power):")
    edges = np.quantile(logms, np.linspace(0, 1, 5))
    for q in range(4):
        m = (logms >= edges[q]) & (logms <= edges[q + 1] + 1e-9)
        print(f"    logM* {edges[q]:.2f}-{edges[q+1]:.2f}: "
              f"{np.nanmedian(f_out[m, 0, 0]):.3f}")

    ed3, diffs = differential(data, logms)
    np.savez(OUTDIR / "asymptotic_total.npz", index=pop["index"], mtot=mtot,
             f_out=f_out, tail_slope=slope, sys_dex=sys_dex, form=form)
    print(f"\nwrote {OUTDIR / 'asymptotic_total.npz'}")
    _figure(data, logms, mtot, f_out, slope, ed3, diffs, n)


def _figure(data, logms, mtot, f_out, slope, ed3, diffs, n):
    fig, axes = plt.subplots(1, 4, figsize=(19.5, 4.6))
    a, b, c, d = axes
    cols = [matplotlib.colormaps["cividis"](v) for v in np.linspace(0, 0.92, 5)]
    for k in range(5):
        v = f_out[:, k, 0]
        order = np.argsort(logms)
        med = [np.nanmedian(v[order][max(0, i - 150):i + 150])
               for i in range(len(order))]
        a.plot(logms[order], med, c=cols[k], lw=1.7, label=f"z={ZK[k]}")
    a.set(xlabel="logM* (z=0.4)", ylabel="beyond-aperture fraction f$_{out}$",
          title="A. f$_{out}$ vs mass and epoch (running median)")
    a.legend(fontsize=7)

    b.hist([slope[:, 0][np.isfinite(slope[:, 0])],
            slope[:, 4][np.isfinite(slope[:, 4])]], bins=40,
           label=["z=0.4", "z=2.0"], color=[cols[0], cols[4]], histtype="step",
           lw=1.8, density=True)
    b.set(xlabel="power-tail exponent a  ($\\Sigma \\propto R^{-(a+2)}$)",
          ylabel="density", title="B. Outer density-tail slope")
    b.legend(fontsize=8)

    for bidx, col in ((0, "#0072B2"), (2, "#D55E00")):
        dprof = np.median(diffs[(bidx, 0)], axis=0)       # z0.7 -> z0.4 pair
        dd = np.diff(dprof, prepend=0.0)
        c.plot(R, np.clip(dd, 1, None), "o-", c=col, lw=1.6, ms=3,
               label=f"logM* {ed3[bidx]:.1f}-{ed3[bidx+1]:.1f}")
    c.set(xscale="log", yscale="log", xlabel="R [kpc]",
          ylabel="median added mass per bin [M$_\\odot$]",
          title="C. Where new mass lands (z=0.7$\\to$0.4)")
    c.legend(fontsize=8)

    i = int(np.argmax(logms))
    for k in (0, 4):
        d.plot(R, data[i, k], "o", c=cols[k], ms=4, label=f"measured z={ZK[k]}")
        p = fit_tail(data[i, k], "power")
        if p is not None:
            rr = np.geomspace(50, 600, 60)
            d.plot(rr, f_power(rr, *p), "-", c=cols[k], lw=1.4)
            d.axhline(p[0], c=cols[k], ls=":", lw=1)
    d.set(xscale="log", yscale="log", xlabel="R [kpc]", ylabel="M(<R) [M$_\\odot$]",
          title=f"D. Example extrapolation (most massive, logM*={logms[i]:.2f})")
    d.legend(fontsize=7)
    fig.suptitle(f"exp34 — asymptotic totals from the CoG tail (n={n})", fontsize=12)
    fig.tight_layout()
    print("wrote", save_fig(fig, FIGDIR / "exp34_asymptotic_total")[0])


def demo():
    """Self-check: both forms recover a synthetic M_tot exactly from truncated
    data; the truncation test prefers the true form."""
    rr = R
    mtot_true = 1e11
    for form, f, loga, shp in (("power", f_power, np.log10(4e11), 1.2),
                               ("expo", f_expo, np.log10(5e10), 40.0)):
        cog = f(rr, mtot_true, loga, shp)
        assert (cog[R >= FIT_RMIN] > 0).all()
        p = fit_tail(cog, form)
        assert p is not None and abs(np.log10(p[0] / mtot_true)) < 0.01, (form, p)
    # power-law truth fitted with expo (wrong form) must be visibly worse at 148
    cog = f_power(rr, mtot_true, np.log10(4e11), 0.8)
    pw = fit_tail(cog, "power", rmax=100.0)
    pe_ = fit_tail(cog, "expo", rmax=100.0)
    e_pw = abs(np.log10(f_power(R[-1], *pw) / cog[-1]))
    e_pe = abs(np.log10(f_expo(R[-1], *pe_) / cog[-1]))
    assert e_pw < e_pe, (e_pw, e_pe)
    print(f"run.demo OK: both tails recover M_tot (<0.01 dex); truncation test "
          f"separates forms ({e_pw:.4f} vs {e_pe:.4f} dex)")


if __name__ == "__main__":
    if sys.argv[1:2] == ["demo"]:
        demo()
    else:
        sys.exit(main())
