"""exp30 phase 3 v1 — the parametric-mass transport emulator.

The LOEO test (holdout.py) showed free NNLS masses are a generalization liability
(dyntrans: 7.5% in-sample -> 53.7% held-out). This replaces them with the parametric
mass law: dM*_i = f(z_i) * dMh_i, f = the two-epoch broken power-law efficiency.
Combined with the dyntrans transport structure (dynamical clock + multi-scale width):

  params (7, ZERO free masses): log10 sigma_0, g, log10 alpha, q, b_early, b_late, z_c
  shape from the model; amplitude per-epoch aperture-pinned (the SHMR's job); the
  native mass-growth prediction (normalized to z=0.4) is reported separately.

Tests:
  1. In-sample joint fit (all radii) vs {additive, free-mass dyntrans, loose-quad}.
  2. FULL LOEO -- now including the z=0.4 holdout that free-mass models structurally
     could not attempt (their late-deposit masses were unconstrained). Success =
     held-out ~ in-sample (small gap), beating additive's 30.9% held-avg.
  3. The v2 gate: do per-galaxy residuals correlate with MAH burstiness? If the
     single-channel model fails preferentially for merger-rich galaxies, the ex-situ
     dual-region channel (phase 3 v2) is warranted.

Run: PYTHONPATH=. uv run python experiments/exp30_transport_kernel/param_emulator.py [n] [--refit]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP29 = ROOT / "experiments" / "exp29_outer_deposit"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP29))
import transport_floor as tf                                                          # noqa: E402
from deposit import eff_two_epoch                                                    # noqa: E402
from run import ANCHOR_SNAP, ANCHOR_Z                                                # noqa: E402
from real_mah_test import real_mah                                                   # noqa: E402
import mass_qa                                                                        # noqa: E402
from hongshao.plotting import set_style, save_fig                                    # noqa: E402

set_style()
FIGDIR, OUTDIR = HERE / "figures", HERE / "outputs"
OUT_NPZ = OUTDIR / "param_emulator.npz"
R, AT = tf.R, tf.AT
Z = np.array(ANCHOR_Z)
HOLDOUTS = [0, 1, 2, 3, 4]                                 # z=0.4 holdout now possible


def model_cogs(theta, mah, data):
    """7-param parametric-mass dyntrans CoGs at all 5 epochs, per-epoch pinned.
    Returns (pinned_cogs (5,24), native_totals (5,)) or (None, None)."""
    if theta[6] <= -1.0:
        return None, None
    with np.errstate(over="ignore", invalid="ignore"):
        w = eff_two_epoch(mah["z"], theta[4], theta[5], theta[6])
    dM = w * mah["dMh"]
    if not np.isfinite(dM).all() or dM.sum() <= 0:
        return None, None
    dM = dM / dM.sum()
    masks = [mah["snap"] <= sa for sa in ANCHOR_SNAP]
    cogs, tot = [], []
    for k in range(5):
        B = tf.basis(theta[:4], mah["t"], mah["t_obs"], AT[k], "dyntrans")
        m = B @ (dM * masks[k])
        if not np.isfinite(m[-1]) or m[-1] <= 0:
            return None, None
        tot.append(m[-1])
        cogs.append(m * (data[k][-1] / m[-1]))
    return np.array(cogs), np.array(tot)


STARTS = [[2.0, 1.5, la, 1.5, be, bl, zc]
          for la in (np.log10(0.3), 0.0)
          for be, bl, zc in ((4.0, 1.5, 2.5), (5.5, 0.8, 3.0), (3.0, 2.0, 1.5))]


def fit(mah, data, fit_ks=(0, 1, 2, 3, 4), starts=STARTS):
    """Joint fit on ``fit_ks`` epochs; mean per-epoch rel-RMS over ALL radii."""
    def loss(th):
        cogs, _ = model_cogs(th, mah, data)
        if cogs is None:
            return 1e3
        v = np.mean([np.sqrt(np.mean(((cogs[k] - data[k]) / data[k]) ** 2)) for k in fit_ks])
        return v if np.isfinite(v) else 1e3

    best = None
    for p0 in starts:
        r = minimize(loss, p0, method="Nelder-Mead",
                     options=dict(maxiter=14000, xatol=1e-4, fatol=1e-10))
        if best is None or r.fun < best.fun:
            best = r
    cogs, tot = model_cogs(best.x, mah, data)
    return cogs, tot, best.x


def burstiness(mah):
    s = mah["dMh"] / 10.0 ** mah["logMh_full"][1:]
    return mah["dMh"][s > 0.10].sum() / mah["dMh"].sum()


def compute(n):
    d = np.load(HERE / "outputs" / "transport_floor.npz")
    idx, logms, datas = d["index"][:n], d["logms"][:n], d["data"][:n]
    ng = len(idx)
    cogs_in = np.zeros((ng, 5, 24)); pars = np.zeros((ng, 7)); tots = np.zeros((ng, 5))
    held = np.full((ng, 5), np.nan); burst = np.zeros(ng)
    for i in range(ng):
        mah = real_mah(int(idx[i]))
        data = [datas[i][k] for k in range(5)]
        burst[i] = burstiness(mah)
        ci, ti_, th = fit(mah, data)
        cogs_in[i], tots[i], pars[i] = ci, ti_, th
        for h in HOLDOUTS:                              # LOEO, fresh starts (no leakage)
            ch, _, _ = fit(mah, data, fit_ks=[k for k in range(5) if k != h],
                           starts=STARTS[:4])
            if ch is not None:
                held[i, h] = np.abs((ch[h] - data[h]) / data[h]).max()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, index=idx, logms=logms, data=datas, cogs=cogs_in, params=pars,
             totals=tots, held=held, burst=burst)
    print(f"wrote {OUT_NPZ}  (n={ng})")
    return np.load(OUT_NPZ)


def main():
    refit = "--refit" in sys.argv
    n = next((int(a) for a in sys.argv[1:] if a.isdigit()), 45)
    d = compute(n) if (refit or not OUT_NPZ.exists()) else np.load(OUT_NPZ)
    datas, ng = d["data"], len(d["index"])
    dtf = np.load(HERE / "outputs" / "transport_floor.npz")

    mr_in = np.array([tf.maxrel(d["cogs"][i], datas[i]) for i in range(ng)])
    mr_dy = np.array([tf.maxrel(dtf["dyntrans"][i], datas[i]) for i in range(ng)])
    mr_ad = np.array([tf.maxrel(dtf["additive"][i], datas[i]) for i in range(ng)])
    avg = lambda m: 100 * np.median([np.median(m[:, k]) for k in range(5)])

    print(f"\nexp30 phase 3 v1 — parametric-mass transport emulator (n={ng}, 7 params, "
          "0 free masses)\n")
    print("  (1) IN-SAMPLE, median max|rel| per epoch (all radii):")
    print(f"    {'model':>16s} | " + " | ".join(f"z={z}".rjust(6) for z in ANCHOR_Z) + " |  avg")
    for nm, m in (("param-emulator", mr_in), ("dyntrans (free)", mr_dy), ("additive", mr_ad)):
        print(f"    {nm:>16s} | " + " | ".join(f"{100*np.median(m[:,k]):5.1f}%" for k in range(5)) +
              f" | {avg(m):4.1f}%")
    p = d["params"]
    print(f"\n    fitted params, median: alpha {np.median(10**p[:,2]):.2f}, q {np.median(p[:,3]):.2f}, "
          f"b_early {np.median(p[:,4]):.2f}, b_late {np.median(p[:,5]):.2f}, z_c {np.median(p[:,6]):.2f}")

    print("\n  (2) LOEO held-out, median max|rel| (param-emulator; gap vs its in-sample):")
    print(f"    {'z out':>8s} | " + " | ".join(f"z={Z[h]}".rjust(6) for h in HOLDOUTS))
    print(f"    {'held':>8s} | " + " | ".join(f"{100*np.nanmedian(d['held'][:,h]):5.1f}%"
                                              for h in HOLDOUTS))
    print(f"    {'in-samp':>8s} | " + " | ".join(f"{100*np.median(mr_in[:,h]):5.1f}%"
                                                 for h in HOLDOUTS))
    ho_avg = 100 * np.mean([np.nanmedian(d["held"][:, h]) for h in HOLDOUTS])
    gap = ho_avg - avg(mr_in)
    print(f"    held-avg {ho_avg:.1f}%  (in-sample avg {avg(mr_in):.1f}%, gap {gap:+.1f})   "
          "[2.3 refs, no z=0.4: additive 30.9%, loose 35.3%, dyntrans-free 53.7%]")

    # native mass growth (shape-independent check): dlog of predicted total vs data,
    # normalized at z=0.4
    dm = np.log10(d["totals"] / d["totals"][:, [0]]) - \
        np.log10(datas[:, :, -1] / datas[:, [0], -1])
    print("\n  (3) native mass-growth prediction |dlog M*(z_k)/M*(z=0.4)|, median:")
    print("    " + "  ".join(f"z={Z[k]}: {np.nanmedian(np.abs(dm[:,k])):.3f}" for k in range(1, 5)))

    ga_in = mr_in.mean(1); ga_ho = np.nanmean(d["held"], axis=1)
    r_in, p_in = spearmanr(ga_in, d["burst"])
    r_ho, p_ho = spearmanr(ga_ho, d["burst"])
    print(f"\n  (4) v2 gate — residual vs MAH burstiness: in-sample rho={r_in:+.2f} (p={p_in:.3f}), "
          f"LOEO rho={r_ho:+.2f} (p={p_ho:.3f})")

    print(f"\n[verdict] in-sample {avg(mr_in):.1f}%, held-out {ho_avg:.1f}% (gap {gap:+.1f})")
    if ho_avg < 30.9 and gap < 15.0:
        print("  the 7-param emulator PREDICTS: held-out beats every 2.3 model and the gap is\n"
              "  small -> parametric masses fixed the generalization failure. Proceed to the\n"
              "  population step (predict the 7 params from halo-only inputs)" +
              (";\n  the burstiness correlation says single-channel residuals are merger-linked ->\n"
               "  build v2 (dual-region ex-situ channel) first." if (p_ho < 0.05 and r_ho > 0)
               else "; v2 gate stays closed\n  (no significant burstiness correlation)."))
    elif ho_avg < 30.9:
        print("  held-out beats the 2.3 models but the gap is still large -> inspect which\n"
              "  epochs/galaxies drive it before the population step.")
    else:
        print("  the parametric-mass emulator does NOT beat the rigid additive baseline\n"
              "  out-of-sample -> the efficiency form is too restrictive; revisit before phase 4.")
    _figure(d, mr_in, mr_dy, mr_ad, ga_in, ga_ho, r_ho, p_ho)
    mass_qa.evaluate(d["cogs"], datas, R, ANCHOR_Z, name="param-emulator", figdir=FIGDIR)


def _figure(d, mr_in, mr_dy, mr_ad, ga_in, ga_ho, r_ho, p_ho):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    a, b, c = axes
    x = np.arange(5)
    for nm, m, col in (("param-emulator", mr_in, "#CC3377"), ("dyntrans (free-mass)", mr_dy, "#56B4E9"),
                       ("additive", mr_ad, "#E69F00")):
        a.plot(x, [100 * np.median(m[:, k]) for k in range(5)], "o-", c=col, lw=2, label=nm)
    a.set_xticks(x); a.set_xticklabels([f"{z}" for z in ANCHOR_Z])
    a.set(xlabel="epoch z", ylabel="median max|rel| [%]", ylim=(0, None),
          title="A. In-sample: 7 params vs free masses")
    a.legend(fontsize=8)

    b.plot(x, [100 * np.nanmedian(d["held"][:, h]) for h in HOLDOUTS], "o-", c="#CC3377", lw=2,
           label="param-emulator held-out")
    b.plot(x, [100 * np.median(mr_in[:, h]) for h in HOLDOUTS], "o--", c="#CC3377", lw=1.2,
           alpha=0.5, label="param-emulator in-sample")
    b.axhline(30.9, c="#E69F00", ls=":", lw=1.4, label="additive held-avg (2.3)")
    b.axhline(53.7, c="#56B4E9", ls=":", lw=1.4, label="dyntrans-free held-avg (2.3)")
    b.set_xticks(x); b.set_xticklabels([f"{z}" for z in ANCHOR_Z])
    b.set(xlabel="held-out epoch", ylabel="median max|rel| [%]", ylim=(0, None),
          title="B. LOEO incl. the z=0.4 holdout")
    b.legend(fontsize=7)

    c.scatter(d["burst"], 100 * ga_ho, s=30, c="#CC3377", edgecolor="0.3", lw=0.4)
    c.set(xlabel="MAH burstiness (growth frac in steps >10%)",
          ylabel="per-galaxy LOEO epoch-avg max|rel| [%]",
          title=f"C. v2 gate: residual vs burstiness\nSpearman $\\rho$={r_ho:+.2f} (p={p_ho:.3f})")
    fig.suptitle("exp30 phase 3 v1 — parametric-mass transport emulator (7 params, 0 free masses)",
                 fontsize=12)
    fig.tight_layout()
    print("\nwrote", save_fig(fig, FIGDIR / "exp30_param_emulator")[0])


def demo():
    """Self-check: model builds, monotonic, pinned; a quick fit is sane."""
    dtf = np.load(HERE / "outputs" / "transport_floor.npz")
    i = 2
    mah = real_mah(int(dtf["index"][i]))
    data = [dtf["data"][i][k] for k in range(5)]
    cogs, tot = model_cogs([2.0, 1.5, 0.0, 1.5, 4.0, 1.5, 2.5], mah, data)
    assert cogs is not None and np.isfinite(cogs).all()
    assert np.all(np.diff(cogs, axis=1) >= -1e-9), "CoG must be monotonic"
    for k in range(5):
        assert abs(cogs[k][-1] / data[k][-1] - 1) < 1e-9, "must be aperture-pinned"
    ci, _, th = fit(mah, data, starts=STARTS[:2])
    mx = tf.maxrel(ci, data).mean()
    assert mx < 0.35, ("fit unexpectedly poor", mx)
    print(f"param_emulator.demo OK: monotonic pinned CoGs; quick-fit mean max|rel| {100*mx:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        sys.exit(main())
