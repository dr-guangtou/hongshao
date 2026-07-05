"""exp30 — transport-kernel feasibility gate: does core-retaining redistribution
break the additive consistency ceiling?

Each deposit keeps a core fraction f_core(dt)=exp(-dt/tau) at its deposition width and
migrates the rest to sigma_w = sigma_0,i (t_k/t_i)^q. tau->inf recovers the exp29
additive floor exactly; tau->0 recovers pure ratio-law puffing. The CoG stays linear
in the free deposit masses -> convex NNLS inner solve; (sigma_0, g, tau, q) outer.

Three modes, SAME method/metric/galaxies (real de-dipped MAH, ALL radii):
  alone     : per-epoch free-mass NNLS (ceiling)
  additive  : joint, f_core==1 (the exp29 floor)
  transport : joint, (tau, q) free

Run: PYTHONPATH=. uv run python experiments/exp30_transport_kernel/transport_floor.py [n] [--refit]
Demo: ... transport_floor.py demo
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, nnls

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP29 = ROOT / "experiments" / "exp29_outer_deposit"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP29))
from run import ANCHOR_SNAP, ANCHOR_Z                                                # noqa: E402
from real_mah_test import real_mah                                                   # noqa: E402
import mass_qa                                                                        # noqa: E402
from hongshao.tng_data import COG_RAD_KPC, load_cosmic_time                          # noqa: E402
from hongshao.plotting import set_style, save_fig                                    # noqa: E402

set_style()
FIGDIR, OUTDIR = HERE / "figures", HERE / "outputs"
IN_NPZ = EXP29 / "outputs" / "integrated_check.npz"          # corrected-standard sample
SCORE_NPZ = EXP29 / "outputs" / "final_scorecard.npz"        # loose-quad reference
OUT_NPZ = OUTDIR / "transport_floor.npz"
R = COG_RAD_KPC
AT = load_cosmic_time()[ANCHOR_SNAP]                          # cosmic time at the 5 epochs
MODES = ["alone", "additive", "transport", "envelope", "combined", "dyntrans"]
COLORS = {"alone": "0.45", "additive": "#E69F00", "transport": "#009E73",
          "envelope": "#CC3377", "combined": "#D55E00", "dyntrans": "#56B4E9",
          "loose-quad": "#0072B2"}


def basis(theta, ti, t_obs, tk, mode):
    """(24, N) unit-mass CoG basis for deposits `ti` observed at `tk`.

    transport: f_core = exp(-dt/tau) global clock; migrated width = sig0_i (tk/ti)^q.
    envelope : f_core = exp(-dt/(alpha ti)) dynamical clock (early deposits migrate
               fast, matching front-loaded merging); migrated width = w0 (tk/t_obs)^gw
               set by the halo at the OBSERVATION time, not the (tiny) birth width.
    combined : two-parameter clock tau_i = tau0 + alpha*ti nesting BOTH the global
               (alpha->0) and dynamical (tau0->0) clocks, + the observation-epoch
               width both variants converged on. (Core floor f_min deferred.)"""
    s0, g = 10.0 ** theta[0], theta[1]
    sig0 = np.clip(s0 * (ti / t_obs) ** g, 1e-4, 1e5)    # clip: underflow*overflow -> NaN
    dt = np.clip(tk - ti, 0.0, None)
    if mode == "transport":
        fc = np.exp(-dt / 10.0 ** theta[2])
        sigw = np.clip(sig0 * (tk / ti) ** max(theta[3], 0.0), 1e-4, 1e5)
    elif mode == "envelope":
        fc = np.exp(-dt / (10.0 ** theta[2] * ti))
        sigw = np.full_like(ti, np.clip(10.0 ** theta[3] * (tk / t_obs) ** theta[4], 1e-4, 1e5))
    elif mode == "combined":
        fc = np.exp(-dt / (10.0 ** theta[2] + 10.0 ** theta[3] * ti))
        sigw = np.full_like(ti, np.clip(10.0 ** theta[4] * (tk / t_obs) ** theta[5], 1e-4, 1e5))
    elif mode == "dyntrans":                             # dynamical clock + multi-scale width
        fc = np.exp(-dt / (10.0 ** theta[2] * ti))
        sigw = np.clip(sig0 * (tk / ti) ** max(theta[3], 0.0), 1e-4, 1e5)
    else:                                                # additive
        fc, sigw = np.ones_like(ti), sig0
    core = 1.0 - np.exp(-R[:, None] ** 2 / (2.0 * sig0[None, :] ** 2))
    wide = 1.0 - np.exp(-R[:, None] ** 2 / (2.0 * sigw[None, :] ** 2))
    return fc[None, :] * core + (1.0 - fc)[None, :] * wide


def solve_joint(theta, mah, data, mode):
    """Weighted joint NNLS over all 5 epochs; returns (cogs (5,24), rnorm)."""
    ti, snap, t_obs = mah["t"], mah["snap"], mah["t_obs"]
    masks = [snap <= sa for sa in ANCHOR_SNAP]
    blocks = [basis(theta, ti, t_obs, AT[k], mode) * masks[k][None, :] for k in range(5)]
    A = np.vstack([b / data[k][:, None] for k, b in enumerate(blocks)])
    if not np.isfinite(A).all():
        return None, np.inf
    x, rnorm = nnls(A, np.ones(A.shape[0]), maxiter=10 * A.shape[1])
    return np.array([b @ x for b in blocks]), rnorm


def fit_joint(mah, data, mode, warm=None):
    """Outer Nelder-Mead over the width/transport params, NNLS inside."""
    def loss(th):
        return solve_joint(th, mah, data, mode)[1]

    if mode == "additive":
        starts = [[s0l, g] for s0l in (1.8, 2.3) for g in (1.0, 1.7)]
    elif mode == "transport":                         # warm (s0, g) from the additive fit
        starts = [[warm[0], warm[1], lt, q]
                  for lt in (np.log10(0.5), np.log10(3.0), np.log10(10.0)) for q in (0.7, 1.8)]
    elif mode == "envelope":                          # envelope: alpha, w0 [kpc], gw
        starts = [[warm[0], warm[1], la, lw, gw]
                  for la in (np.log10(0.3), np.log10(2.0)) for lw in (1.7, 2.0) for gw in (0.3, 1.0)]
    elif mode == "combined":                          # warm = (tr_th, en_th)
        tr, en = warm
        starts = [
            [en[0], en[1], -1.3, en[2], en[3], en[4]],           # ~pure dynamical (envelope)
            [tr[0], tr[1], tr[2], -2.0, tr[0], max(tr[3], 0.1)],  # ~pure global (transport-ish)
            [en[0], en[1], tr[2], en[2], en[3], en[4]],          # transport clock + envelope rest
        ]
    else:                                             # dyntrans: warm = (tr_th, en_th)
        tr, en = warm
        starts = [[tr[0], tr[1], la, max(tr[3], 0.1)]
                  for la in (np.log10(0.3), np.log10(2.0), np.log10(8.0))] + \
                 [[tr[0], tr[1], en[2], 1.0]]                    # envelope's fitted clock
    best = None
    for p0 in starts:
        r = minimize(loss, p0, method="Nelder-Mead",
                     options=dict(maxiter=2000 * len(p0), xatol=1e-4, fatol=1e-10))
        if best is None or r.fun < best.fun:
            best = r
    return solve_joint(best.x, mah, data, mode)[0], best.x


def fit_alone(mah, data):
    """Per-epoch free-mass NNLS (own width law per epoch) — the ceiling."""
    ti, snap, t_obs = mah["t"], mah["snap"], mah["t_obs"]
    cogs = []
    for k in range(5):
        m = snap <= ANCHOR_SNAP[k]

        def loss(th):
            B = basis(th, ti[m], t_obs, AT[k], "additive")
            x, rn = nnls(B / data[k][:, None], np.ones(len(R)), maxiter=10 * int(m.sum()))
            return rn

        best = None
        for p0 in ([1.8, 1.0], [2.3, 1.7], [2.6, 1.7]):
            r = minimize(loss, p0, method="Nelder-Mead",
                         options=dict(maxiter=1500, xatol=1e-4, fatol=1e-10))
            if best is None or r.fun < best.fun:
                best = r
        B = basis(best.x, ti[m], t_obs, AT[k], "additive")
        x, _ = nnls(B / data[k][:, None], np.ones(len(R)), maxiter=10 * int(m.sum()))
        cogs.append(B @ x)
    return np.array(cogs)


def maxrel(cogs, data):
    return np.array([np.abs((cogs[k] - data[k]) / data[k]).max() for k in range(5)])


def compute(n):
    d = np.load(IN_NPZ)
    idx, logms, datas = d["index"][:n], d["logms"][:n], d["data"][:n]
    out = {m: [] for m in MODES}; ptr, pen, pco, pdy = [], [], [], []
    for i in range(len(idx)):
        mah = real_mah(int(idx[i]))
        data = [datas[i][k] for k in range(5)]
        out["alone"].append(fit_alone(mah, data))
        add_cogs, add_th = fit_joint(mah, data, "additive")
        out["additive"].append(add_cogs)
        tr_cogs, tr_th = fit_joint(mah, data, "transport", warm=add_th)
        out["transport"].append(tr_cogs); ptr.append(tr_th)
        en_cogs, en_th = fit_joint(mah, data, "envelope", warm=add_th)
        out["envelope"].append(en_cogs); pen.append(en_th)
        co_cogs, co_th = fit_joint(mah, data, "combined", warm=(tr_th, en_th))
        out["combined"].append(co_cogs); pco.append(co_th)
        dy_cogs, dy_th = fit_joint(mah, data, "dyntrans", warm=(tr_th, en_th))
        out["dyntrans"].append(dy_cogs); pdy.append(dy_th)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, index=idx, logms=logms, data=datas,
             params_transport=np.array(ptr), params_envelope=np.array(pen),
             params_combined=np.array(pco), params_dyntrans=np.array(pdy),
             **{m: np.array(v) for m, v in out.items()})
    print(f"wrote {OUT_NPZ}  (n={len(idx)})")
    return np.load(OUT_NPZ)


def main():
    refit = "--refit" in sys.argv
    n = next((int(a) for a in sys.argv[1:] if a.isdigit()), 45)
    d = compute(n) if (refit or not OUT_NPZ.exists()) else np.load(OUT_NPZ)
    logms, datas = d["logms"], d["data"]
    ptr, pen, pco = d["params_transport"], d["params_envelope"], d["params_combined"]
    ng = len(logms)
    mr = {m: np.array([maxrel(d[m][i], datas[i]) for i in range(ng)]) for m in MODES}
    avg = lambda m: 100 * np.median([np.median(mr[m][:, k]) for k in range(5)])

    loose = None
    if SCORE_NPZ.exists():                               # loose-quad reference, same sample
        s = np.load(SCORE_NPZ, allow_pickle=True)
        loo = np.array([maxrel(s["models"][i, 1], s["data"][i]) for i in range(len(s["data"]))])
        loose = [100 * np.median(loo[:, k]) for k in range(5)]

    print(f"\nexp30 — transport gate (real MAH, ALL radii, n={ng}), median max|rel| per epoch\n")
    print(f"  {'mode':>10s} | " + " | ".join(f"z={z}".rjust(6) for z in ANCHOR_Z) + " |  avg")
    for m in MODES:
        print(f"  {m:>10s} | " + " | ".join(f"{100*np.median(mr[m][:,k]):5.1f}%" for k in range(5)) +
              f" | {avg(m):4.1f}%")
    if loose is not None:
        print(f"  {'loose-quad':>10s} | " + " | ".join(f"{v:5.1f}%" for v in loose) +
              f" | {np.median(loose):4.1f}%  (exp29 reference)")

    tau, q = 10.0 ** ptr[:, 2], np.clip(ptr[:, 3], 0.0, None)
    al, w0, gw = 10.0 ** pen[:, 2], 10.0 ** pen[:, 3], pen[:, 4]
    print(f"\n  transport params: tau median {np.median(tau):.1f} Gyr, q median {np.median(q):.2f}")
    print(f"  envelope  params: alpha median {np.median(al):.2f} (tau = alpha*t_i), "
          f"w0 median {np.median(w0):.0f} kpc, gw median {np.median(gw):.2f}")
    ct0, cal = 10.0 ** pco[:, 2], 10.0 ** pco[:, 3]
    cw0, cgw = 10.0 ** pco[:, 4], pco[:, 5]
    print(f"  combined  params: tau0 median {np.median(ct0):.2f} Gyr, alpha median "
          f"{np.median(cal):.2f} (tau_i = tau0 + alpha*t_i), w0 median {np.median(cw0):.0f} kpc, "
          f"gw median {np.median(cgw):.2f}")
    pdy_ = d["params_dyntrans"]
    print(f"  dyntrans  params: alpha median {np.median(10.0 ** pdy_[:, 2]):.2f} "
          f"(tau_i = alpha*t_i), q median {np.median(np.clip(pdy_[:, 3], 0, None)):.2f}")

    a_d, a_c, a_e, a_t = avg("dyntrans"), avg("combined"), avg("envelope"), avg("transport")
    a_a, a_al = avg("additive"), avg("alone")
    cand = [(v, nm) for v, nm in ((a_d, "dyntrans"), (a_c, "combined"), (a_e, "envelope"),
                                  (a_t, "transport")) if np.isfinite(v)]
    a_best, best_name = min(cand)
    print(f"\n[gate] additive floor {a_a:.1f}%  ->  transport {a_t:.1f}%  |  envelope {a_e:.1f}%  |  "
          f"combined {a_c:.1f}%  |  dyntrans {a_d:.1f}%   "
          f"(ceiling {a_al:.1f}%" + (f", loose-quad {np.median(loose):.1f}%" if loose else "") + ")")
    if a_best < 0.5 * a_a and (loose is None or a_best < np.median(loose)):
        print(f"  core-retaining redistribution ({best_name}) BREAKS the additive ceiling and beats the\n"
              "  loose-zdep reference with a consistent history -> redistribution is the missing\n"
              "  freedom. Next: event-driven kicks at the real-MAH bursts, then the population model.")
    elif a_best < 0.85 * a_a:
        print(f"  {best_name} helps but does not decisively break the ceiling -> try event-triggered\n"
              "  kicks / a different clock or envelope form before concluding.")
    else:
        print("  neither redistribution form improves on the additive floor -> stop before\n"
              "  over-investing; the limit is deeper than this transport family.")
    _figure(logms, datas, d, mr, loose)
    mass_qa.evaluate(d[best_name], datas, R, ANCHOR_Z, name=best_name, figdir=FIGDIR)


def _figure(logms, datas, d, mr, loose):
    x = np.arange(5)
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for m in MODES:
        a.plot(x, [100 * np.median(mr[m][:, k]) for k in range(5)], "o-", c=COLORS[m], lw=2,
               ms=6, label=m)
    if loose is not None:
        a.plot(x, loose, "s--", c=COLORS["loose-quad"], lw=1.6, ms=5, label="loose-quad (exp29)")
    a.set_xticks(x); a.set_xticklabels([f"{z}" for z in ANCHOR_Z])
    a.set(xlabel="epoch z", ylabel="median profile max|rel| [%] (all radii)", ylim=(0, None),
          title="A. Transport vs additive floor vs ceiling")
    a.legend(fontsize=8)

    cmap = matplotlib.colormaps["viridis"]                # BCG = hardest case, index 0
    for k in range(5):
        c = cmap(k / 4)
        b.plot(R, 100 * (d["combined"][0, k] - datas[0, k]) / datas[0, k], "-", c=c, lw=1.7,
               label=f"z={ANCHOR_Z[k]}")
        b.plot(R, 100 * (d["transport"][0, k] - datas[0, k]) / datas[0, k], ":", c=c, lw=1.4)
    b.axhline(0, c="0.6", lw=0.8)
    b.set(xscale="log", xlabel="R [kpc]", ylabel="(model$-$data)/data [%]", ylim=(-25, 25),
          title=f"B. BCG (logM*={logms[0]:.2f}): combined (solid) vs transport (dotted)")
    b.legend(fontsize=7)
    fig.suptitle("exp30 — core-retaining transport kernel: joint 5-epoch free-mass fit "
                 "(real MAH, all radii)", fontsize=12)
    fig.tight_layout()
    print("\nwrote", save_fig(fig, FIGDIR / "exp30_transport_gate")[0])


def demo():
    """Self-check: tau->inf reduces to additive; basis mass-conserving & monotonic;
    transport (nesting additive) never fits worse than additive."""
    d = np.load(IN_NPZ)
    gi, data = int(d["index"][2]), [d["data"][2][k] for k in range(5)]
    mah = real_mah(gi)
    th_a, th_t = [2.0, 1.5], [2.0, 1.5, 9.0, 1.0]        # tau = 1e9 Gyr ~ inf
    Ba = basis(th_a, mah["t"], mah["t_obs"], AT[0], "additive")
    Bt = basis(th_t, mah["t"], mah["t_obs"], AT[0], "transport")
    assert np.allclose(Ba, Bt, rtol=1e-6), "tau->inf must recover the additive basis"
    big = basis(th_t, mah["t"], mah["t_obs"], AT[0], "transport")
    assert np.all(np.diff(big, axis=0) >= -1e-12), "basis CoG must be monotonic in R"
    th_env = [2.0, 1.5, np.log10(0.3), 1.7, 0.7]
    th_co_env = [2.0, 1.5, -9.0, np.log10(0.3), 1.7, 0.7]     # tau0 -> 0: pure dynamical
    th_co_add = [2.0, 1.5, 9.0, -9.0, 1.7, 0.7]               # tau_i -> inf: fc ~ 1
    Be = basis(th_env, mah["t"], mah["t_obs"], AT[0], "envelope")
    assert np.allclose(basis(th_co_env, mah["t"], mah["t_obs"], AT[0], "combined"), Be,
                       rtol=1e-5), "combined(tau0->0) must recover the envelope basis"
    assert np.allclose(basis(th_co_add, mah["t"], mah["t_obs"], AT[0], "combined"), Ba,
                       atol=1e-6), "combined(tau_i->inf) must recover the additive basis"
    th_dy_add = [2.0, 1.5, 9.0, 1.0]                          # alpha -> inf: fc ~ 1
    assert np.allclose(basis(th_dy_add, mah["t"], mah["t_obs"], AT[0], "dyntrans"), Ba,
                       atol=1e-6), "dyntrans(alpha->inf) must recover the additive basis"
    th_t2 = [2.0, 1.5, np.log10(2.0), 1.0]
    ca, ta = fit_joint(mah, data, "additive")
    ct, tt = fit_joint(mah, data, "transport", warm=ta)
    ce, te = fit_joint(mah, data, "envelope", warm=ta)
    cc, _ = fit_joint(mah, data, "combined", warm=(tt, te))
    assert solve_joint(th_t2, mah, data, "transport")[1] > 0
    assert maxrel(ct, data).mean() <= maxrel(ca, data).mean() * 1.10, \
        "transport (nests additive) should not fit meaningfully worse"
    assert maxrel(cc, data).mean() <= min(maxrel(ct, data).mean(), maxrel(ce, data).mean()) * 1.10, \
        "combined (warm-started from both variants) should not fit meaningfully worse than either"
    print("transport_floor.demo OK: tau->inf == additive, combined nests envelope+additive, "
          "monotonic basis, nesting holds")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        sys.exit(main())
