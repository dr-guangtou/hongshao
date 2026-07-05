"""exp30 phase 2.2 — event-triggered kicks: mergers drive the migration clock.

dyntrans (7.5%, the 2x2 winner) migrates mass on a SMOOTH self-similar clock
tau_i = alpha*t_i. Physically, redistribution is driven by discrete mergers. The real
de-dipped MAH carries those events: each peak-history step s_j = dMh_j / Mh(t_j) is
net growth above the previous peak (dip-recovery is discounted by construction).

Model: keep dyntrans's proven multi-scale migrated width sig0_i (t_k/t_i)^q; replace
the smooth clock with a kick product over the events between deposition and observation:

    f_core,i(t_k) = prod_{t_i < t_j <= t_k} (1 - eps0 * s_j)      (clipped to [1e-3, 1])

eps0 -> 0 recovers the additive model; many small events -> the smooth-clock limit.
Same parameter count as dyntrans: (log10 sigma_0, g, log10 eps0, q).

Steps:
  1. PRE-TEST: does the fitted per-galaxy dyntrans alpha (migration speed) correlate
     with MAH burstiness? (Expect anti-correlation: merger-rich -> faster migration.)
  2. Fit the event model at event thresholds s_min in {0.03, 0.05, 0.10} (sensitivity).
  3. Compare vs dyntrans: per-epoch medians AND per-galaxy scatter (events should
     shrink the badly-fit tail if merger individuality is the driver).

Run: PYTHONPATH=. uv run python experiments/exp30_transport_kernel/event_kicks.py [n] [--refit]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, nnls
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP29 = ROOT / "experiments" / "exp29_outer_deposit"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP29))
import transport_floor as tf                                                          # noqa: E402
from run import ANCHOR_SNAP, ANCHOR_Z                                                # noqa: E402
from real_mah_test import real_mah                                                   # noqa: E402
import mass_qa                                                                        # noqa: E402
from hongshao.plotting import set_style, save_fig                                    # noqa: E402

set_style()
FIGDIR, OUTDIR = HERE / "figures", HERE / "outputs"
OUT_NPZ = OUTDIR / "event_kicks.npz"
R, AT = tf.R, tf.AT
S_MINS = [0.03, 0.05, 0.10]


def mah_events(mah, s_min):
    """Event times/strengths from the peak-history steps: s_j = dMh_j / Mh(t_j)."""
    s = mah["dMh"] / 10.0 ** mah["logMh_full"][1:]
    sel = s > s_min
    return mah["t"][sel], s[sel]


def basis_events(theta, ti, t_obs, tk, t_ev, s_ev):
    """(24, N) unit-mass CoG basis; kick-product core fraction, multi-scale width."""
    s0, g, eps, q = 10.0 ** theta[0], theta[1], 10.0 ** theta[2], max(theta[3], 0.0)
    sig0 = np.clip(s0 * (ti / t_obs) ** g, 1e-4, 1e5)
    logf = np.log(np.clip(1.0 - eps * s_ev, 1e-3, 1.0))
    ccum = np.concatenate([[0.0], np.cumsum(logf)])           # C(t) = sum_{t_j <= t} logf_j
    c_ti = ccum[np.searchsorted(t_ev, ti, side="right")]      # events with t_j <= t_i excluded
    c_tk = ccum[np.searchsorted(t_ev, tk, side="right")]
    fc = np.exp(c_tk - c_ti)                                  # product over t_i < t_j <= t_k
    sigw = np.clip(sig0 * (tk / ti) ** q, 1e-4, 1e5)
    core = 1.0 - np.exp(-R[:, None] ** 2 / (2.0 * sig0[None, :] ** 2))
    wide = 1.0 - np.exp(-R[:, None] ** 2 / (2.0 * sigw[None, :] ** 2))
    return fc[None, :] * core + (1.0 - fc)[None, :] * wide


def solve(theta, mah, data, ev):
    ti, snap, t_obs = mah["t"], mah["snap"], mah["t_obs"]
    masks = [snap <= sa for sa in ANCHOR_SNAP]
    blocks = [basis_events(theta, ti, t_obs, AT[k], *ev) * masks[k][None, :] for k in range(5)]
    A = np.vstack([b / data[k][:, None] for k, b in enumerate(blocks)])
    if not np.isfinite(A).all():
        return None, np.inf
    x, rnorm = nnls(A, np.ones(A.shape[0]), maxiter=10 * A.shape[1])
    return np.array([b @ x for b in blocks]), rnorm


def fit(mah, data, ev, dy_th):
    def loss(th):
        return solve(th, mah, data, ev)[1]

    starts = [[dy_th[0], dy_th[1], le, max(dy_th[3], 0.1)]
              for le in (np.log10(0.3), 0.0, np.log10(3.0))] + [[dy_th[0], dy_th[1], 0.0, 1.5]]
    best = None
    for p0 in starts:
        r = minimize(loss, p0, method="Nelder-Mead",
                     options=dict(maxiter=8000, xatol=1e-4, fatol=1e-10))
        if best is None or r.fun < best.fun:
            best = r
    return solve(best.x, mah, data, ev)[0], best.x


LAG_NPZ = OUTDIR / "event_lag.npz"
LAG_SMINS = [0.03, 0.10]


def fit_lagged(mah, data, ev, warm4):
    """Lagged-kick fit: events fire at t'_j = (1+beta) t_j (stellar coalescence after
    the halo merger); beta >= 0 is a free 5th parameter. Window uses the LAGGED time,
    so deposits laid between halo merger and coalescence are also kicked."""
    t_ev, s_ev = ev

    def loss(th):
        beta = max(th[4], 0.0)
        return solve(th[:4], mah, data, ((1.0 + beta) * t_ev, s_ev))[1]

    starts = [list(warm4) + [b] for b in (0.3, 0.7)] + [[warm4[0], warm4[1], 0.0, 1.5, 0.5]]
    best = None
    for p0 in starts:
        r = minimize(loss, p0, method="Nelder-Mead",
                     options=dict(maxiter=9000, xatol=1e-4, fatol=1e-10))
        if best is None or r.fun < best.fun:
            best = r
    beta = max(best.x[4], 0.0)
    cogs, _ = solve(best.x[:4], mah, data, ((1.0 + beta) * t_ev, s_ev))
    return cogs, best.x


def lagged_main():
    refit = "--refit" in sys.argv
    n = next((int(a) for a in sys.argv[2:] if a.isdigit()), 45)
    dtf = np.load(HERE / "outputs" / "transport_floor.npz")
    dev = np.load(OUT_NPZ)
    datas = dtf["data"][:n]; ng = len(datas)

    if refit or not LAG_NPZ.exists():
        out = {}
        for sm in LAG_SMINS:
            cogs, pars = [], []
            for i in range(ng):
                mah = real_mah(int(dtf["index"][i]))
                data = [datas[i][k] for k in range(5)]
                c, th = fit_lagged(mah, data, mah_events(mah, sm), dev[f"pars_{sm}"][i][:4])
                cogs.append(c); pars.append(th)
            out[f"cogs_{sm}"] = np.array(cogs); out[f"pars_{sm}"] = np.array(pars)
        np.savez(LAG_NPZ, index=dtf["index"][:ng], **out)
        print(f"wrote {LAG_NPZ}  (n={ng})")
    d = np.load(LAG_NPZ)

    mr_dy = np.array([tf.maxrel(dtf["dyntrans"][i], datas[i]) for i in range(ng)])
    mr_un = np.array([tf.maxrel(dev["cogs_0.03"][i], datas[i]) for i in range(ng)])
    avg = lambda m: 100 * np.median([np.median(m[:, k]) for k in range(5)])
    print(f"\nlagged-kick test (n={ng}), median max|rel| per epoch:")
    print(f"  {'model':>22s} | " + " | ".join(f"z={z}".rjust(6) for z in ANCHOR_Z) +
          " |  avg | median beta")
    print(f"  {'dyntrans (smooth)':>22s} | " +
          " | ".join(f"{100*np.median(mr_dy[:,k]):5.1f}%" for k in range(5)) +
          f" | {avg(mr_dy):4.1f}% |   --")
    print(f"  {'events s>0.03 unlag':>22s} | " +
          " | ".join(f"{100*np.median(mr_un[:,k]):5.1f}%" for k in range(5)) +
          f" | {avg(mr_un):4.1f}% |  0.00")
    best = None
    for sm in LAG_SMINS:
        mr = np.array([tf.maxrel(d[f"cogs_{sm}"][i], datas[i]) for i in range(ng)])
        betas = np.clip(d[f"pars_{sm}"][:, 4], 0.0, None)
        if best is None or avg(mr) < best[1]:
            best = (sm, avg(mr), betas)
        print(f"  {'events s>' + format(sm, '.2f') + ' LAGGED':>22s} | " +
              " | ".join(f"{100*np.median(mr[:,k]):5.1f}%" for k in range(5)) +
              f" | {avg(mr):4.1f}% | {np.median(betas):5.2f}")

    print(f"\n[verdict] dyntrans {avg(mr_dy):.1f}%  vs  lagged events {best[1]:.1f}% "
          f"(s_min={best[0]}, median beta={np.median(best[2]):.2f}, "
          f"IQR {np.percentile(best[2],25):.2f}-{np.percentile(best[2],75):.2f})")
    if best[1] < 0.9 * avg(mr_dy):
        print("  the dynamical-friction lag REVIVES the event clock -> merger timing matters once\n"
              "  coalescence delay is modeled; adopt lagged events for the phase-3 v2 channel.")
    elif best[1] < 1.05 * avg(mr_dy):
        print("  lagged events ~match the smooth clock -> consistent with the smooth self-similar\n"
              "  clock BEING the delay-averaged merger clock; discreteness adds nothing in-sample.\n"
              "  Keep dyntrans; the ex-situ channel case rests on the dual-region deposit (v2).")
    else:
        print("  lag does not rescue the event clock -> keep dyntrans.")
    _lag_figure(mr_dy, mr_un, d, datas, ng)


def _lag_figure(mr_dy, mr_un, d, datas, ng):
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.0, 5.0))
    x = np.arange(5)
    a.plot(x, [100 * np.median(mr_dy[:, k]) for k in range(5)], "o-", c="#56B4E9", lw=2,
           label="dyntrans (smooth)")
    a.plot(x, [100 * np.median(mr_un[:, k]) for k in range(5)], "s--", c="0.6", lw=1.5,
           label="events s>0.03 (unlagged)")
    for sm, col in zip(LAG_SMINS, ("#E69F00", "#CC3377")):
        mr = np.array([tf.maxrel(d[f"cogs_{sm}"][i], datas[i]) for i in range(ng)])
        a.plot(x, [100 * np.median(mr[:, k]) for k in range(5)], "D-", c=col, lw=1.8,
               label=f"events s>{sm:.2f} lagged")
    a.set_xticks(x); a.set_xticklabels([f"{z}" for z in ANCHOR_Z])
    a.set(xlabel="epoch z", ylabel="median max|rel| [%]", ylim=(0, None),
          title="A. Does the coalescence lag revive the event clock?")
    a.legend(fontsize=8)
    for sm, col in zip(LAG_SMINS, ("#E69F00", "#CC3377")):
        betas = np.clip(d[f"pars_{sm}"][:, 4], 0.0, None)
        b.hist(betas, bins=np.arange(0, 2.6, 0.2), histtype="step", lw=2, color=col,
               label=f"s>{sm:.2f} (median {np.median(betas):.2f})")
    b.set(xlabel=r"fitted lag $\beta$  ($t'_j = (1+\beta)\,t_j$)", ylabel="galaxies",
          title="B. Fitted coalescence lag")
    b.legend(fontsize=9)
    fig.suptitle("exp30 phase 2.2b — lagged event kicks (dynamical-friction delay)", fontsize=12)
    fig.tight_layout()
    print("\nwrote", save_fig(fig, FIGDIR / "exp30_event_lag")[0])


def pretest(d):
    """Spearman correlation of fitted dyntrans migration speed vs MAH burstiness."""
    alphas, b_frac, b_max = [], [], []
    for i, gi in enumerate(d["index"]):
        mah = real_mah(int(gi))
        s = mah["dMh"] / 10.0 ** mah["logMh_full"][1:]
        tot = mah["dMh"].sum()
        alphas.append(d["params_dyntrans"][i, 2])              # log10 alpha
        b_frac.append(mah["dMh"][s > 0.10].sum() / tot)        # growth fraction in big steps
        b_max.append(s.max())
    alphas, b_frac, b_max = map(np.array, (alphas, b_frac, b_max))
    r1, p1 = spearmanr(alphas, b_frac)
    r2, p2 = spearmanr(alphas, b_max)
    print("[pre-test] fitted log10(alpha) (dyntrans migration timescale) vs MAH burstiness:")
    print(f"  vs growth-fraction in steps>10%: Spearman rho={r1:+.2f} (p={p1:.3f})")
    print(f"  vs max single-step fraction    : Spearman rho={r2:+.2f} (p={p2:.3f})")
    print("  (expected sign: NEGATIVE -- merger-rich galaxies need faster migration)\n")
    return alphas, b_frac, r1, p1


def compute(n):
    d = np.load(HERE / "outputs" / "transport_floor.npz")
    idx, logms, datas = d["index"][:n], d["logms"][:n], d["data"][:n]
    cogs = {sm: [] for sm in S_MINS}; pars = {sm: [] for sm in S_MINS}; nev = {sm: [] for sm in S_MINS}
    for i in range(len(idx)):
        mah = real_mah(int(idx[i]))
        data = [datas[i][k] for k in range(5)]
        for sm in S_MINS:
            ev = mah_events(mah, sm)
            c, th = fit(mah, data, ev, d["params_dyntrans"][i])
            cogs[sm].append(c); pars[sm].append(th); nev[sm].append(len(ev[0]))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, index=idx, logms=logms, data=datas,
             **{f"cogs_{sm}": np.array(cogs[sm]) for sm in S_MINS},
             **{f"pars_{sm}": np.array(pars[sm]) for sm in S_MINS},
             **{f"nev_{sm}": np.array(nev[sm]) for sm in S_MINS})
    print(f"wrote {OUT_NPZ}  (n={len(idx)})")
    return np.load(OUT_NPZ)


def main():
    refit = "--refit" in sys.argv
    n = next((int(a) for a in sys.argv[1:] if a.isdigit()), 45)
    dtf = np.load(HERE / "outputs" / "transport_floor.npz")
    alphas, burst, rho, pval = pretest(dtf)

    d = compute(n) if (refit or not OUT_NPZ.exists()) else np.load(OUT_NPZ)
    datas = d["data"]; ng = len(datas)
    dy = dtf["dyntrans"][:ng]
    mr_dy = np.array([tf.maxrel(dy[i], datas[i]) for i in range(ng)])

    print(f"event-kick model (n={ng}), median max|rel| per epoch, by event threshold s_min:")
    print(f"  {'model':>16s} | " + " | ".join(f"z={z}".rjust(6) for z in ANCHOR_Z) +
          " |  avg | median n_events")
    print(f"  {'dyntrans (ref)':>16s} | " +
          " | ".join(f"{100*np.median(mr_dy[:,k]):5.1f}%" for k in range(5)) +
          f" | {100*np.median([np.median(mr_dy[:,k]) for k in range(5)]):4.1f}% |    --")
    mr_ev, best = {}, None
    for sm in S_MINS:
        cogs = d[f"cogs_{sm}"]
        mr = np.array([tf.maxrel(cogs[i], datas[i]) for i in range(ng)])
        mr_ev[sm] = mr
        avg = 100 * np.median([np.median(mr[:, k]) for k in range(5)])
        if best is None or avg < best[1]:
            best = (sm, avg)
        print(f"  {'events s>' + format(sm, '.2f'):>16s} | " +
              " | ".join(f"{100*np.median(mr[:,k]):5.1f}%" for k in range(5)) +
              f" | {avg:4.1f}% | {int(np.median(d[f'nev_{sm}'])):5d}")

    sm = best[0]
    ga_dy = mr_dy.mean(1); ga_ev = mr_ev[sm].mean(1)          # per-galaxy epoch-avg
    print(f"\n  per-galaxy epoch-avg max|rel| (best events s_min={sm}):")
    for lab, v in (("dyntrans", ga_dy), (f"events", ga_ev)):
        print(f"    {lab:>9s}: p50 {100*np.percentile(v,50):5.1f}%  p75 {100*np.percentile(v,75):5.1f}%  "
              f"p90 {100*np.percentile(v,90):5.1f}%")

    dy_avg = 100 * np.median([np.median(mr_dy[:, k]) for k in range(5)])
    print(f"\n[verdict] dyntrans {dy_avg:.1f}%  vs  events {best[1]:.1f}% (s_min={sm})")
    if best[1] < 0.9 * dy_avg:
        print("  event-triggered kicks BEAT the smooth clock -> merger individuality carries real\n"
              "  signal; adopt the event clock and proceed to the held-out-epoch test with it.")
    elif best[1] < 1.1 * dy_avg:
        print("  events ~match the smooth clock: merger timing adds no net in-sample gain over the\n"
              "  self-similar tau=alpha*t_i. Keep dyntrans (simpler, no threshold); the event clock\n"
              "  remains interesting for GENERALIZATION (phase 2.3) and per-galaxy scatter.")
    else:
        print("  events UNDERPERFORM the smooth clock -> the burst schedule as extracted does not\n"
              "  drive migration; keep dyntrans and revisit event extraction before retrying.")
    _figure(alphas, burst, rho, pval, mr_dy, mr_ev, sm, ga_dy, ga_ev)
    if best[1] < 0.9 * dy_avg:
        mass_qa.evaluate(d[f"cogs_{sm}"], datas, R, ANCHOR_Z, name="events", figdir=FIGDIR)


def _figure(alphas, burst, rho, pval, mr_dy, mr_ev, sm, ga_dy, ga_ev):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    a, b, c = axes
    a.scatter(burst, alphas, s=28, c="#009E73", edgecolor="0.3", lw=0.4)
    a.set(xlabel="MAH burstiness (growth fraction in steps >10%)",
          ylabel=r"fitted $\log_{10}\alpha$ (dyntrans)",
          title=f"A. Pre-test: migration speed vs burstiness\nSpearman $\\rho$={rho:+.2f} (p={pval:.3f})")

    x = np.arange(5)
    b.plot(x, [100 * np.median(mr_dy[:, k]) for k in range(5)], "o-", c="#56B4E9", lw=2,
           label="dyntrans (smooth clock)")
    for s_, col in zip(S_MINS, ("#E69F00", "#D55E00", "#CC3377")):
        b.plot(x, [100 * np.median(mr_ev[s_][:, k]) for k in range(5)], "s--", c=col, lw=1.6,
               label=f"events s>{s_:.2f}")
    b.set_xticks(x); b.set_xticklabels([f"{z}" for z in ANCHOR_Z])
    b.set(xlabel="epoch z", ylabel="median max|rel| [%]", ylim=(0, None),
          title="B. Smooth clock vs event kicks")
    b.legend(fontsize=8)

    lim = 100 * max(ga_dy.max(), ga_ev.max()) * 1.05
    c.scatter(100 * ga_dy, 100 * ga_ev, s=30, c="#D55E00", edgecolor="0.3", lw=0.4)
    c.plot([0, lim], [0, lim], "k--", lw=1)
    c.set(xlabel="dyntrans per-galaxy epoch-avg max|rel| [%]",
          ylabel=f"events (s>{sm:.2f}) [%]", xlim=(0, lim), ylim=(0, lim),
          title="C. Per-galaxy: below 1:1 = events better")
    fig.suptitle("exp30 phase 2.2 — event-triggered kicks vs the smooth dynamical clock", fontsize=12)
    fig.tight_layout()
    print("\nwrote", save_fig(fig, FIGDIR / "exp30_event_kicks")[0])


def demo():
    """Self-check: kick-window bookkeeping and the eps->0 limit."""
    ti = np.array([1.0, 3.0, 5.5])
    t_ev, s_ev = np.array([2.0, 5.0]), np.array([0.2, 0.1])
    th = [2.0, 1.5, 0.0, 1.0]                                  # eps0 = 1
    # expected core fractions at tk=6: deposit0 sees both events, deposit1 only t=5, deposit2 none
    Rsave, expected = R, [(1 - 0.2) * (1 - 0.1), (1 - 0.1), 1.0]
    B = basis_events(th, ti, 9.4, 6.0, t_ev, s_ev)
    huge = 1.0 - np.exp(-R[-1] ** 2 / 1e10)                    # ~0; use small-R behaviour instead
    # infer fc from the basis at a small radius where core ~ N(sig0) dominates:
    s0 = 100.0 * (ti / 9.4) ** 1.5
    core = 1.0 - np.exp(-R[0] ** 2 / (2 * s0 ** 2))
    sigw = np.clip(s0 * (6.0 / ti) ** 1.0, 1e-4, 1e5)
    wide = 1.0 - np.exp(-R[0] ** 2 / (2 * sigw ** 2))
    fc = (B[0] - wide) / (core - wide)
    assert np.allclose(fc, expected, atol=1e-9), (fc, expected)
    B0 = basis_events([2.0, 1.5, -12.0, 1.0], ti, 9.4, 6.0, t_ev, s_ev)   # eps -> 0
    Badd = tf.basis([2.0, 1.5], ti, 9.4, 6.0, "additive")
    assert np.allclose(B0, Badd, atol=1e-6), "eps->0 must recover the additive basis"
    # lag window: event at t=2 lagged by beta=0.5 fires at t'=3 -> now kicks the
    # deposit laid at t=2.5 (between halo merger and coalescence)
    ti2 = np.array([1.0, 2.5, 5.5])
    B2 = basis_events(th, ti2, 9.4, 6.0, 1.5 * t_ev, s_ev)     # lagged events: t'=[3, 7.5]
    s0b = 100.0 * (ti2 / 9.4) ** 1.5
    coreb = 1.0 - np.exp(-R[0] ** 2 / (2 * s0b ** 2))
    sigwb = np.clip(s0b * (6.0 / ti2) ** 1.0, 1e-4, 1e5)
    wideb = 1.0 - np.exp(-R[0] ** 2 / (2 * sigwb ** 2))
    fcb = (B2[0] - wideb) / (coreb - wideb)
    assert np.allclose(fcb, [0.8, 0.8, 1.0], atol=1e-9), fcb   # t'=3 kicks ti=1 & 2.5; t'=7.5 > tk
    print("event_kicks.demo OK: kick windows exact, eps->0 == additive, lag window correct")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    elif len(sys.argv) > 1 and sys.argv[1] == "lagged":
        sys.exit(lagged_main())
    else:
        sys.exit(main())
