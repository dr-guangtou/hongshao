"""exp29 — the puff-up deposition model: one consistent history, widths migrate.

The fixed-param joint fit fails (~10% max-rel) and a loose polynomial-in-z parameter
fit plateaus at ~4.5% (loose_zdep.py). Both keep each deposit's width frozen at its
deposition time. The single-epoch param trends (param_trends.py) said the early-formed
mass must DE-CONCENTRATE with time (pre-z2 fraction inside 5 kpc 0.64->0.44). The
puff-up model gives exactly that one structured DOF: a deposit laid at t_i keeps its
mass but its width GROWS after deposition, so observed at t_k it is

    sigma_i(t_k) = sigma_0 * (t_i / t_obs)^g * (t_k / t_i)^q ,   q >= 0

  - (t_i/t_obs)^g : initial width at deposition (inside-out: late deposits land wide)
  - (t_k/t_i)^q   : post-deposition puffing, grows with the OBSERVATION time t_k
  - q = 0         : recovers the fixed model exactly (strict superset)

Masses dM*_i = f(t_i)*dMh_i use ONE efficiency f for all epochs (a single assembly
history; only widths depend on t_k). 6 params: (log10 sigma_0, g, q, b_e, b_l, z_c),
fit jointly to all 5 CoGs, aperture-pinned at 148 kpc, same objective as loose_zdep.

Compared per galaxy: no-puff (q=0, 5 params) vs puff (q free, 6 params) vs the cached
independent ceiling. Question: does the puff DOF beat the loose-zdep ~4.5% benchmark
and approach the ~0.7% single-epoch ceiling?

Run: PYTHONPATH=. uv run python experiments/exp29_outer_deposit/puff_fit.py [n]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from deposit import deposit_cog, eff_two_epoch, deposited                           # noqa: E402
from run import dipfree_mah, ANCHOR_SNAP, ANCHOR_Z                                   # noqa: E402
from cog_extrapolate import measured_cog                                            # noqa: E402
from single_epoch_all import anchor_times                                           # noqa: E402
from hongshao.tng_data import COG_RAD_KPC                                           # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO                        # noqa: E402

set_style()
FIGDIR = HERE / "figures"
NPZ = HERE / "outputs" / "single_epoch_params.npz"
R = COG_RAD_KPC
R_IN = 3.0
EVAL = R > R_IN


def build_cogs(s0, g, qp, be, bl, zc, mah, data, t_obs, at, law):
    """Aperture-pinned puff-up CoG at all 5 epochs; None if pathological.

    law='ratio': sigma = sigma0 * (t_k/t_i)^qp   (qp = q, multiplicative)
    law='diff' : sigma = sqrt(sigma0^2 + qp*(t_k - t_i))  (qp = kappa, additive area)
    with the initial width sigma0 = s0 * (t_i/t_obs)^g."""
    if zc <= -1.0:
        return None
    dM = deposited(eff_two_epoch(mah["z"], be, bl, zc), mah["dMh"], 1.0)
    out = []
    for k in range(5):
        inc = mah["snap"] <= ANCHOR_SNAP[k]
        ti = mah["t"][inc]
        sig0 = s0 * (ti / t_obs) ** g
        sig = sig0 * (at[k] / ti) ** qp if law == "ratio" else np.sqrt(sig0 ** 2 + qp * (at[k] - ti))
        m = deposit_cog(dM[inc], sig, 0.0, R)
        if not np.isfinite(m[-1]) or m[-1] <= 0:
            return None
        out.append(m * (data[k][-1] / m[-1]))
    return out


def _unpack(p, puff):
    s0 = 10.0 ** p[0]
    if puff:
        return s0, p[1], max(p[2], 0.0), p[3], p[4], p[5]         # q clipped >= 0
    return s0, p[1], 0.0, p[2], p[3], p[4]


def _epoch_relrms(m, d):
    return np.sqrt(np.mean(((m[EVAL] - d[EVAL]) / d[EVAL]) ** 2))


PUFF_STARTS = {"ratio": (0.0, 0.5, 1.0, 1.8), "diff": (0.0, 5.0, 20.0, 80.0)}


def fit(mah, data, t_obs, at, Pg, puff, law="ratio"):
    """Joint multi-epoch fit; mean per-epoch rel-RMS objective. Returns (models, p)."""
    def loss(p):
        cogs = build_cogs(*_unpack(p, puff), mah, data, t_obs, at, law)
        if cogs is None:
            return 1e3
        return float(np.mean([_epoch_relrms(cogs[k], data[k]) for k in range(5)]))

    base = [np.median(Pg[:, j]) for j in range(5)]                # log10 s0, g, b_e, b_l, z_c
    if puff:
        starts = [[base[0], base[1], q0, base[2], base[3], base[4]] for q0 in PUFF_STARTS[law]]
    else:
        starts = [base, [base[0], base[1], 4.0, 1.5, 2.0]]
    best = None
    for p0 in starts:
        r = minimize(loss, p0, method="Nelder-Mead",
                     options=dict(maxiter=3000 * len(p0), xatol=1e-4, fatol=1e-9))
        if best is None or r.fun < best.fun:
            best = r
    return build_cogs(*_unpack(best.x, puff), mah, data, t_obs, at, law), best.x


def metrics(cogs, data):
    out = []
    for k in range(5):
        rel = np.abs((cogs[k][EVAL] - data[k][EVAL]) / data[k][EVAL])
        logr = np.sqrt(np.mean((np.log10(cogs[k][EVAL]) - np.log10(data[k][EVAL])) ** 2))
        out.append((float(logr), float(rel.max())))
    return np.array(out)                                          # (5,2)


def main():
    at = anchor_times()
    d = np.load(NPZ)
    idx, P, logms, ind_met = d["index"], d["params"], d["logms"], d["metrics"]
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(idx)
    sel = np.linspace(0, len(idx) - 1, min(n, len(idx))).round().astype(int)

    nop_m, ratio_m, diff_m, ind_m, qs, ks, kept_logm = [], [], [], [], [], [], []
    for i in sel:
        gi = int(idx[i])
        logC = measured_cog(gi)
        if logC is None:
            continue
        data = [10.0 ** logC[k] for k in range(5)]
        mah = dipfree_mah(gi); t_obs = mah["t_obs"]
        nc, _ = fit(mah, data, t_obs, at, P[i], puff=False)
        rc, rp = fit(mah, data, t_obs, at, P[i], puff=True, law="ratio")
        dc, dp = fit(mah, data, t_obs, at, P[i], puff=True, law="diff")
        if nc is None or rc is None or dc is None:
            continue
        nop_m.append(metrics(nc, data)); ratio_m.append(metrics(rc, data)); diff_m.append(metrics(dc, data))
        ind_m.append(ind_met[i][:, :2]); qs.append(max(rp[2], 0.0)); ks.append(max(dp[2], 0.0))
        kept_logm.append(logms[i])
    nop_m, ratio_m, diff_m, ind_m = map(np.array, (nop_m, ratio_m, diff_m, ind_m))
    qs, ks, kept_logm = np.array(qs), np.array(ks), np.array(kept_logm)

    print(f"exp29 — puff-up joint multi-epoch fit (n={len(ratio_m)}), max|rel| over R>3 kpc, "
          f"aperture-pinned\n")
    print(f"  {'z':>4s} | {'no-puff':>9s} | {'puff:ratio':>11s} | {'puff:diff':>10s} | "
          f"{'loose-zdep*':>12s} | {'independent':>12s}")
    LOOSE = {0.4: 3.7, 0.7: 5.6, 1.0: 5.1, 1.5: 4.5, 2.0: 3.4}    # quad, n=60 (loose_zdep.py)
    for k, z in enumerate(ANCHOR_Z):
        print(f"  {z:4.1f} | {100*np.median(nop_m[:,k,1]):7.1f}% | "
              f"{100*np.median(ratio_m[:,k,1]):9.1f}% | {100*np.median(diff_m[:,k,1]):8.1f}% | "
              f"{LOOSE[z]:10.1f}% | {100*np.median(ind_m[:,k,1]):10.1f}%")
    avg = lambda m: 100 * np.median([np.median(m[:, k, 1]) for k in range(5)])
    print(f"\n  epoch-avg median max|rel|:  no-puff {avg(nop_m):.1f}%   ratio {avg(ratio_m):.1f}%   "
          f"diff {avg(diff_m):.1f}%   loose-zdep ~4.5%   independent {avg(ind_m):.1f}%")
    print(f"  ratio q : median {np.median(qs):.2f} (IQR {np.percentile(qs,25):.2f}-{np.percentile(qs,75):.2f})"
          f"   diff kappa: median {np.median(ks):.1f} (IQR {np.percentile(ks,25):.1f}-{np.percentile(ks,75):.1f})")

    best = min(("ratio", avg(ratio_m)), ("diff", avg(diff_m)), key=lambda t: t[1])
    print(f"\n[verdict] best puffing law: {best[0]} at {best[1]:.1f}% epoch-avg "
          f"(no-puff {avg(nop_m):.1f}%, loose-zdep ~4.5%, ceiling {avg(ind_m):.1f}%)")
    if best[1] < 4.5:
        print(f"  the {best[0]} puffing law beats the loose-zdep benchmark with ONE consistent\n"
              "  history (6 params) -> post-deposition width growth is the missing freedom.\n"
              "  Next: population forward emulator with puffing (legal halo-only inputs).")
    elif best[1] < 0.85 * avg(nop_m):
        print("  puffing helps but neither law clears the loose-zdep benchmark -> with frozen\n"
              "  mass, width migration is less effective than letting the params float; a hybrid\n"
              "  (mild z-dependent efficiency + puffing) or halo-driven law is the next lever.")
    else:
        print("  puffing barely helps -> the deposit mass distribution, not the width, is the\n"
              "  dominant missing freedom; favour the loose z-dependent (or hybrid) model.")
    _figure(nop_m, ratio_m, diff_m, ind_m, qs, ks, kept_logm)


def _figure(nop_m, ratio_m, diff_m, ind_m, qs, ks, logms):
    x = np.arange(5)
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for m, c, lab, mk in ((nop_m, OKABE_ITO[5], "no-puff (5p)", "s"),
                          (ratio_m, OKABE_ITO[2], "puff:ratio (6p)", "o"),
                          (diff_m, OKABE_ITO[0], "puff:diff (6p)", "D"),
                          (ind_m, "0.5", "independent (ceiling)", "^")):
        a.plot(x, [100 * np.median(m[:, k, 1]) for k in range(5)], mk + "-", c=c, lw=2, ms=6, label=lab)
    a.axhline(4.5, c=OKABE_ITO[1], ls="--", lw=1.2, label="loose-zdep quad (~4.5%)")
    a.axhline(2.0, c="0.7", ls=":", lw=1.0)
    a.set_xticks(x); a.set_xticklabels([f"{z}" for z in ANCHOR_Z])
    a.set(xlabel="observation epoch z", ylabel="median max |rel residual| over R>3 kpc [%]",
          ylim=(0, None), title="A. Two puffing laws vs no-puff vs ceiling")
    a.legend(fontsize=8)

    b.scatter(logms, qs, s=30, c=OKABE_ITO[2], edgecolor="0.3", lw=0.3, label=f"ratio q (med {np.median(qs):.2f})")
    b.scatter(logms, ks / 50.0, s=30, marker="D", c=OKABE_ITO[0], edgecolor="0.3", lw=0.3,
              label=f"diff kappa/50 (med {np.median(ks):.0f})")
    b.axhline(0.0, c="0.6", lw=0.8)
    b.set(xlabel=r"$\log M_*$", ylabel="best-fit puffing strength", title="B. Puffing strength vs mass")
    b.legend(fontsize=8)
    fig.suptitle("exp29 — puff-up deposition model: one history, post-deposition width growth", fontsize=12)
    fig.tight_layout()
    print("\nwrote", save_fig(fig, FIGDIR / "exp29_puff_fit")[0])


def demo():
    """Self-check: q=0 puff width == the frozen (t_i/t_obs)^g width."""
    at = anchor_times()
    mah = dipfree_mah(int(np.load(NPZ)["index"][0])); t_obs = mah["t_obs"]
    inc = mah["snap"] <= ANCHOR_SNAP[0]
    ti = mah["t"][inc]
    frozen = 40.0 * (ti / t_obs) ** 1.5
    puffed0 = 40.0 * (ti / t_obs) ** 1.5 * (at[0] / ti) ** 0.0
    assert np.allclose(frozen, puffed0), "q=0 must recover the frozen width"
    assert np.all((at[0] / ti) ** 0.8 >= 1.0 - 1e-9), "puffing factor must be >= 1 (t_k >= t_i)"
    print("puff_fit.demo OK: q=0 recovers frozen width; puffing factor >= 1")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        sys.exit(main())
