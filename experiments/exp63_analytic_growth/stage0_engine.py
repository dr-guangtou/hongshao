"""exp63 Stage 0 — the engine gates (G0a-G0d) and the D3 truncation measurement.

Four gates, all frozen in `doc/plans/2026-08-28-exp63-stage0-implementation.md`
before this ran, and one measurement the user asked for before deciding D3:

  G0a  the bridge. On the incumbent's own 72-step grid with the catalog R200c,
       the engine's `discrete_sum(sub=1)` reproduces `exp54/model.forward` to
       1e-5 dex on every galaxy without a mid-history R200c gap (the stored
       curve is float32, 2.7e-6 dex; the bridge cannot beat that).
  G0b  the limit. Splitting each step 2/4/8/16 times halves the distance to
       `predict(mode="catalog")` per doubling (first-order convergence, ratios
       in [1.8, 2.2]), and the Richardson extrapolation 2 S16 - S8 lands on
       `predict` to 5e-4 dex (median over galaxies). The integral IS the limit
       of the incumbent's sum.
  G0c  the closed forms of tech note 04 (Eq. 5, and the Mellin constants of
       the three kernel families) to 1e-8 — checked in `engine.py`'s
       self-check, re-run here so the gate log carries it.
  G0d  the cost: one full-sample `predict` (2397 galaxies, five epochs),
       timed. No threshold; the number schedules the later fits.
  D3   for the UNTRUNCATED gompertz_log kernel at the incumbent's parameters,
       the fraction of each galaxy's z=0.4 deposited stellar mass that lies
       beyond 3 R200c(t') of its own deposit. Rule (plan §4): keep the 3 R200c
       truncation iff the population median exceeds 1 per cent.

Plus the two bookkeeping quantities the integral removes, measured on the
full sample: the stellar mass the incumbent drops at mid-history gaps and
before its first catalog R200c, as a fraction of what it keeps.

REFUSES TO WRITE unless every gate passes. `--smoke` runs every 24th galaxy
and writes `*_smoke.*` only.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp63_analytic_growth/stage0_engine.py [--smoke]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term", HERE):
    sys.path.insert(0, str(p))

import engine as E                                       # noqa: E402  (imports fit first)
import fit as F                                          # noqa: E402
import model as M                                        # noqa: E402
import families                                          # noqa: E402
import stage0_cost as S0                                 # noqa: E402

RULE = "=" * 88
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
#: the gates, frozen
G0A_TOL = 1e-5
G0B_RATIO = (1.8, 2.2)
G0B_RICH = 5e-4
G0C_TOL = 1e-8
D3_KEEP_ABOVE = 0.01


def gate_g0a(recs, curves, spec, theta, R):
    worst, n_gap, dropped = 0.0, 0, []
    eff, _, _ = spec.unpack(theta)
    for h, hc in zip(recs, curves):
        j0 = int(np.argmax(hc.ok_steps))
        gap = (~hc.ok_steps[j0:]).any()
        em = hc.epoch_mask_steps[0]
        dm = 10.0 ** E.log_eps_e2(eff, hc.logmh_steps, E.z_of_t(hc.t_steps)) * hc.dmh_steps
        kept = dm[em & hc.ok_steps].sum()
        dropped.append(((dm[em & ~hc.ok_steps & (np.arange(len(em)) >= j0)].sum()) / kept,
                        dm[em & (np.arange(len(em)) < j0)].sum() / kept))
        if gap:
            n_gap += 1
            continue
        ref = M.forward(spec, theta, h, [0, 1, 2, 3, 4], R)
        got = E.discrete_sum(spec, theta, hc, R, sub=1)
        worst = max(worst, float(np.max(np.abs(np.log10(got / ref)))))
    return worst, n_gap, np.array(dropped)


def gate_g0b(curves, spec, theta, R, every=10):
    ratios, rich = [], []
    for hc in curves[::every]:
        # only galaxies whose kept steps run unbroken to the anchor: with a
        # gap, `discrete_sum` integrates a different domain than `predict`
        if (~hc.ok_steps[int(np.argmax(hc.ok_steps)):]).any():
            continue
        pc = E.predict(spec, theta, [hc], R, mode="catalog",
                       t_start=10.0 ** E.step_start(hc))[0]
        sums = {s: E.discrete_sum(spec, theta, hc, R, sub=s) for s in (2, 4, 8, 16)}
        seq = np.array([np.max(np.abs(np.log10(sums[s] / pc))) for s in (2, 4, 8, 16)])
        ratios.append(seq[:-1] / seq[1:])
        rich.append(np.max(np.abs(np.log10((2.0 * sums[16] - sums[8]) / pc))))
    return np.array(ratios), np.array(rich)


def gate_g0c(spec, theta, R):
    from scipy.integrate import quad
    eff, size, shape = spec.unpack(theta)
    worst = 0.0
    for alpha in (0.5, 1.0, 2.0, 3.0):
        beta = E.powerlaw_beta(alpha, eff[1], eff[2], size[1])
        tau = np.linspace(np.log(0.3), np.log(9.39), 40001); tt = np.exp(tau)
        Mt = 10 ** 13.5 * (tt / 9.39) ** alpha; opz = (13.8 / tt) ** (2.0 / 3.0)
        r200 = (3 * Mt / (4 * np.pi * 200 * E.RHO_C0 * opz ** 3)) ** (1 / 3) * 1e3
        s = 10 ** (size[0] + size[1] * np.log10(opz)) * r200
        eps = 10 ** (eff[0] + eff[1] * (np.log10(Mt) - 13.5) + eff[2] * np.log(opz))
        Wtau = eps * alpha * Mt
        brute = np.trapezoid(families.cog_unit("gompertz_log", s, shape, R) * Wtau[None, :], tau, axis=1)
        gr = alpha / 3 + 2 * (1 - size[1]) / 3
        A = (Wtau[-1] / gr) / s[-1] ** beta
        cf = E.gompertz_closed_form(R, A, beta, shape[0], s[0], s[-1])
        worst = max(worst, float(np.max(np.abs(np.log10(cf / brute)))))
    for fam, sh, bt in (("moffat", (1.3771,), 0.7), ("sersic", (4.0,), 0.5),
                        ("sersic", (1.0,), 1.2), ("gompertz_log", (0.8,), 0.5)):
        # the Mellin constants are for the kernels in their NATIVE scale
        # (moffat rc, sersic a); `cog_unit` is parametrised by R50, so pass the
        # R50 that makes the native scale equal to 1
        r50_native = (families.scale_from_r50(fam, np.array([1.0]), sh) ** -1
                      if fam != "gompertz_log" else np.array([1.0]))
        num = quad(lambda y: y ** (-bt - 1) * families.cog_unit(fam, r50_native, sh, np.array([y]))[0, 0],
                   0, np.inf, limit=400)[0]
        worst = max(worst, abs(num / E.mellin_constant(fam, bt, sh) - 1))
    return worst


def figure(curves, spec, theta, R, frac_d3, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hongshao.plotting import set_style, save_fig
    from hongshao.qa import _tex, _pct
    set_style()
    dev = []
    for k, hc in enumerate(curves):
        j0 = int(np.argmax(hc.ok_steps))
        if (~hc.ok_steps[j0:]).any():
            continue
        s1 = E.discrete_sum(spec, theta, hc, R, sub=1)
        pc = E.predict(spec, theta, [hc], R, mode="catalog",
                       t_start=10.0 ** E.step_start(hc))[0]
        dev.append(np.log10(s1 / pc))
    dev = np.array(dev)                                    # (n, 5, R)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    cols = ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]
    for j, z in enumerate(E.ANCHOR_Z):
        ax[0].fill_between(R, np.percentile(dev[:, j], 16, 0), np.percentile(dev[:, j], 84, 0),
                           color=cols[j], alpha=0.15)
        ax[0].plot(R, np.median(dev[:, j], 0), color=cols[j], lw=2, label=f"z={z}")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xscale("log"); ax[0].set_xlabel("R [kpc]")
    ax[0].set_ylabel(r"$\log_{10}$ (72-step sum / exact integral)")
    ax[0].set_title(f"(a) the incumbent's discretisation error ({len(dev)} galaxies)")
    ax[0].legend(fontsize=8)
    hc = curves[len(curves) // 3]
    pc = E.predict(spec, theta, [hc], R, mode="catalog", t_start=10.0 ** E.step_start(hc))[0]
    subs = (1, 2, 4, 8, 16)
    errs = [np.max(np.abs(np.log10(E.discrete_sum(spec, theta, hc, R, sub=s)[0] / pc[0]))) for s in subs]
    ax[1].loglog(subs, errs, "o-", label=_tex("max |log10(sum/integral)|, z=0.4"))
    ax[1].loglog(subs, errs[0] / np.array(subs), "--", color="gray", label="first order (1/n)")
    ax[1].set_xlabel("sub-steps per snapshot step"); ax[1].set_ylabel("dex")
    ax[1].set_title(f"(b) the sum converges to the integral (galaxy idx {hc.index})")
    ax[1].legend(fontsize=8)
    ax[2].hist(100 * frac_d3, bins=40, color="#0072B2", alpha=0.7)
    ax[2].axvline(100 * D3_KEEP_ABOVE, color="k", ls="--", label=f"the D3 rule: 1{_pct()}")
    ax[2].axvline(100 * np.median(frac_d3), color="#D55E00",
                  label=f"median {100 * np.median(frac_d3):.2f}{_pct()}")
    ax[2].set_xlabel(f"stellar mass beyond $3R_{{200c}}$ of its deposit, untruncated kernel [{_pct()}]")
    ax[2].set_ylabel("galaxies"); ax[2].legend(fontsize=8)
    ax[2].set_title("(c) the D3 measurement, z=0.4")
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    save_fig(fig, FIGDIR / f"exp63_stage0_engine{tag}")
    return FIGDIR / f"exp63_stage0_engine{tag}.png"


def main(smoke=False, figure_only=False):
    tag = "_smoke" if smoke else ""
    if figure_only:
        recs, data, mask, lmh, spec, theta, _ = S0.build(smoke)
        curves = E.build_curves(recs)
        frac = np.load(OUTDIR / f"stage0_engine{tag}.npz")["frac_beyond_3r200"]
        print(f"  wrote {figure(curves, spec, theta, F.R_GRID, frac, tag).relative_to(ROOT)}")
        return
    print(f"{RULE}\nexp63 STAGE 0 — the engine gates and the D3 measurement"
          f"{' (SMOKE)' if smoke else ''}\n{RULE}\n")
    recs, data, mask, lmh, spec, theta, _ = S0.build(smoke)
    R = F.R_GRID
    curves = E.build_curves(recs)
    print(f"  {len(curves)} galaxies; {spec.label}; incumbent theta "
          f"{np.round(theta, 4)}\n")
    results, passed = {}, {}

    worst, n_gap, dropped = gate_g0a(recs, curves, spec, theta, R)
    passed["G0a"] = worst < G0A_TOL
    print(f"  G0a bridge: discrete_sum(sub=1) vs exp54 forward on {len(curves) - n_gap} "
          f"no-gap galaxies: max |dlog| {worst:.2e} dex (tol {G0A_TOL:.0e}) "
          f"-> {'PASS' if passed['G0a'] else 'FAIL'}")
    print(f"      {n_gap} of {len(curves)} galaxies ({100 * n_gap / len(curves):.1f}%) have a "
          f"mid-history R200c gap, where the incumbent DROPS the step's stars:")
    print(f"      dropped at gaps, as a fraction of kept z=0.4 stellar mass: median "
          f"{100 * np.median(dropped[:, 0]):.2f}%, 90th pct {100 * np.percentile(dropped[:, 0], 90):.2f}%, "
          f"max {100 * dropped[:, 0].max():.2f}%")
    print(f"      dropped before the first catalog R200c (z >~ 10): median "
          f"{100 * np.median(dropped[:, 1]):.3f}%, max {100 * dropped[:, 1].max():.3f}%")
    results.update(g0a_worst=worst, n_gap=n_gap, dropped_gap=dropped[:, 0],
                   dropped_early=dropped[:, 1])

    ratios, rich = gate_g0b(curves, spec, theta, R, every=(1 if smoke else 10))
    passed["G0b"] = bool(np.all((ratios > G0B_RATIO[0]) & (ratios < G0B_RATIO[1]))
                         and np.median(rich) < G0B_RICH)
    print(f"\n  G0b limit: error ratios per doubling of sub-steps, {len(rich)} galaxies: "
          f"min {ratios.min():.2f}, max {ratios.max():.2f} (want {G0B_RATIO}); "
          f"Richardson limit vs predict: median {np.median(rich):.1e}, max {rich.max():.1e} dex "
          f"(tol {G0B_RICH:.0e} on the median) -> {'PASS' if passed['G0b'] else 'FAIL'}")
    results.update(g0b_ratios=ratios, g0b_rich=rich)

    w = gate_g0c(spec, theta, R)
    passed["G0c"] = w < G0C_TOL
    print(f"\n  G0c closed forms (Eq. 5 at four MAH slopes; Mellin constants of three "
          f"kernels): worst {w:.1e} (tol {G0C_TOL:.0e}) -> {'PASS' if passed['G0c'] else 'FAIL'}")
    results["g0c_worst"] = w

    t0 = time.time(); pa = E.predict(spec, theta, curves, R); dt = time.time() - t0
    passed["G0d"] = bool(np.isfinite(pa).all())
    print(f"\n  G0d cost: one full predict, {len(curves)} galaxies x 5 epochs x 24 radii: "
          f"{dt:.2f} s ({1e3 * dt / len(curves):.2f} ms per galaxy); all finite -> "
          f"{'PASS' if passed['G0d'] else 'FAIL'}")
    results["timing_full_sample_s"] = dt

    frac = E.mass_beyond_truncation(spec, theta, curves)
    keep = np.median(frac) > D3_KEEP_ABOVE
    print(f"\n  D3: untruncated gompertz_log (c={spec.unpack(theta)[2][0]:.3f}) at the incumbent's "
          f"parameters — z=0.4 stellar mass beyond 3 R200c of its own deposit:")
    print(f"      median {100 * np.median(frac):.3f}%, 16-84% {100 * np.percentile(frac, 16):.3f}-"
          f"{100 * np.percentile(frac, 84):.3f}%, max {100 * frac.max():.3f}%")
    print(f"      rule: keep the 3 R200c truncation iff median > {100 * D3_KEEP_ABOVE:.0f}% -> "
          f"{'KEEP the truncation' if keep else 'DROP the truncation (untruncated kernels from Stage 1 on)'}")
    results.update(frac_beyond_3r200=frac, d3_keep_truncation=keep)

    ok_all = all(passed.values())
    print(f"\n  gates: " + ", ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in passed.items()))
    if not ok_all:
        raise SystemExit("REFUSING TO WRITE: a gate failed. Fix the engine, not the tolerance.")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"stage0_engine{tag}.npz"
    np.savez(out, gate_names=np.array(list(passed)), gate_passed=np.array(list(passed.values())),
             rows=np.array([hc.row for hc in curves]), **results)
    print(f"  saved {out.relative_to(ROOT)}")
    print(f"  wrote {figure(curves, spec, theta, R, frac, tag).relative_to(ROOT)}")


if __name__ == "__main__":
    main("--smoke" in sys.argv[1:], "--figure-only" in sys.argv[1:])
