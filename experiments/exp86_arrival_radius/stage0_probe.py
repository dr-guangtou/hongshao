"""exp86 Stage 0 — where the delayed mass lands: the anatomy of the 50-100 kpc
shell and the FROZEN-THETA probe of sizing the delayed deposits by the halo
at ARRIVAL (`size_law` knob `w_arr`), on THE ADOPTED MEAN. No fit.

Part 1, the anatomy (before the probe): at z = 1.5 and z = 2 the model's
50-100 kpc shell by channel and by sample (fitting | mh-complete), the
shell's residual on both samples, and per galaxy the mass-weighted
log R200c(t_a) / R200c(t') of the extended deposits that sit in the shell
(the factor the knob applies) — the premise: the complete progenitors'
delayed deposits must be sized up MORE than the rest's.

Part 2, the probe: w_arr in {0.25, 0.5, 0.75, 1.0} and the CONTROL w_arr = 0,
each with (log_f_e, b_e) RE-TUNED on the radius term (exp80 Stage 0 C: the
rms over R20/R50/R80 x three halo-mass terciles x five epochs of the
tercile-median log R_f(model)/R_f(truth) at fixed halo mass; everything else
frozen), then a0 re-centred on the z = 0.4 median M(<103). Gates against the
control (the plan): G1 the mh-complete 50-100 kpc shell deficit at z = 1.5
AND z = 2 at least halved with the fitting sample's shell not pushed away
from zero by more than 3 points; G2 the size-offset cells within 0.05 not
fewer than the control's; G3 the future-dependence gate at no epoch farther
from the truth than the control's by more than 0.01, and the z = 2
recent-growth correlation <= 0.15.

Run (one process; ~100 fit-node predictions per probe point):
    HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 PYTHONPATH=. \\
        uv run python -u experiments/exp86_arrival_radius/stage0_probe.py [--smoke] [--no-retune] [--grid 0.5 1.0]
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink", ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term", ROOT / "experiments/exp63_analytic_growth",
          ROOT / "experiments/exp73_size_relative", ROOT / "experiments/exp74_c19_history_leak",
          ROOT / "experiments/exp78_size_aware_objective", ROOT / "experiments/exp80_deposit_size_law", HERE):
    sys.path.insert(0, str(p))
import fit as F                                          # noqa: E402
import engine as E                                       # noqa: E402
import model as M                                        # noqa: E402
import model2 as M2                                      # noqa: E402
import selection as SEL                                  # noqa: E402
import size_terms as ST                                  # noqa: E402
import size_law as SL                                    # noqa: E402


def _by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S1 = _by_path("exp80_stage1_fit", ROOT / "experiments/exp80_deposit_size_law/stage1_fit.py")
RB = S1.RB
RULE = "=" * 100
ANCHOR_Z = list(E.ANCHOR_Z)
EPOCHS = (0, 1, 2, 3, 4)
OUTDIR = HERE / "outputs"
SEL_NPZ = ROOT / "experiments/exp54_unpinned_amplitude/outputs/selection.npz"
KNOBS = ("q_e", "w_arr")
DEFAULT_GRID = [0.25, 0.5, 0.75, 1.0]
R_SHELL = (50.0, 100.0)
R_CORE, R_10, R_OUTER = 2.0, 10.25, 103.45
SIZE_OFFSET_TOL = 0.05
RETUNE_MAX_EVALS = {"smoke": 25, "full": 120}
SHELL_EPOCHS = (3, 4)


def cum(m, R, r):
    """M(<r) by interpolation on the grid, (n, ...) along the last axis."""
    return np.array([np.interp(r, R, row) for row in m.reshape(-1, m.shape[-1])]).reshape(m.shape[:-1])


def shell(m, R):
    return cum(m, R, R_SHELL[1]) - cum(m, R, R_SHELL[0])


def pct(model, data):
    return 100 * float(np.nanmedian((model - data) / data))


def gate_readings(m, truth, lt_truth, fit_all, complete, lmh_cat, growth, lmh_bin, feats, i_2, i_10, i_o):
    g = {}
    sh_m, sh_t = shell(m, F.R_GRID), shell(truth, F.R_GRID)
    for k in EPOCHS:
        c = complete[:, k]
        g[f"shell_fit_{k}"] = pct(sh_m[:, k], sh_t[:, k])
        g[f"shell_mhc_{k}"] = pct(sh_m[c, k], sh_t[c, k])
        for nm, i in (("10", i_10), ("103", i_o)):
            g[f"m{nm}_fit_{k}"] = pct(m[:, k, i], truth[:, k, i])
            g[f"m{nm}_mhc_{k}"] = pct(m[c, k, i], truth[c, k, i])
        g[f"core_pct_{k}"] = pct(m[:, k, i_2], truth[:, k, i_2])
        _, med, _ = ST.radius_term(m[:, k], truth[:, k], F.R_GRID, lmh_bin[:, k])
        for i, f in enumerate(("R20", "R50", "R80")):
            g[f"size_{f}_{k}"] = float(np.nanmax(np.abs(med[i])))
    g["size_offsets_pass"] = int(sum(g[f"size_{f}_{k}"] <= SIZE_OFFSET_TOL for k in EPOCHS for f in ("R20", "R50", "R80")))
    r = np.log10(m) - lt_truth
    for k in range(1, 5):
        g[f"leak_{k}"] = SEL.partial_growth(r[:, k, i_o], lmh_cat[:, k], growth[:, k], np.isfinite(lmh_cat[:, k]))[1]
    for nm, i in (("5", int(np.argmin(np.abs(F.R_GRID - 4.92)))), ("103", i_o)):
        g[f"rho_recent_z2_{nm}"] = partial(r[:, 4, i], feats[4]["recent"], feats[4]["lmh"])
    return g


def resid_on(y, x, lmh):
    ok = np.isfinite(y) & np.isfinite(x) & np.isfinite(lmh)
    X = np.column_stack([np.ones(ok.sum()), lmh[ok], lmh[ok] ** 2])
    ry = y[ok] - X @ np.linalg.lstsq(X, y[ok], rcond=None)[0]
    rx = x[ok] - X @ np.linalg.lstsq(X, x[ok], rcond=None)[0]
    return ry, rx


def partial(y, x, lmh):
    ry, rx = resid_on(y, x, lmh)
    return float(spearmanr(ry, rx)[0])


def print_gates(label, g, ref=None):
    def d(key, fmt="+.1f"):
        v = g[key]
        return f"{v:{fmt}}" + (f" ({v - ref[key]:+.1f})" if ref is not None and key in ref and fmt == "+.1f" else
                                (f" ({v - ref[key]:+.3f})" if ref is not None and key in ref else ""))
    print(f"  --- {label} ---")
    print("    G1 the 50-100 kpc shell, median (model-data)/data [%], fitting | mh-complete, z=0.4..2:  "
          + "   ".join(f"{d(f'shell_fit_{k}')} | {d(f'shell_mhc_{k}')}" for k in EPOCHS))
    print("    M(<10) fitting | mh-complete: " + "   ".join(f"{g[f'm10_fit_{k}']:+.1f} | {g[f'm10_mhc_{k}']:+.1f}" for k in EPOCHS)
          + ";  M(<103): " + "   ".join(f"{g[f'm103_fit_{k}']:+.1f} | {g[f'm103_mhc_{k}']:+.1f}" for k in EPOCHS))
    print("    M(<2 kpc) median per cent: " + " / ".join(f"{g[f'core_pct_{k}']:+.1f}" for k in EPOCHS))
    print("    G2 size offsets at fixed Mh, max |tercile median| [dex] R20/R50/R80 per epoch: "
          + "   ".join(f"z={ANCHOR_Z[k]} " + "/".join(f"{g[f'size_{f}_{k}']:.3f}" for f in ("R20", "R50", "R80")) for k in EPOCHS)
          + f";  cells within {SIZE_OFFSET_TOL}: {g['size_offsets_pass']}/15" + (f" ({g['size_offsets_pass'] - ref['size_offsets_pass']:+d})" if ref is not None else ""))
    print("    G3 future-dependence gate at 103 kpc [dex per dex] z=0.7/1/1.5/2: " + " / ".join(d(f'leak_{k}', '+.3f') for k in range(1, 5))
          + ("  (truth " + " / ".join(f"{ref[f'leak_truth_{k}']:+.3f}" for k in range(1, 5)) + ")" if ref is not None else "")
          + f";  rho(recent) z=2 at 5 / 103 kpc {d('rho_recent_z2_5', '+.2f')} / {d('rho_recent_z2_103', '+.2f')}")
    if "loss" in g:
        print(f"    standard loss at the fit nodes: {g['loss']:.4f}" + (f" ({g['loss'] - ref['loss']:+.3f})" if ref is not None else "")
              + (f";  re-tuned log_f_e {g['log_f_e']:+.3f}, b_e {g['b_e']:+.3f}; radius term {g['zrms']:.4f}" if "log_f_e" in g else ""))


def gate_verdict(g, ctl, leak_truth):
    g1 = all(abs(g[f"shell_mhc_{k}"]) <= 0.5 * abs(ctl[f"shell_mhc_{k}"]) for k in SHELL_EPOCHS) and \
        all(abs(g[f"shell_fit_{k}"]) <= abs(ctl[f"shell_fit_{k}"]) + 3.0 for k in SHELL_EPOCHS)
    g2 = g["size_offsets_pass"] >= ctl["size_offsets_pass"]
    g3 = all(abs(g[f"leak_{k}"] - leak_truth[k]) <= abs(ctl[f"leak_{k}"] - leak_truth[k]) + 0.01 for k in range(1, 5)) \
        and abs(g["rho_recent_z2_5"]) <= 0.15
    return g1, g2, g3


def main(smoke=False, grid=None, retune=True):
    grid = grid or DEFAULT_GRID
    tag = "_smoke" if smoke else ""
    print(f"{RULE}\nexp86 STAGE 0 — where the delayed mass lands: the shell's anatomy and the frozen probe of arrival sizing (w_arr) on the adopted mean"
          f"{' (SMOKE)' if smoke else ''}{'' if retune else ' (NO RE-TUNE)'}\n{RULE}\n")
    recs, data, mask, lmh_dm, lmh_bins, meas, fz, spec2, th_inc, th_nested, pr, lp, bounds = S1.build(smoke, KNOBS, True, "step")
    ad = RB.adopted_mean()
    th_ad = np.r_[ad["theta_full"], 0.0]
    assert list(lp.names) == ad["names"] + ["w_arr"], lp.names
    print(f"  the adopted mean: {ad['file'].name}\n    {lp.describe(th_ad)}")
    rows = pr.all_rows
    truth_all = data[rows]
    good = np.isfinite(truth_all).all(axis=(1, 2)) & (truth_all > 0).all(axis=(1, 2))
    truth = truth_all[good]
    lt_truth = np.log10(truth)
    cvg = [c for c, g in zip([meas[i] for i in rows], good) if g]
    n = len(cvg)
    i_2, i_10, i_o = (int(np.argmin(np.abs(F.R_GRID - r))) for r in (R_CORE, R_10, R_OUTER))
    sn = np.load(SEL_NPZ, allow_pickle=True)
    hs = np.load(SEL.HS_NPZ, allow_pickle=True)
    lmh_cat = SEL.sample_masses(hs)[np.array([h.row for h in recs])][rows][good]
    complete = np.isfinite(lmh_cat) & (lmh_cat >= np.asarray(sn["cuts"], float)[None, :])
    growth = lmh_cat[:, 0][:, None] - lmh_cat
    lmh_bin = np.asarray(lmh_bins, float)[rows][good]
    feats = {}
    for k in EPOCHS:
        lt_k = np.log10(E.T_ANCHOR[k])
        lm_k = np.array([E.log_mah(np.array([lt_k]), hc)[0] for hc in cvg])
        lm_prev = np.array([E.log_mah(np.array([np.log10(E.T_ANCHOR[k] - 1.0)]), hc)[0] for hc in cvg])
        feats[k] = dict(lmh=lm_k, recent=lm_k - lm_prev)
    ix = {nm: list(lp.names).index(nm) for nm in ("a0", "log_f_e", "b_e", "w_arr")}
    p_ad = spec2.unpack(th_ad[:spec2.n_theta])
    print(f"  {n} galaxies of the fitting sample; mh-complete per epoch " + "/".join(str(int(complete[:, k].sum())) for k in EPOCHS)
          + f"; tau_d {p_ad['tau_d']:.3f} (step arrival)")

    def predict(th, nodes=M2.FULL_NODES):
        return np.clip(lp.predict(th, F.R_GRID, nodes=nodes)[good], 1.0, None)

    t0 = time.time()
    m_ad = predict(th_ad)
    print(f"  one full-node prediction {time.time() - t0:.0f} s")

    # ---- 1. the anatomy ------------------------------------------------------- #
    print(f"\n{RULE}\n1. THE ANATOMY OF THE 50-100 kpc SHELL in the adopted mean, by channel and by sample, and the arrival factor of its deposits\n{RULE}")
    lt, w, include, _ = E.nodes(**M2.FULL_NODES)
    th13, law = lp.split(th_ad)
    lt_a = SL.arrival_time(spec2, p_ad, lt)
    Rsh = np.array(R_SHELL)
    sh_c, sh_e = np.zeros((n, 5)), np.zeros((n, 5))
    fac, fac_all = np.full((n, 5), np.nan), np.full((n, 5), np.nan)
    delay_gyr = np.full((n, 5), np.nan)
    for lo in range(0, n, 300):
        cv = cvg[lo:lo + 300]
        dm_c, dm_e, s_c, s_e, r_tr = SL.deposits_law(spec2, th13, law, cv, lt, EPOCHS)
        nb, N = dm_c.shape
        r200 = np.array([E.r200c_of(hc, lt, "analytic") for hc in cv])
        r200_a = np.array([E.r200c_of(hc, lt_a, "analytic") for hc in cv])
        lfac = np.log10(r200_a / r200)
        r_tr_c = spec2.trunc_C * r200
        Bc = M.cog_truncated(M2.COMPACT_FAMILY, (p_ad["n_c"],), s_c.ravel(), r_tr_c.ravel(), Rsh).reshape(2, nb, N)
        inc_e = SL.arrival_weights(spec2, p_ad, law, lt, include)
        for k in EPOCHS:
            Be = M.cog_truncated(spec2.extended_family, (p_ad["c_e"],), s_e[k].ravel(), r_tr[k].ravel(), Rsh).reshape(2, nb, N)
            wc_ = dm_c * w[None, :] * include[k][None, :]
            we_ = dm_e * w[None, :] * inc_e[k][None, :]
            sh_c[lo:lo + nb, k] = ((Bc[1] - Bc[0]) * wc_).sum(1)
            e_node = (Be[1] - Be[0]) * we_
            sh_e[lo:lo + nb, k] = e_node.sum(1)
            fac[lo:lo + nb, k] = (e_node * lfac).sum(1) / np.maximum(e_node.sum(1), 1e-30)
            fac_all[lo:lo + nb, k] = (we_ * lfac).sum(1) / np.maximum(we_.sum(1), 1e-30)
            delay_gyr[lo:lo + nb, k] = (e_node * (10.0 ** lt_a - 10.0 ** lt)[None, :]).sum(1) / np.maximum(e_node.sum(1), 1e-30)
    sh_t = shell(truth, F.R_GRID)
    print(f"  {'reading (median; fitting sample | mh-complete)':<78}" + "".join(f"{'z = ' + str(ANCHOR_Z[k]):>18}" for k in EPOCHS))

    def row(label, arr, fmt="+.3f"):
        cells = []
        for k in EPOCHS:
            c = complete[:, k]
            cells.append(f"{np.nanmedian(arr[:, k]):{fmt}} | {np.nanmedian(arr[c, k]):{fmt}}")
        print(f"  {label:<78}" + "".join(f"{s:>18}" for s in cells))
    row("shell (model - data)/data [%]", 100 * ((sh_c + sh_e) - sh_t) / sh_t, "+.1f")
    row("extended-channel share of the model's shell mass", sh_e / (sh_c + sh_e), ".2f")
    row("log10 M_shell model [Msun]", np.log10(sh_c + sh_e))
    row("log10 M_shell data [Msun]", np.log10(sh_t))
    row("arrival factor log10 R200c(t_a)/R200c(t') of the extended deposits IN THE SHELL (mass-weighted)", fac)
    row("the same for the WHOLE extended channel arrived by the epoch", fac_all)
    row("mean delay of the shell's deposits, t_a - t' [Gyr]", delay_gyr, ".2f")
    for k in SHELL_EPOCHS:
        ok = np.isfinite(sh_t[:, k]) & (sh_t[:, k] > 0) & np.isfinite(fac[:, k])
        c = complete[:, k] & ok
        r_ = ~complete[:, k] & ok
        res = np.full(n, np.nan)
        res[ok] = np.log10((sh_c[ok, k] + sh_e[ok, k]) / sh_t[ok, k])
        print(f"  z={ANCHOR_Z[k]}: rank correlation of the shell residual with the arrival factor: all {spearmanr(res[ok], fac[ok, k])[0]:+.2f}, "
              f"mh-complete {spearmanr(res[c], fac[c, k])[0]:+.2f}, the rest {spearmanr(res[r_], fac[r_, k])[0]:+.2f}; the factor's median, complete minus the rest: "
              f"{np.median(fac[c, k]) - np.median(fac[r_, k]):+.3f} dex (what the knob can use); the shell residual's median, complete vs the rest: "
              f"{np.median(res[c]):+.3f} vs {np.median(res[r_]):+.3f} dex ({int((~ok).sum())} zero-shell truths dropped)")

    # ---- 2. the probe ---------------------------------------------------------- #
    print(f"\n{RULE}\n2. THE FROZEN-THETA PROBE — (log_f_e, b_e) re-tuned on the radius term (the control at w_arr = 0), a0 re-centred on M(<103) at z=0.4\n{RULE}")
    leak_truth = {k: SEL.partial_growth(lt_truth[:, k, i_o], lmh_cat[:, k], growth[:, k], np.isfinite(lmh_cat[:, k]))[1] for k in range(1, 5)}
    ref = gate_readings(m_ad, truth, lt_truth, None, complete, lmh_cat, growth, lmh_bin, feats, i_2, i_10, i_o)
    ref["loss"] = lp.loss(th_ad)
    for k in range(1, 5):
        ref[f"leak_truth_{k}"] = leak_truth[k]
    print_gates("the adopted mean (w_arr = 0, constants as fitted)", ref, ref)

    def radius_rms(th):
        m = predict(th, nodes=M2.FIT_NODES)
        raw = np.array([ST.radius_term(m[:, k], truth[:, k], F.R_GRID, lmh_bin[:, k])[0] for k in EPOCHS])
        return float(np.sqrt(np.mean(raw ** 2)))

    def retune_extended(th):
        th = th.copy()
        x0 = np.array([th[ix["log_f_e"]], th[ix["b_e"]]])
        lo = np.array([bounds[ix["log_f_e"]][0], bounds[ix["b_e"]][0]])
        hi = np.array([bounds[ix["log_f_e"]][1], bounds[ix["b_e"]][1]])
        n_ev = [0]

        def f(x):
            n_ev[0] += 1
            xc = np.clip(x, lo, hi)
            th[ix["log_f_e"]], th[ix["b_e"]] = xc
            return radius_rms(th) + 10.0 * float(np.sum((x - xc) ** 2))
        r = minimize(f, x0, method="Nelder-Mead", options=dict(maxfev=RETUNE_MAX_EVALS["smoke" if smoke else "full"], xatol=2e-3, fatol=1e-4,
                                                                 initial_simplex=np.array([x0, x0 + [0.1, 0.0], x0 + [0.0, 0.3]])))
        th[ix["log_f_e"]], th[ix["b_e"]] = np.clip(r.x, lo, hi)
        return th, float(r.fun), n_ev[0]

    results = {"adopted": ref}
    for wv in [0.0] + list(grid):
        label = "control (w_arr = 0, re-tuned)" if wv == 0.0 else f"w_arr = {wv:.2f}"
        th = th_ad.copy()
        th[ix["w_arr"]] = float(wv)
        t0 = time.time()
        zr0 = radius_rms(th)
        if retune:
            th, zr, n_ev = retune_extended(th)
            print(f"  [{label}] re-tuned in {n_ev} evaluations ({time.time() - t0:.0f} s): radius term {zr0:.4f} -> {zr:.4f}; "
                  f"log_f_e {th_ad[ix['log_f_e']]:+.3f} -> {th[ix['log_f_e']]:+.3f}, b_e {th_ad[ix['b_e']]:+.3f} -> {th[ix['b_e']]:+.3f}")
        else:
            zr = zr0
        m = predict(th)
        shift = float(np.median(np.log10(m[:, 0, i_o]) - np.log10(m_ad[:, 0, i_o])))
        th[ix["a0"]] -= shift
        m = m * 10.0 ** (-shift)
        g = gate_readings(m, truth, lt_truth, None, complete, lmh_cat, growth, lmh_bin, feats, i_2, i_10, i_o)
        g["loss"] = lp.loss(th)
        g.update(log_f_e=float(th[ix["log_f_e"]]), b_e=float(th[ix["b_e"]]), zrms=zr, shift=shift, w_arr=float(wv))
        for k in range(1, 5):
            g[f"leak_truth_{k}"] = leak_truth[k]
        ctl = results.get("control", None)
        print_gates(f"{label} (a0 re-centred by {-shift:+.3f})", g, ctl if ctl is not None else ref)
        if wv == 0.0:
            results["control"] = g
            print("    GATES for a probe, against the control: G1 the mh-complete shell at z=1.5 / 2 within "
                  + " / ".join(f"{0.5 * abs(g[f'shell_mhc_{k}']):.1f}" for k in SHELL_EPOCHS)
                  + " points of zero with the fitting sample's within +3 of " + " / ".join(f"{abs(g[f'shell_fit_{k}']):.1f}" for k in SHELL_EPOCHS)
                  + f"; G2 size cells >= {g['size_offsets_pass']}; G3 the leak gate at no epoch worse than the control's by more than 0.01, rho(recent) z=2 <= 0.15")
        else:
            g1, g2, g3 = gate_verdict(g, results["control"], leak_truth)
            print(f"    GATE {'PASS' if (g1 and g2 and g3) else 'fail'}: G1 shell halved on the complete sample without pushing the fitting sample {g1}, "
                  f"G2 size cells kept {g2}, G3 leak and recent-growth {g3}")
            results[label] = g
    OUTDIR.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for g in results.values() for k in g})
    np.savez(OUTDIR / f"stage0_probe{tag}.npz", probe_keys=np.array(list(results)), gate_keys=np.array(keys),
             gates=np.array([[results[p].get(k, np.nan) for k in keys] for p in results]),
             shell_model_c=sh_c, shell_model_e=sh_e, shell_truth=sh_t, arrival_factor=fac, arrival_factor_all=fac_all,
             complete=complete, theta_adopted=th_ad, names=np.array(lp.names))
    print(f"\nwrote {OUTDIR / f'stage0_probe{tag}.npz'}")


if __name__ == "__main__":
    a = sys.argv
    grid = None
    if "--grid" in a:
        grid = []
        for item in a[a.index("--grid") + 1:]:
            if item.startswith("--"):
                break
            grid.append(float(item))
    main(smoke="--smoke" in a, grid=grid, retune="--no-retune" not in a)
