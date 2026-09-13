"""exp83 Stage 0 — the early-mass residual of the ADOPTED mean (exp82: the
delay + expansion model, `rebaseline.adopted_mean()`), its functional form,
and the FROZEN-THETA probe of one early-mass term. No fit here.

Section 1  the adopted model's residual r = log10(model/truth) on the fitting
           sample at the cells that matter (z = 0.4 at 5 and 103 kpc, z >= 1.5
           at 5 / 33 / 103 kpc): the partial Spearman at fixed log Mh with the
           early-mass fraction x = log M(2 Gyr) - log Mh(z_k) and with the
           recent growth
           the slope of r on x at fixed log Mh next to the
           truth's own slope
           r's binned medians in quintiles of x per
           halo-mass tercile (the FORM: linear, saturating, or a tail)
           and
           the same split by the central decline (exp81 measurement 1, section
           3), with the decliner fraction per quintile of x.
Section 2  the probe. ONE per-deposit variable, phi(t') = min(log M(2 Gyr) -
           log M(t'), 0) (`size_law.early_fraction`), placed either in the
           efficiency (`a_early`: eps x 10^(a_early phi)), in the compact
           share's logit (`s_early`) or on the compact size (`c_early`, the
           roadmap's third candidate), each nesting at zero. The adopted theta
           is held
           each probe value's prediction is re-centred by a0 so the
           fitting sample's median M*(<103 kpc) at z = 0.4 is unchanged (a0
           scales every deposit, so this is exact and costs no prediction).
           Read per probe: the z = 0.4 early-mass correlation (GATE: halved),
           the decliners' central change z = 2 -> 0.4 against the truth's
           (GATE: the mismatch reduced), the non-decliners' z = 2 centre median
           (GATE: moved by at most 0.02 dex), the future-dependence gate at
           103 kpc, the recent-growth correlation (must stay closed), the
           centre M(<2) per cent at z = 0.4 / 1.5 / 2, and the standard loss
           at the fit nodes for the record.

Run (one process
a full-node prediction of the fitting sample per probe):
    HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 PYTHONPATH=. \\
        uv run python -u experiments/exp83_early_mass_centre/stage0_probe.py [--smoke] [--grid a_early=-0.3 c_early=0.5 a_early=-0.3,c_early=0.5]
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
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
import model2 as M2                                      # noqa: E402
import selection as SEL                                  # noqa: E402
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
#: the size-law knobs the probe theta carries: the adopted q_e, then the two placements of the early-mass term
KNOBS = ("q_e", "a_early", "s_early", "c_early")
CELLS = [(0, 4.92), (0, 103.45), (2, 103.45), (3, 4.92), (3, 32.58), (3, 103.45), (4, 4.92), (4, 32.58), (4, 103.45)]
R_CENTRE, R_OUTER, R_CORE = 4.92, 103.45, 2.0
#: each probe point is {knob: value}; points may carry more than one knob (the combination is read for the record)
DEFAULT_GRID = [{"a_early": v} for v in (-0.4, -0.3, -0.2, -0.15, -0.1, 0.1)] + [{"s_early": v} for v in (-2.0, 2.0)] \
    + [{"c_early": v} for v in (0.3, 0.6, 1.0)] + [{"a_early": -0.3, "c_early": v} for v in (0.5, 1.0)]


def resid_on(y, x, lmh):
    """Both stripped of a quadratic in log Mh."""
    ok = np.isfinite(y) & np.isfinite(x) & np.isfinite(lmh)
    X = np.column_stack([np.ones(ok.sum()), lmh[ok], lmh[ok] ** 2])
    ry = y[ok] - X @ np.linalg.lstsq(X, y[ok], rcond=None)[0]
    rx = x[ok] - X @ np.linalg.lstsq(X, x[ok], rcond=None)[0]
    return ry, rx


def partial(y, x, lmh):
    ry, rx = resid_on(y, x, lmh)
    return float(spearmanr(ry, rx)[0])


def slope(y, x, lmh):
    ry, rx = resid_on(y, x, lmh)
    return float(np.polyfit(rx, ry, 1)[0])


def quintile_medians(y, x, lmh, n_min=15):
    """Median of y in quintiles of x, per halo-mass tercile: (3, 5) with NaN where thin."""
    e = np.quantile(lmh, [0, 1 / 3, 2 / 3, 1])
    q = np.quantile(x, np.linspace(0, 1, 6))
    out = np.full((3, 5), np.nan)
    for b in range(3):
        t = (lmh >= e[b]) & (lmh <= e[b + 1] + 1e-9)
        for j in range(5):
            s = t & (x >= q[j]) & (x <= q[j + 1])
            if s.sum() >= n_min:
                out[b, j] = np.median(y[s])
    return out, q


def features(cvg):
    """Per-epoch halo variables of the fitting sample from the measured curves."""
    feats = {}
    lm_2 = np.array([E.log_mah(np.array([np.log10(SL.T_EARLY_GYR)]), hc)[0] for hc in cvg])
    for k in EPOCHS:
        lt_k = np.log10(E.T_ANCHOR[k])
        t_k = E.T_ANCHOR[k]
        lm_k = np.array([E.log_mah(np.array([lt_k]), hc)[0] for hc in cvg])
        lm_prev = np.array([E.log_mah(np.array([np.log10(t_k - 1.0)]), hc)[0] for hc in cvg])
        feats[k] = dict(lmh=lm_k, early2=lm_2 - lm_k, recent=lm_k - lm_prev)
    return feats


def gate_readings(r, m, truth, feats, dec, lmh_cat, growth, i_c, i_o, i_2):
    """The numbers every probe is read by (all on the fitting sample)."""
    g = {}
    g["rho_early_z04_5"] = partial(r[:, 0, i_c], feats[0]["early2"], feats[0]["lmh"])
    g["rho_early_z04_103"] = partial(r[:, 0, i_o], feats[0]["early2"], feats[0]["lmh"])
    g["rho_early_z15_5"] = partial(r[:, 3, i_c], feats[3]["early2"], feats[3]["lmh"])
    g["rho_early_z2_5"] = partial(r[:, 4, i_c], feats[4]["early2"], feats[4]["lmh"])
    g["rho_early_z2_103"] = partial(r[:, 4, i_o], feats[4]["early2"], feats[4]["lmh"])
    g["rho_recent_z04_103"] = partial(r[:, 0, i_o], feats[0]["recent"], feats[0]["lmh"])
    g["rho_recent_z2_5"] = partial(r[:, 4, i_c], feats[4]["recent"], feats[4]["lmh"])
    g["dec_z04"] = float(np.median(r[dec, 0, i_c]))
    g["dec_z2"] = float(np.median(r[dec, 4, i_c]))
    g["non_z04"] = float(np.median(r[~dec, 0, i_c]))
    g["non_z2"] = float(np.median(r[~dec, 4, i_c]))
    g["dec_change_model"] = float(np.median(np.log10(m[dec, 0, i_c] / m[dec, 4, i_c])))
    g["dec_change_truth"] = float(np.median(np.log10(truth[dec, 0, i_c] / truth[dec, 4, i_c])))
    g["non_change_model"] = float(np.median(np.log10(m[~dec, 0, i_c] / m[~dec, 4, i_c])))
    g["non_change_truth"] = float(np.median(np.log10(truth[~dec, 0, i_c] / truth[~dec, 4, i_c])))
    for k in EPOCHS:
        g[f"core_pct_{k}"] = 100 * float(np.median((m[:, k, i_2] - truth[:, k, i_2]) / truth[:, k, i_2]))
        g[f"med_{k}_103"] = float(np.median(r[:, k, i_o]))
        g[f"rms_{k}_5"] = float(np.sqrt(np.mean(r[:, k, i_c] ** 2)))
    for k in range(1, 5):
        y = r[:, k, i_o]
        ok = np.isfinite(lmh_cat[:, k])
        g[f"leak_{k}"] = SEL.partial_growth(y, lmh_cat[:, k], growth[:, k], ok)[1]
    return g


def print_gates(label, g, ref=None):
    def d(key, fmt="+.3f"):
        v = g[key]
        return f"{v:{fmt}}" + (f" ({v - ref[key]:+.3f})" if ref is not None and key in ref else "")
    print(f"  --- {label} ---")
    print(f"    rho(early) at fixed Mh: z=0.4 5 kpc {d('rho_early_z04_5', '+.2f')}, 103 kpc {d('rho_early_z04_103', '+.2f')}; "
          f"z=1.5 5 kpc {d('rho_early_z15_5', '+.2f')}; z=2 5 kpc {d('rho_early_z2_5', '+.2f')}, 103 kpc {d('rho_early_z2_103', '+.2f')}")
    print(f"    rho(recent): z=0.4 103 kpc {d('rho_recent_z04_103', '+.2f')}; z=2 5 kpc {d('rho_recent_z2_5', '+.2f')}")
    print(f"    centre 4.9 kpc medians log10(model/truth): decliners z=0.4 {d('dec_z04')} z=2 {d('dec_z2')}; "
          f"non-decliners z=0.4 {d('non_z04')} z=2 {d('non_z2')}")
    print(f"    the decliners' change z=2 -> 0.4: model {d('dec_change_model')} vs truth {g['dec_change_truth']:+.3f} "
          f"(mismatch {abs(g['dec_change_model'] - g['dec_change_truth']):.3f}); the non-decliners': model {d('non_change_model')} vs truth {g['non_change_truth']:+.3f}")
    print("    M(<2 kpc) median per cent z=0.4/0.7/1/1.5/2: " + " / ".join(f"{g[f'core_pct_{k}']:+.1f}" for k in EPOCHS)
          + ";  M(<103) median dex: " + " / ".join(f"{g[f'med_{k}_103']:+.3f}" for k in EPOCHS)
          + ";  rms at 5 kpc: " + " / ".join(f"{g[f'rms_{k}_5']:.3f}" for k in EPOCHS))
    print("    future-dependence gate at 103 kpc [dex per dex] z=0.7/1/1.5/2: " + " / ".join(d(f'leak_{k}') for k in range(1, 5))
          + ("  (truth " + " / ".join(f"{ref[f'leak_truth_{k}']:+.3f}" for k in range(1, 5)) + ")" if ref is not None else ""))
    if "loss" in g:
        print(f"    standard loss at the fit nodes (a0 re-centred): {g['loss']:.4f}" + (f" ({g['loss'] - ref['loss']:+.3f})" if ref is not None else ""))


def main(smoke=False, grid=None):
    grid = grid or DEFAULT_GRID
    tag = "_smoke" if smoke else ""
    print(f"{RULE}\nexp83 STAGE 0 — the adopted mean's early-mass residual and the frozen-theta probe of the early-mass term"
          f"{' (SMOKE)' if smoke else ''}\n{RULE}\n")
    recs, data, mask, lmh_dm, lmh_bins, meas, fz, spec2, th_inc, th_nested, pr, lp, bounds = S1.build(smoke, KNOBS, True, "step")
    ad = RB.adopted_mean()
    th_ad = np.r_[ad["theta_full"], 0.0, 0.0, 0.0]
    assert list(lp.names) == ad["names"] + ["a_early", "s_early", "c_early"], lp.names
    print(f"  the adopted mean: {ad['file'].name}\n    {lp.describe(th_ad)}")
    rows = pr.all_rows
    truth_all = data[rows]
    good = np.isfinite(truth_all).all(axis=(1, 2)) & (truth_all > 0).all(axis=(1, 2))
    truth = truth_all[good]
    cvg = [c for c, g in zip([meas[i] for i in rows], good) if g]
    i_c = int(np.argmin(np.abs(F.R_GRID - R_CENTRE)))
    i_o = int(np.argmin(np.abs(F.R_GRID - R_OUTER)))
    i_2 = int(np.argmin(np.abs(F.R_GRID - R_CORE)))
    hs = np.load(SEL.HS_NPZ, allow_pickle=True)
    lmh_cat = SEL.sample_masses(hs)[np.array([h.row for h in recs])][rows][good]
    growth = lmh_cat[:, 0][:, None] - lmh_cat
    feats = features(cvg)
    dec = truth[:, 4, i_c] > truth[:, 0, i_c]
    lt_truth = np.log10(truth)

    def predict(th):
        t0 = time.time()
        m = lp.predict(th, F.R_GRID, nodes=M2.FULL_NODES)[good]
        return np.clip(m, 1.0, None), time.time() - t0

    m_ad, dt = predict(th_ad)
    r_ad = np.log10(m_ad) - lt_truth
    print(f"  {len(cvg)} galaxies of the fitting sample with a finite truth at every epoch and radius; one full-node prediction {dt:.0f} s; "
          f"{100 * dec.mean():.0f}% declined at 4.9 kpc from z=2 to 0.4")

    # ---- 1. the residual and its form -------------------------------------- #
    print(f"\n{RULE}\n1. THE ADOPTED MEAN'S RESIDUAL log10(model/truth) AGAINST THE EARLY-MASS FRACTION x = log M(2 Gyr) - log Mh(z_k), at fixed log Mh\n{RULE}")
    print(f"  {'cell':<16}{'rho early | recent':>22}{'slope r on x | truth on x [dex/dex]':>38}{'median':>10}{'rms':>8}")
    for k, R in CELLS:
        i = int(np.argmin(np.abs(F.R_GRID - R)))
        fk = feats[k]
        print(f"  z={ANCHOR_Z[k]} R={R:<6.0f}{partial(r_ad[:, k, i], fk['early2'], fk['lmh']):>+11.2f} | {partial(r_ad[:, k, i], fk['recent'], fk['lmh']):+.2f}"
              f"{slope(r_ad[:, k, i], fk['early2'], fk['lmh']):>+22.3f} | {slope(lt_truth[:, k, i], fk['early2'], fk['lmh']):+.3f}"
              f"{np.median(r_ad[:, k, i]):>+10.3f}{np.sqrt(np.mean(r_ad[:, k, i] ** 2)):>8.3f}")
    print(f"\n  the TWO components at z=0.4: the residual at 5 kpc against x at fixed log Mh AND at fixed residual at 103 kpc "
          f"(the concentration part, what no efficiency term can reach): rho {partial(r_ad[:, 0, i_c] - r_ad[:, 0, i_o], feats[0]['early2'], feats[0]['lmh']):+.2f}, "
          f"slope {slope(r_ad[:, 0, i_c] - r_ad[:, 0, i_o], feats[0]['early2'], feats[0]['lmh']):+.3f} dex per dex "
          f"(the truth's own log M(<5)/M(<103) on x: {slope(lt_truth[:, 0, i_c] - lt_truth[:, 0, i_o], feats[0]['early2'], feats[0]['lmh']):+.3f}); "
          f"the same at z=2: rho {partial(r_ad[:, 4, i_c] - r_ad[:, 4, i_o], feats[4]['early2'], feats[4]['lmh']):+.2f}, "
          f"slope {slope(r_ad[:, 4, i_c] - r_ad[:, 4, i_o], feats[4]['early2'], feats[4]['lmh']):+.3f}")
    print("\n  binned median residual [dex] in quintiles of x, per halo-mass tercile (low / mid / high); the FORM of the dependence")
    for k, R in CELLS:
        i = int(np.argmin(np.abs(F.R_GRID - R)))
        qm, q = quintile_medians(r_ad[:, k, i], feats[k]["early2"], feats[k]["lmh"])
        print(f"  z={ANCHOR_Z[k]} R={R:<6.0f} x edges {'/'.join(f'{v:+.2f}' for v in q)}:  "
              + "   ".join("/".join(f"{v:+.3f}" if np.isfinite(v) else "  nan " for v in row) for row in qm))
    print("\n  the same at 4.9 kpc SPLIT BY THE CENTRAL DECLINE: per quintile of x (all masses) the decliner fraction, "
          "then the median residual of the decliners | the non-decliners")
    for k in (0, 3, 4):
        x = feats[k]["early2"]
        q = np.quantile(x, np.linspace(0, 1, 6))
        cells = []
        for j in range(5):
            s = (x >= q[j]) & (x <= q[j + 1])
            cells.append(f"{100 * dec[s].mean():3.0f}% {np.median(r_ad[s & dec, k, i_c]):+.3f} | {np.median(r_ad[s & ~dec, k, i_c]):+.3f}")
        print(f"  z={ANCHOR_Z[k]} 4.9 kpc  " + "   ".join(cells))
    print("\n  the deposit variable phi(t') = min(log M(2 Gyr) - log M(t'), 0) at the full nodes, weighted by the adopted model's deposited "
          "stellar mass by z=0.4: quantiles 10/25/50/75/90")
    lt, w, include, _ = E.nodes(**M2.FULL_NODES)
    dm_c, dm_e, *_ = SL.deposits_law(lp.spec2, th_ad[:lp.spec2.n_theta], lp.split(th_ad)[1], cvg, lt, (0,))
    wt = (dm_c + dm_e) * w[None, :] * include[0][None, :]
    lm_nodes = np.array([E.log_mah(lt, hc) for hc in cvg])
    phi_raw = SL.early_fraction_raw(cvg, lm_nodes)
    phi = SL.early_fraction(cvg, lm_nodes, lt)
    for nm, ph in (("raw phi", phi_raw), ("phi - phi_ref(t') (the term's variable)", phi)):
        order = np.argsort(ph.ravel())
        cw = np.cumsum(wt.ravel()[order])
        cw /= cw[-1]
        qs = [ph.ravel()[order][np.searchsorted(cw, f)] for f in (0.1, 0.25, 0.5, 0.75, 0.9)]
        print(f"    {nm:<40} " + " / ".join(f"{v:+.2f}" for v in qs))
    print(f"    mass fraction deposited before 2 Gyr {float((wt * (phi_raw == 0)).sum() / wt.sum()):.2f}; phi_ref(t') at 2 / 3.3 / 5.9 / 9.3 Gyr "
          + " / ".join(f"{SL.early_ref(np.log10(t)):+.2f}" for t in (2.0, 3.3, 5.9, 9.3)))

    # ---- 2. the probe --------------------------------------------------------- #
    print(f"\n{RULE}\n2. THE FROZEN-THETA PROBE — the adopted theta with ONE early-mass term, a0 re-centred to hold the z=0.4 median M(<103)\n{RULE}")
    ref = gate_readings(r_ad, m_ad, truth, feats, dec, lmh_cat, growth, i_c, i_o, i_2)
    ref["loss"] = lp.loss(th_ad)
    for k in range(1, 5):
        ref[f"leak_truth_{k}"] = SEL.partial_growth(lt_truth[:, k, i_o], lmh_cat[:, k], growth[:, k], np.isfinite(lmh_cat[:, k]))[1]
    print_gates("the adopted mean (a_early = s_early = 0)", ref, ref)
    print(f"    GATES for a probe: rho(early) at z=0.4 halved (5 kpc <= {ref['rho_early_z04_5'] / 2:+.2f}, 103 kpc <= {ref['rho_early_z04_103'] / 2:+.2f}); "
          f"the decliners' mismatch below {abs(ref['dec_change_model'] - ref['dec_change_truth']):.3f}; the non-decliners' z=2 median within 0.02 of {ref['non_z2']:+.3f}")
    results = {"adopted": ref}
    for point in grid:
        label = ", ".join(f"{k} = {v:+.2f}" for k, v in point.items())
        knob, v = list(point.items())[0]
        if True:
            th = th_ad.copy()
            for k_, v_ in point.items():
                th[list(lp.names).index(k_)] = float(v_)
            m, dt = predict(th)
            shifts = [float(np.median(np.log10(m[:, k, i_o]) - np.log10(m_ad[:, k, i_o]))) for k in EPOCHS]
            shift = shifts[0]
            th[list(lp.names).index("a0")] -= shift
            m = m * 10.0 ** (-shift)
            r = np.log10(m) - lt_truth
            print(f"  [{label}] the term's own median shift of log M(<103) per epoch before re-centring: "
                  + " / ".join(f"{x:+.3f}" for x in shifts) + " (a time trend here is what the fit's a_z would absorb)")
            g = gate_readings(r, m, truth, feats, dec, lmh_cat, growth, i_c, i_o, i_2)
            g["loss"] = lp.loss(th)
            g["shift"] = shift
            moved = float(np.max(np.abs(np.log10(m / m_ad))))
            print_gates(f"{label} (a0 re-centred by {-shift:+.3f}; {dt:.0f} s; the profile moved by up to {moved:.3f} dex)", g, ref)
            passed = (abs(g["rho_early_z04_5"]) <= abs(ref["rho_early_z04_5"]) / 2 and abs(g["rho_early_z04_103"]) <= abs(ref["rho_early_z04_103"]) / 2
                      and abs(g["dec_change_model"] - g["dec_change_truth"]) < abs(ref["dec_change_model"] - ref["dec_change_truth"])
                      and abs(g["non_z2"] - ref["non_z2"]) <= 0.02)
            print(f"    GATE {'PASS' if passed else 'fail'}: early z=0.4 halved {abs(g['rho_early_z04_5']) <= abs(ref['rho_early_z04_5']) / 2 and abs(g['rho_early_z04_103']) <= abs(ref['rho_early_z04_103']) / 2}, "
                  f"decliners' mismatch reduced {abs(g['dec_change_model'] - g['dec_change_truth']) < abs(ref['dec_change_model'] - ref['dec_change_truth'])}, "
                  f"non-decliners' z=2 within 0.02 {abs(g['non_z2'] - ref['non_z2']) <= 0.02}")
            results[label] = g
    OUTDIR.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for g in results.values() for k in g})
    np.savez(OUTDIR / f"stage0_probe{tag}.npz", r_adopted=r_ad, lt_truth=lt_truth, dec=dec, rows=rows[good],
             theta_adopted=th_ad, names=np.array(lp.names), probe_keys=np.array(list(results)), gate_keys=np.array(keys),
             gates=np.array([[results[p].get(k, np.nan) for k in keys] for p in results]),
             **{f"{k}_{nm}": feats[k][nm] for k in EPOCHS for nm in ("lmh", "early2", "recent")})
    print(f"\nwrote {OUTDIR / f'stage0_probe{tag}.npz'}")


if __name__ == "__main__":
    a = sys.argv
    grid = None
    if "--grid" in a:
        # each item is one probe point: knob=value[,knob=value...]
        grid = []
        for item in a[a.index("--grid") + 1:]:
            if item.startswith("--"):
                break
            grid.append({kv.split("=")[0]: float(kv.split("=")[1]) for kv in item.split(",")})
    main(smoke="--smoke" in a, grid=grid)
