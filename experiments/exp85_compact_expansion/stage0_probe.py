"""exp85 Stage 0 — the FROZEN-THETA probe of the compact channel's
post-deposition expansion (the centre's mechanism) on THE ADOPTED MEAN
(exp82: the delay + expansion model, `rebaseline.adopted_mean()`). No fit.

The knob: a compact deposit made at t' evaluated at the observed epoch t_obs
has size s_c(t') x (t_obs / t')^q_c (`size_law` knob `q_c`, age driven, the
candidate) or s_c(t') x (R200c(t_obs) / R200c(t'))^q_ch (`q_ch`, halo
driven, the control form). Both nest at zero.

At every probe point the compact constants (log_f_c, b_c) are RE-TUNED so
the fitting sample's median concentration log M(<4.9) - log M(<103) at
z = 0.4 AND at z = 2 is the adopted mean's (a 2-D Newton root find; what the
fit's shape terms would do first), then a0 is re-centred to hold the z = 0.4
median M(<103) (exact: a0 scales every deposit). The probe then reads the
REDISTRIBUTION of central mass between galaxies, not a uniform shift.

Read per probe (the plan's gates): G1 the split of the central change
z = 2 -> 0.4 between the decliners and the non-decliners (truth 0.226 dex,
the adopted mean 0.063; PASS at >= half way with both groups' mismatches
reduced); G2 the concentration part of the z = 0.4 early-mass residual
(5 kpc at fixed Mh and fixed M(<103); PASS when halved); G3 the
non-decliners' z = 2 centre median within 0.02 of the adopted mean's.
Reported: the early-mass and recent-growth correlations, the
future-dependence gate at 103 kpc, the centre M(<2) per cent per epoch,
the tercile-median R20 / R50 / R80 offsets per epoch at fixed halo mass
(`size_terms.radius_term`), the standard loss at the fit nodes.

Run (one process; a handful of full-node predictions per probe point):
    HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 PYTHONPATH=. \\
        uv run python -u experiments/exp85_compact_expansion/stage0_probe.py [--smoke] [--no-retune] \\
        [--grid q_c=0.3 q_ch=0.3 q_c=0.3,q_ch=0.1]
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
#: the size-law knobs the probe theta carries: the adopted q_e, then the two forms of the compact expansion
KNOBS = ("q_e", "q_c", "q_ch")
R_CENTRE, R_OUTER, R_CORE = 4.92, 103.45, 2.0
CELLS = [(0, 4.92), (0, 103.45), (3, 4.92), (4, 4.92), (4, 103.45)]
DEFAULT_GRID = [{"q_c": v} for v in (0.1, 0.2, 0.3, 0.5, 0.8)] + [{"q_ch": v} for v in (0.1, 0.2, 0.3, 0.5)]
#: the re-tune: the concentration at these epochs held at the adopted mean's
RETUNE_EPOCHS = (0, 4)
RETUNE_TOL, RETUNE_STEP, RETUNE_MAX_ITER = 3e-3, 0.02, 12
#: the largest Newton step per iteration in (log_f_c, b_c)
RETUNE_STEP_CAP = np.array([0.3, 0.6])
SIZE_OFFSET_TOL = 0.05


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


def gate_readings(r, m, truth, feats, dec, lmh_cat, growth, lmh_bin, i_c, i_o, i_2):
    """The numbers every probe is read by (all on the fitting sample)."""
    g = {}
    for k, R, nm in ((0, i_c, "z04_5"), (0, i_o, "z04_103"), (3, i_c, "z15_5"), (4, i_c, "z2_5"), (4, i_o, "z2_103")):
        g[f"rho_early_{nm}"] = partial(r[:, k, R], feats[k]["early2"], feats[k]["lmh"])
    # the concentration part: the residual at 5 kpc at fixed Mh AND fixed residual at 103 kpc
    for k, nm in ((0, "z04"), (4, "z2")):
        g[f"rho_conc_{nm}"] = partial(r[:, k, i_c] - r[:, k, i_o], feats[k]["early2"], feats[k]["lmh"])
        g[f"slope_conc_{nm}"] = slope(r[:, k, i_c] - r[:, k, i_o], feats[k]["early2"], feats[k]["lmh"])
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
    g["split_model"] = g["non_change_model"] - g["dec_change_model"]
    g["split_truth"] = g["non_change_truth"] - g["dec_change_truth"]
    g["dec_mismatch"] = abs(g["dec_change_model"] - g["dec_change_truth"])
    g["non_mismatch"] = abs(g["non_change_model"] - g["non_change_truth"])
    # the per-galaxy central change: how much of its scatter the model follows
    dch_m = np.log10(m[:, 0, i_c] / m[:, 4, i_c])
    dch_t = np.log10(truth[:, 0, i_c] / truth[:, 4, i_c])
    g["rho_change"] = float(spearmanr(dch_m, dch_t)[0])
    g["sd_change_model"], g["sd_change_truth"] = float(np.std(dch_m)), float(np.std(dch_t))
    for k in EPOCHS:
        g[f"core_pct_{k}"] = 100 * float(np.median((m[:, k, i_2] - truth[:, k, i_2]) / truth[:, k, i_2]))
        g[f"med_{k}_5"] = float(np.median(r[:, k, i_c]))
        g[f"med_{k}_103"] = float(np.median(r[:, k, i_o]))
        g[f"rms_{k}_5"] = float(np.sqrt(np.mean(r[:, k, i_c] ** 2)))
        _, med, _ = ST.radius_term(m[:, k], truth[:, k], F.R_GRID, lmh_bin[:, k])
        for i, f in enumerate(("R20", "R50", "R80")):
            g[f"size_{f}_{k}"] = float(np.nanmax(np.abs(med[i])))
    g["size_offsets_pass"] = int(sum(g[f"size_{f}_{k}"] <= SIZE_OFFSET_TOL for k in EPOCHS for f in ("R20", "R50", "R80")))
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
    print(f"    G1 the central change z=2 -> 0.4 at 4.9 kpc: decliners model {d('dec_change_model')} vs truth {g['dec_change_truth']:+.3f} "
          f"(mismatch {g['dec_mismatch']:.3f}); non-decliners model {d('non_change_model')} vs truth {g['non_change_truth']:+.3f} "
          f"(mismatch {g['non_mismatch']:.3f}); the SPLIT model {d('split_model')} vs truth {g['split_truth']:+.3f}")
    print(f"       per galaxy: rank correlation of the model's change with the truth's {d('rho_change', '+.2f')}; "
          f"its scatter {g['sd_change_model']:.3f} vs the truth's {g['sd_change_truth']:.3f} dex")
    print(f"    G2 the concentration part of the early-mass residual (5 kpc at fixed Mh and M(<103)): z=0.4 rho {d('rho_conc_z04', '+.2f')} "
          f"slope {d('slope_conc_z04')}; z=2 rho {d('rho_conc_z2', '+.2f')} slope {d('slope_conc_z2')}")
    print(f"    G3 centre 4.9 kpc medians log10(model/truth): decliners z=0.4 {d('dec_z04')} z=2 {d('dec_z2')}; "
          f"non-decliners z=0.4 {d('non_z04')} z=2 {d('non_z2')}")
    print(f"    rho(early) at fixed Mh: z=0.4 5 kpc {d('rho_early_z04_5', '+.2f')}, 103 kpc {d('rho_early_z04_103', '+.2f')}; "
          f"z=1.5 5 kpc {d('rho_early_z15_5', '+.2f')}; z=2 5 kpc {d('rho_early_z2_5', '+.2f')}, 103 kpc {d('rho_early_z2_103', '+.2f')};  "
          f"rho(recent): z=0.4 103 kpc {d('rho_recent_z04_103', '+.2f')}; z=2 5 kpc {d('rho_recent_z2_5', '+.2f')}")
    print("    M(<2 kpc) median per cent z=0.4/0.7/1/1.5/2: " + " / ".join(f"{g[f'core_pct_{k}']:+.1f}" for k in EPOCHS)
          + ";  M(<5) median dex: " + " / ".join(f"{g[f'med_{k}_5']:+.3f}" for k in EPOCHS)
          + ";  M(<103): " + " / ".join(f"{g[f'med_{k}_103']:+.3f}" for k in EPOCHS)
          + ";  rms at 5 kpc: " + " / ".join(f"{g[f'rms_{k}_5']:.3f}" for k in EPOCHS))
    print("    size offsets at fixed Mh, max |tercile median| [dex] R20 / R50 / R80 per epoch: "
          + "   ".join(f"z={ANCHOR_Z[k]} " + "/".join(f"{g[f'size_{f}_{k}']:.3f}" for f in ("R20", "R50", "R80")) for k in EPOCHS)
          + f";  cells within {SIZE_OFFSET_TOL}: {g['size_offsets_pass']}/15" + (f" ({g['size_offsets_pass'] - ref['size_offsets_pass']:+d})" if ref is not None else ""))
    print("    future-dependence gate at 103 kpc [dex per dex] z=0.7/1/1.5/2: " + " / ".join(d(f'leak_{k}') for k in range(1, 5))
          + ("  (truth " + " / ".join(f"{ref[f'leak_truth_{k}']:+.3f}" for k in range(1, 5)) + ")" if ref is not None else ""))
    if "loss" in g:
        print(f"    standard loss at the fit nodes: {g['loss']:.4f}" + (f" ({g['loss'] - ref['loss']:+.3f})" if ref is not None else ""))


def gate_verdict(g, ref):
    half_way = ref["split_model"] + 0.5 * (ref["split_truth"] - ref["split_model"])
    g1 = g["split_model"] >= half_way and g["dec_mismatch"] < ref["dec_mismatch"] and g["non_mismatch"] < ref["non_mismatch"]
    g2 = abs(g["rho_conc_z04"]) <= abs(ref["rho_conc_z04"]) / 2
    g3 = abs(g["non_z2"] - ref["non_z2"]) <= 0.02
    return g1, g2, g3


def main(smoke=False, grid=None, retune=True):
    grid = grid or DEFAULT_GRID
    tag = "_smoke" if smoke else ""
    print(f"{RULE}\nexp85 STAGE 0 — the frozen-theta probe of the compact channel's post-deposition expansion on the adopted mean"
          f"{' (SMOKE)' if smoke else ''}{'' if retune else ' (NO RE-TUNE of log_f_c, b_c)'}\n{RULE}\n")
    recs, data, mask, lmh_dm, lmh_bins, meas, fz, spec2, th_inc, th_nested, pr, lp, bounds = S1.build(smoke, KNOBS, True, "step")
    ad = RB.adopted_mean()
    th_ad = np.r_[ad["theta_full"], 0.0, 0.0]
    assert list(lp.names) == ad["names"] + ["q_c", "q_ch"], lp.names
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
    lmh_bin = np.asarray(lmh_bins, float)[rows][good]
    assert lmh_bin.shape == (len(cvg), 5), lmh_bin.shape
    feats = features(cvg)
    dec = truth[:, 4, i_c] > truth[:, 0, i_c]
    lt_truth = np.log10(truth)
    ix = {n: list(lp.names).index(n) for n in ("a0", "log_f_c", "b_c", "q_c", "q_ch")}

    def predict(th, epochs=EPOCHS):
        t0 = time.time()
        m = lp.predict(th, F.R_GRID, nodes=M2.FULL_NODES)[good] if epochs == EPOCHS else \
            SL.predict_law(lp.spec2, th[:lp.spec2.n_theta], lp.split(th)[1], cvg, F.R_GRID, epochs=epochs, nodes=M2.FULL_NODES)
        return np.clip(m, 1.0, None), time.time() - t0

    def concentration(m, epochs):
        return np.array([np.median(np.log10(m[:, j, i_c]) - np.log10(m[:, j, i_o])) for j in range(len(epochs))])

    m_ad, dt = predict(th_ad)
    r_ad = np.log10(m_ad) - lt_truth
    conc_ref = concentration(m_ad[:, list(RETUNE_EPOCHS)], RETUNE_EPOCHS)
    print(f"  {len(cvg)} galaxies of the fitting sample with a finite truth at every epoch and radius; one full-node prediction {dt:.0f} s; "
          f"{100 * dec.mean():.0f}% declined at 4.9 kpc from z=2 to 0.4; the adopted mean's median concentration log M(<5)/M(<103) at z=0.4 / 2: "
          + " / ".join(f"{v:+.3f}" for v in conc_ref))

    def retune_compact(th):
        """Damped Newton on (log_f_c, b_c) for the median concentration at the re-tune epochs (the two
        constants are nearly collinear in their effect, so the raw step overshoots: it is capped and
        backtracked on the miss). Returns (theta, iterations, final miss, predictions)."""
        th = th.copy()
        lo = np.array([bounds[ix["log_f_c"]][0], bounds[ix["b_c"]][0]])
        hi = np.array([bounds[ix["log_f_c"]][1], bounds[ix["b_c"]][1]])
        x = np.array([th[ix["log_f_c"]], th[ix["b_c"]]])
        n_pred = 0

        def miss_of(xv):
            nonlocal n_pred
            th[ix["log_f_c"]], th[ix["b_c"]] = xv
            m0, _ = predict(th, RETUNE_EPOCHS)
            n_pred += 1
            return concentration(m0, RETUNE_EPOCHS) - conc_ref

        f0 = miss_of(x)
        for it in range(RETUNE_MAX_ITER):
            if np.max(np.abs(f0)) < RETUNE_TOL:
                return th, it, float(np.max(np.abs(f0))), n_pred
            J = np.empty((2, 2))
            for jcol in range(2):
                xs = x.copy()
                xs[jcol] += RETUNE_STEP
                J[:, jcol] = (miss_of(xs) - f0) / RETUNE_STEP
            step = -np.linalg.lstsq(J, f0, rcond=None)[0]
            scale = float(np.max(np.abs(step) / RETUNE_STEP_CAP))
            if scale > 1.0:
                step = step / scale
            lam = 1.0
            for _ in range(6):
                x_new = np.clip(x + lam * step, lo, hi)
                f_new = miss_of(x_new)
                if np.linalg.norm(f_new) < np.linalg.norm(f0):
                    break
                lam *= 0.5
            x, f0 = x_new, f_new
        th[ix["log_f_c"]], th[ix["b_c"]] = x
        return th, RETUNE_MAX_ITER, float(np.max(np.abs(f0))), n_pred

    # ---- 1. the mechanism's own action at frozen constants ---------------- #
    print(f"\n{RULE}\n1. THE KNOB'S OWN ACTION at the adopted constants (no re-tune): the median shift of log M(<5) and log M(<103) per epoch\n{RULE}")
    for point in grid[:1] + [pt for pt in grid if list(pt)[0] != list(grid[0])[0]][:1]:
        th = th_ad.copy()
        for k_, v_ in point.items():
            th[ix[k_]] = float(v_)
        m, dt = predict(th)
        label = ", ".join(f"{k} = {v:+.2f}" for k, v in point.items())
        print(f"  [{label}] M(<5): " + " / ".join(f"{np.median(np.log10(m[:, k, i_c] / m_ad[:, k, i_c])):+.3f}" for k in EPOCHS)
              + "   M(<103): " + " / ".join(f"{np.median(np.log10(m[:, k, i_o] / m_ad[:, k, i_o])):+.3f}" for k in EPOCHS)
              + f"   ({dt:.0f} s)")
        print(f"    the decliners' vs the non-decliners' M(<5) shift at z=0.4: {np.median(np.log10(m[dec, 0, i_c] / m_ad[dec, 0, i_c])):+.3f} vs "
              f"{np.median(np.log10(m[~dec, 0, i_c] / m_ad[~dec, 0, i_c])):+.3f}; at z=2: {np.median(np.log10(m[dec, 4, i_c] / m_ad[dec, 4, i_c])):+.3f} vs "
              f"{np.median(np.log10(m[~dec, 4, i_c] / m_ad[~dec, 4, i_c])):+.3f}  (the differential the re-tune keeps)")

    # ---- 2. the probe --------------------------------------------------------- #
    print(f"\n{RULE}\n2. THE FROZEN-THETA PROBE — (log_f_c, b_c) re-tuned to the adopted median concentration at z=0.4 and z=2, a0 re-centred on M(<103) at z=0.4\n{RULE}")
    ref = gate_readings(r_ad, m_ad, truth, feats, dec, lmh_cat, growth, lmh_bin, i_c, i_o, i_2)
    ref["loss"] = lp.loss(th_ad)
    for k in range(1, 5):
        ref[f"leak_truth_{k}"] = SEL.partial_growth(lt_truth[:, k, i_o], lmh_cat[:, k], growth[:, k], np.isfinite(lmh_cat[:, k]))[1]
    print_gates("the adopted mean (q_c = q_ch = 0)", ref, ref)
    half_way = ref["split_model"] + 0.5 * (ref["split_truth"] - ref["split_model"])
    print(f"    GATES for a probe: G1 the split >= {half_way:+.3f} with both mismatches reduced (below {ref['dec_mismatch']:.3f} / {ref['non_mismatch']:.3f}); "
          f"G2 the z=0.4 concentration rho halved (<= {abs(ref['rho_conc_z04']) / 2:+.2f}); G3 the non-decliners' z=2 median within 0.02 of {ref['non_z2']:+.3f}")
    results = {"adopted": ref}
    for point in grid:
        label = ", ".join(f"{k} = {v:+.2f}" for k, v in point.items())
        th = th_ad.copy()
        for k_, v_ in point.items():
            th[ix[k_]] = float(v_)
        t0 = time.time()
        if retune:
            th, n_it, miss, n_pred = retune_compact(th)
            print(f"  [{label}] re-tuned in {n_it} Newton steps ({n_pred} two-epoch predictions, {time.time() - t0:.0f} s; final miss {miss:.4f} dex): "
                  f"log_f_c {th_ad[ix['log_f_c']]:+.3f} -> {th[ix['log_f_c']]:+.3f} (compact size x{10 ** (th[ix['log_f_c']] - th_ad[ix['log_f_c']]):.2f}), "
                  f"b_c {th_ad[ix['b_c']]:+.3f} -> {th[ix['b_c']]:+.3f}")
        m, dt = predict(th)
        shift = float(np.median(np.log10(m[:, 0, i_o]) - np.log10(m_ad[:, 0, i_o])))
        th[ix["a0"]] -= shift
        m = m * 10.0 ** (-shift)
        r = np.log10(m) - lt_truth
        g = gate_readings(r, m, truth, feats, dec, lmh_cat, growth, lmh_bin, i_c, i_o, i_2)
        g["loss"] = lp.loss(th)
        g["shift"] = shift
        g.update({f"theta_{n}": float(th[ix[n]]) for n in ix})
        moved = float(np.max(np.abs(np.log10(m / m_ad))))
        print_gates(f"{label} (a0 re-centred by {-shift:+.3f}; the profile moved by up to {moved:.3f} dex)", g, ref)
        g1, g2, g3 = gate_verdict(g, ref)
        print(f"    GATE {'PASS' if (g1 and g2 and g3) else 'fail'}: G1 split half way with both mismatches reduced {g1}, "
              f"G2 concentration rho halved {g2}, G3 non-decliners' z=2 within 0.02 {g3}")
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
    main(smoke="--smoke" in a, grid=grid, retune="--no-retune" not in a)
