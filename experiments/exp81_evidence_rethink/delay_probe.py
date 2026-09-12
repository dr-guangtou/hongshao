"""exp81 measurement 3 — does a DEPOSITION DELAY remove the baseline's
recent-growth residual? (measurement 2: the residual correlates +0.3 to +0.6
at fixed halo mass with the mass the halo gained in its last Gyr, in the
outskirts at every epoch and everywhere at z >= 1.5; the truth's own
dependence is the opposite sign.)

exp63's Stage 2b lever, `model2.Spec2.with_delay()`: a satellite's stars
join the central tau_d Hubble times after the halo accreted it; mass accreted
but not yet arrived is in transit. Applied to the baseline at FROZEN theta
for tau_d in {0, 0.15, 0.3, 0.5, 0.8}. Reported per tau_d: the partial
correlations of the residual with recent growth / early mass / t50 at the
cells of measurement 2 (amplitude-free); the median profile per epoch on
the fitting sample and the mh-complete subset; the R50 offsets; the loss
terms (A, F, S, B) at frozen theta. The amplitude will drop with tau_d (mass
in transit is not deposited yet); the amplitude-pinned shape residual and
the correlations are what to read, the amplitude is the fit's to re-absorb.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term",
          ROOT / "experiments/exp63_analytic_growth",
          ROOT / "experiments/exp73_size_relative",
          ROOT / "experiments/exp74_c19_history_leak",
          ROOT / "experiments/exp78_size_aware_objective", HERE):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import engine as E                                       # noqa: E402
import model2 as M2                                      # noqa: E402
import coordinate as C                                   # noqa: E402
import selection as SEL                                  # noqa: E402


def _by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S0T = _by_path("exp78_stage0_terms", ROOT / "experiments/exp78_size_aware_objective/stage0_terms.py")
RB = S0T.RB
ANCHOR_Z = list(E.ANCHOR_Z)
EPOCHS = (0, 1, 2, 3, 4)
CELLS = [(0, 4.92), (0, 103.45), (2, 103.45), (3, 32.58), (4, 4.92), (4, 33), (4, 103.45)]
TAUS = (0.0, 0.15, 0.3, 0.5, 0.8)
R_SHOW = (2.0, 10.25, 52.30, 103.45)
SEL_NPZ = ROOT / "experiments/exp54_unpinned_amplitude/outputs/selection.npz"
OUTDIR = HERE / "outputs"


def partial(y, x, lmh):
    ok = np.isfinite(y) & np.isfinite(x) & np.isfinite(lmh)
    X = np.column_stack([np.ones(ok.sum()), lmh[ok], lmh[ok] ** 2])
    ry = y[ok] - X @ np.linalg.lstsq(X, y[ok], rcond=None)[0]
    rx = x[ok] - X @ np.linalg.lstsq(X, x[ok], rcond=None)[0]
    return float(spearmanr(ry, rx)[0])


def main():
    recs, data, mask, lmh_dm, lmh_bins, meas, fz, spec2, th_inc, pr, Rm, truth_m, n_inner, rows_m = S0T.build(False)
    spec_b, th_b = RB.adopted_baseline()
    spec_d = M2.Spec2(theta_names=M2.THETA_NAMES_DELAY, extended_family=spec2.extended_family, compact_in_kpc=spec2.compact_in_kpc)
    rows = pr.all_rows
    cv = [meas[i] for i in rows]
    truth = data[rows]
    good = np.isfinite(truth).all(axis=(1, 2)) & (truth > 0).all(axis=(1, 2))
    t = truth[good]; lt_t = np.log10(t)
    cvg = [c for c, g in zip(cv, good) if g]
    hs = np.load(SEL.HS_NPZ, allow_pickle=True)
    sn = np.load(SEL_NPZ, allow_pickle=True)
    lmh_cat = SEL.sample_masses(hs)[np.array([h.row for h in recs])][rows][good]
    complete = np.isfinite(lmh_cat) & (lmh_cat >= sn["cuts"][None, :])
    feats = {}
    for k in EPOCHS:
        lt_k = np.log10(E.T_ANCHOR[k]); t_k = E.T_ANCHOR[k]
        lm_k = np.array([E.log_mah(np.array([lt_k]), hc)[0] for hc in cvg])
        lm_1 = np.array([E.log_mah(np.array([np.log10(2.0)]), hc)[0] for hc in cvg])
        lm_prev = np.array([E.log_mah(np.array([np.log10(t_k - 1.0)]), hc)[0] for hc in cvg])
        feats[k] = dict(lmh=lm_k, early=lm_1 - lm_k, recent=lm_k - lm_prev)
    ir = [int(np.argmin(np.abs(F.R_GRID - r))) for r in R_SHOW]
    print(f"  {int(good.sum())} galaxies; the baseline with a deposition delay tau_d (Hubble times at accretion) at FROZEN theta\n")
    print(f"  A. partial Spearman at fixed log Mh(z_k) of the residual with RECENT growth (last Gyr) | EARLY mass (2 Gyr): the truth's own in the first row")
    print(f"  {'tau_d':<8}" + "".join(f"{f'z={ANCHOR_Z[k]} R={R:.0f}':>18}" for k, R in CELLS))
    print(f"  {'truth':<8}" + "".join(f"{partial(lt_t[:, k, int(np.argmin(np.abs(F.R_GRID - R)))], feats[k]['recent'], feats[k]['lmh']):+.2f} | {partial(lt_t[:, k, int(np.argmin(np.abs(F.R_GRID - R)))], feats[k]['early'], feats[k]['lmh']):+.2f}".rjust(18) for k, R in CELLS))
    res = {}
    for tau in TAUS:
        th = np.r_[th_b, tau]
        m = M2.predict2(spec_d, th, cvg, F.R_GRID, epochs=EPOCHS)
        r = np.log10(np.clip(m, 1, None)) - lt_t
        pin = np.log10(np.clip(m / m[:, :, F.I100][:, :, None] * t[:, :, F.I100][:, :, None], 1, None)) - lt_t
        cells = []
        for k, R in CELLS:
            i = int(np.argmin(np.abs(F.R_GRID - R)))
            cells.append(f"{partial(r[:, k, i], feats[k]['recent'], feats[k]['lmh']):+.2f} | {partial(r[:, k, i], feats[k]['early'], feats[k]['lmh']):+.2f}")
        print(f"  {tau:<8.2f}" + "".join(c.rjust(18) for c in cells))
        res[tau] = (m, r, pin)
    print(f"\n  B. median residual [%] of M(<R), fitting sample | mh-complete, per tau_d and epoch (raw; the amplitude drops with tau_d)")
    print(f"  {'tau_d':<6}{'z':>5}" + "".join(f"{f'M(<{F.R_GRID[i]:.0f})':>22}" for i in ir))
    for tau in TAUS:
        m = res[tau][0]
        for k in EPOCHS:
            mc = complete[:, k]
            print(f"  {tau if k == 0 else '':<6}{ANCHOR_Z[k]:>5}" + "".join(
                f"{100 * np.median((m[:, k, i] - t[:, k, i]) / t[:, k, i]):>+10.1f} |{100 * np.median((m[mc, k, i] - t[mc, k, i]) / t[mc, k, i]):>+9.1f}" for i in ir))
    print(f"\n  C. the AMPLITUDE-PINNED shape residual (model rescaled to the truth's M(<100)): median [dex] at 2 / 10 / 52 kpc and the 50-100 kpc shell [%], per epoch")
    for tau in TAUS:
        m, r, pin = res[tau]
        sh = lambda c: c[:, :, F.I100] - c[:, :, ir[2]]
        mp = m / m[:, :, F.I100][:, :, None] * t[:, :, F.I100][:, :, None]
        print(f"  tau {tau:<5.2f}" + "".join(
            f"  z={ANCHOR_Z[k]}: {np.median(pin[:, k, ir[0]]):+.3f}/{np.median(pin[:, k, ir[1]]):+.3f}/{np.median(pin[:, k, ir[2]]):+.3f} shell {100 * np.median((sh(mp)[:, k] - sh(t)[:, k]) / sh(t)[:, k]):+.0f}%" for k in EPOCHS))
    print(f"\n  D. per-galaxy rms of the pinned shape residual at 10 kpc and 52 kpc, per epoch (does the delay shrink the SCATTER?)")
    for tau in TAUS:
        pin = res[tau][2]
        print(f"  tau {tau:<5.2f}" + "".join(f"  z={ANCHOR_Z[k]}: {np.sqrt(np.mean(pin[:, k, ir[1]] ** 2)):.4f}/{np.sqrt(np.mean(pin[:, k, ir[2]] ** 2)):.4f}" for k in EPOCHS))
    print(f"\n  E. R50 median log(model/truth) per epoch, fitting sample")
    for tau in TAUS:
        m = res[tau][0]
        r50m, r50t = C.size_radius(m, F.R_GRID, 0.5), C.size_radius(t, F.R_GRID, 0.5)
        print(f"  tau {tau:<5.2f}" + "".join(f"  z={ANCHOR_Z[k]}: {np.nanmedian(np.log10(r50m[:, k] / r50t[:, k])):+.3f}" for k in EPOCHS))
    print(f"\n  F. the loss terms A/F/S/B per epoch at frozen theta (the amplitude term A rises with tau_d until refitted)")
    for tau in TAUS:
        m = res[tau][0]
        # score on the loss's own rows: pr.problems[k].score_model wants the fit rows' order; use all good rows via index
        print(f"  tau {tau:<5.2f}" + "".join(
            f"  z={ANCHOR_Z[k]}: " + "/".join(f"{v:.2f}" for v in np.array(pr.problems[k].score_model(m[:, k]))[[0, 1, 2, 4]]) for k in EPOCHS))
    np.savez(OUTDIR / "delay_probe.npz", taus=np.array(TAUS), **{f"r_{tau}": res[tau][1] for tau in TAUS}, **{f"pin_{tau}": res[tau][2] for tau in TAUS})


if __name__ == "__main__":
    main()
