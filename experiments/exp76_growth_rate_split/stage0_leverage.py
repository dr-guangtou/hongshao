"""exp76 Stage 0 — LEVERAGE BY EVALUATION, no fit: does letting the deposit's
shape depend on how fast the halo was growing fix the assembly correlations
under the honest (measured) input, and what does it do to the sizes, the
centre and the loss?

THE LEVER. exp63's model decides whether a deposit is compact or extended
from the halo mass at that moment, w_c = expit((m_half - log M)/d). The
optional lever `g_split` adds g (alpha - 1) inside the logit, alpha = dlnM/dlnt
at the deposit: g < 0 makes mass arriving during fast growth (mergers) land
extended and mass arriving during slow growth land compact. exp63's P9 found
by evaluation that g ~ -0.5 to -1 reproduces the SIGN and size of every
assembly correlation the data show and the g = 0 model gets backwards -- but
alpha was then DiffMAH's slope, a smooth function of the final mass (C19), and
the assembly variables were DiffMAH-derived. Under the measured input alpha at
every node is the history's own local slope, and the assembly variables can
be measured from the same history.

WHAT IS MEASURED, all at exp74's measured-input optimum (the adopted
baseline's predecessor) with everything but g held fixed -- a conditional
slice, used to show the shape of the trade along one axis, never to locate an
optimum (exp54 Stage 3.9):

  1. P9's table, re-measured: partial Spearman correlations at fixed z=0.4
     halo mass (measured) of (a) the DATA's inner share (exp63 Stage 1's
     deconvolution: the share of the z=0.4 deposit-size distribution inside
     5 kpc) and (b) the MODEL's compact share (the mass-weighted share of its
     deposits that went into the compact channel) with assembly variables --
     MEASURED ones from the running-peak history (t50, t80 = when the halo
     reached half / 80 per cent of its z=0.4 mass; late growth = dlog M over
     the last 2.1 and 3.5 Gyr) and the DiffMAH ones P9 used (`late`, `f_form`,
     `logtc`, `t50`) for continuity;
  2. the g sweep: for g in G_SWEEP, the model's correlations, the loss under
     the ADOPTED references (nested incumbent on measured curves, measured-mass
     bins), the R50 width ratio at fixed stellar mass per epoch (the size
     gate's width sub-gate), the centre (median residual at 2 kpc) per epoch,
     and the z=2 mh-complete total.

GATE A: a g exists at which every measured-variable correlation of the model's
share has the data's sign. If none does, exp76 stops here.

Run:
    HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 \\
    PYTHONPATH=. uv run python -u experiments/exp76_growth_rate_split/stage0_leverage.py [--smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term",
          ROOT / "experiments/exp63_analytic_growth",
          ROOT / "experiments/exp73_size_relative",
          ROOT / "experiments/exp74_c19_history_leak", HERE):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import engine as E                                       # noqa: E402
import model2 as M2                                      # noqa: E402
import selection as SEL                                  # noqa: E402
import stage0_cost as S0                                 # noqa: E402
import stage2_fit as S2F                                 # noqa: E402
import coordinate as C                                   # noqa: E402
import history as HI                                     # noqa: E402
import measured as MB                                    # noqa: E402
from hongshao import qa                                  # noqa: E402

RULE, THIN = "=" * 100, "-" * 100
ANCHOR_Z = list(E.ANCHOR_Z)
EPOCHS = (0, 1, 2, 3, 4)
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
E74 = ROOT / "experiments/exp74_c19_history_leak/outputs"
E63 = ROOT / "experiments/exp63_analytic_growth/outputs"
FIT_NPZ = E63 / "stage2_fit_joint_kpc_free_sane.npz"
POP = ROOT / "experiments/exp32_full_population/outputs/population.npz"
G_SWEEP = (0.0, -0.25, -0.5, -0.75, -1.0, -1.5, -2.0)
MEASURED_VARS = ("t50_meas", "t80_meas", "growth_2gyr", "growth_3p5gyr")
DIFFMAH_VARS = ("late", "f_form", "logtc", "t50_dmah")


def adopted_problem(spec, recs, data, mask, lmh_dm, meas, th_inc):
    """exp63's joint problem on the measured curves with the ADOPTED references
    (nested incumbent on measured curves) and measured-mass bins."""
    hs = np.load(SEL.HS_NPZ, allow_pickle=True)
    rows = np.array([h.row for h in recs])
    lmh_cat = SEL.sample_masses(hs)[rows]
    lmh_bins = np.where(np.isfinite(lmh_cat), lmh_cat, lmh_dm)
    return S2F.JointProblem2(spec, meas, data, mask, lmh_bins, F.R_GRID, th_inc, binned=True), lmh_cat


def measured_assembly(recs, hs):
    """Assembly variables from the running-peak history: t50, t80 [Gyr] (when
    the peak first reached 50 / 80 per cent of its z=0.4 value, interpolated
    in log t), and the late growth dlog M over the last 2.1 Gyr (z=0.7 -> 0.4)
    and 3.5 Gyr (z=1.0 -> 0.4)."""
    rows = np.array([h.row for h in recs])
    snaps = np.asarray(hs["snaps"], int)
    t = np.array([E.T_SNAP[s] for s in snaps]); lt = np.log10(t)
    peak = HI.peak_history(hs)[rows]
    i0 = int(np.where(snaps == E.ANCHOR_SNAP[0])[0][0])
    i07 = int(np.where(snaps == E.ANCHOR_SNAP[1])[0][0])
    i10 = int(np.where(snaps == E.ANCHOR_SNAP[2])[0][0])
    n = len(recs)
    out = {k: np.full(n, np.nan) for k in MEASURED_VARS}
    for g in range(n):
        y = peak[g]; ok = np.isfinite(y) & (t <= t[i0])
        if ok.sum() < 3 or not np.isfinite(y[i0]):
            continue
        yy, ll = y[ok], lt[ok]
        for key, frac in (("t50_meas", 0.5), ("t80_meas", 0.8)):
            target = y[i0] + np.log10(frac)
            if yy[0] >= target:
                out[key][g] = 10 ** ll[0]
            else:
                out[key][g] = 10 ** np.interp(target, yy, ll)
        if np.isfinite(y[i07]):
            out["growth_2gyr"][g] = y[i0] - y[i07]
        if np.isfinite(y[i10]):
            out["growth_3p5gyr"][g] = y[i0] - y[i10]
    return out


def partial_spearman(x, y, ctrl, mask):
    ok = mask & np.isfinite(x) & np.isfinite(y) & np.isfinite(ctrl)
    if ok.sum() < 50:
        return np.nan
    D = np.column_stack([np.ones(int(ok.sum())), ctrl[ok]])
    rx = x[ok] - D @ np.linalg.lstsq(D, x[ok], rcond=None)[0]
    ry = y[ok] - D @ np.linalg.lstsq(D, y[ok], rcond=None)[0]
    return float(spearmanr(rx, ry).correlation)


def compact_share(spec, theta, curves, include):
    """(n, 5) mass-weighted share of the deposits that went compact, per epoch."""
    lt, w, inc, _ = E.nodes(**M2.FULL_NODES)
    n = len(curves)
    out = np.full((n, 5), np.nan)
    for lo in range(0, n, 300):
        cv = curves[lo:lo + 300]
        dm_c, dm_e, *_ = M2._deposits2(spec, theta, cv, lt)
        for k in EPOCHS:
            wk = w * inc[k]
            tot = ((dm_c + dm_e) * wk[None, :]).sum(1)
            out[lo:lo + len(cv), k] = (dm_c * wk[None, :]).sum(1) / np.where(tot > 0, tot, np.nan)
    return out


def width_ratio_r50(pred, data, R, fit_all):
    """The size gate's width sub-gate for R50 at fixed STELLAR mass, per epoch,
    through qa.evaluate (figures off)."""
    logms = np.log10(np.clip(data[fit_all][:, 0, -1], 1.0, None))
    out = qa.evaluate(pred[fit_all], data[fit_all], R, ANCHOR_Z, figdir=None, figures=False, verbose=False,
                      bin_by=logms)
    g = out["size_gate_ms"][0]
    return np.array([g[("R50", j)]["width_ratio"] for j in range(5)]), \
        np.array([g[("R50", j)]["offset"] for j in range(5)])


def main(smoke=False):
    tag = "_smoke" if smoke else ""
    print(f"{RULE}\nexp76 Stage 0 — leverage of the growth-rate split under the measured input, by evaluation\n{RULE}\n")
    recs, data, mask_legacy, lmh_dm, spec_inc, th_inc, _ = S0.build(smoke)
    mask, _ = S2F.fit_masks(data, mask_legacy, lmh_dm, "sane")
    fz = np.load(FIT_NPZ, allow_pickle=True)
    spec2 = S2F.spec_from_fit(fz)
    # the growth split is a spec OPTION (theta gains `g_split`), not a lever
    spec_g = M2.Spec2(theta_names=M2.THETA_NAMES_GROWTH, extended_family=spec2.extended_family,
                      compact_in_kpc=spec2.compact_in_kpc)
    assert spec_g.growth_split and spec_g.n_theta == spec2.n_theta + 1
    hist_path = E74 / f"history_curves{tag}.npz"
    meas = MB.build_input(recs, "measured", hist_path=hist_path, verbose=True)[0]
    f0 = E74 / f"stage1_refit_measured{tag}.npz"
    if not f0.exists():                          # the smoke has no fit of its own; use the full one
        f0 = E74 / "stage1_refit_measured.npz"
    th0 = np.asarray(np.load(f0, allow_pickle=True)["theta_best"], float)
    thg0 = np.r_[th0, 0.0]
    jg = spec_g.index("g_split")
    print(f"  theta_0 = exp74's measured-input refit; lever `g_split` appended at 0 (nests bit for bit)")
    pr, lmh_cat = adopted_problem(spec_g, recs, data, mask, lmh_dm, meas, th_inc)
    th_nested = M2.nested_theta(th_inc, growth=True)
    l_null = pr.loss(th_nested)
    print(f"  adopted references: nested incumbent on measured curves, loss {l_null:.4f}")

    hs = np.load(SEL.HS_NPZ, allow_pickle=True)
    rows = np.array([h.row for h in recs])
    good = np.isfinite(data).all(axis=(1, 2)) & (data > 0).all(axis=(1, 2))
    fit_all = good & mask.all(1)
    ctrl = np.where(np.isfinite(lmh_cat[:, 0]), lmh_cat[:, 0], lmh_dm[:, 0])

    # ---- the assembly variables ------------------------------------------- #
    va = measured_assembly(recs, hs)
    pop = np.load(POP, allow_pickle=True)
    curves = E.build_curves(recs, verbose=False)
    va["late"] = np.array([hc.late for hc in curves])
    va["logtc"] = np.array([hc.logtc for hc in curves])
    va["t50_dmah"] = np.asarray(pop["t50"], float)[rows]
    va["f_form"] = np.array([h.f_form[h.epoch_mask[0]][-1] for h in recs])
    # the data's inner share, exp63 Stage 1 (aligned by record index)
    s1 = np.load(E63 / "stage1_deconvolve.npz", allow_pickle=True)
    pos = {int(i): j for j, i in enumerate(s1["index"])}
    w = np.asarray(s1["w_gompertz_c0.80"], float)
    s_grid = np.asarray(s1["s_grid"], float)
    share_data = np.full(len(recs), np.nan)
    for g, h in enumerate(recs):
        j = pos.get(int(h.index))
        if j is not None:
            share_data[g] = w[j, s_grid < 5].sum() / max(w[j].sum(), 1e-30)
    print(f"  data inner share (exp63 Stage 1, gompertz c0.80, s < 5 kpc): median "
          f"{np.nanmedian(share_data[fit_all]):.2f}, finite for {int(np.isfinite(share_data[fit_all]).sum())}")

    all_vars = MEASURED_VARS + DIFFMAH_VARS
    def corr_row(share):
        return np.array([partial_spearman(share, va[v], ctrl, fit_all) for v in all_vars])

    # ---- 1. P9's table ------------------------------------------------------ #
    print(f"\n{RULE}\n1. P9 RE-MEASURED — partial Spearman rho at fixed z=0.4 halo mass (measured) of the compact "
          f"share\n   with assembly variables; the data's inner share first. MEASURED variables, then DiffMAH's.\n{RULE}")
    hdr = f"  {'share':<22}" + "".join(f"{v:>13}" for v in all_vars)
    print(hdr)
    r_data = corr_row(share_data)
    print(f"  {'DATA inner share':<22}" + "".join(f"{v:>13.3f}" for v in r_data))
    sweep = {}
    for gval in G_SWEEP:
        th = thg0.copy(); th[jg] = gval
        sh = compact_share(spec_g, th, meas, None)
        r = corr_row(sh[:, 0])
        sweep[gval] = dict(share=sh, corr=r, median_share=float(np.nanmedian(sh[fit_all, 0])))
        print(f"  {f'model g={gval:+.2f} (share {sweep[gval]['median_share']:.2f})':<22}" + "".join(f"{v:>13.3f}" for v in r))
    sign_ok = {gval: bool(np.all(np.sign(sweep[gval]["corr"][:4]) == np.sign(r_data[:4]))) for gval in G_SWEEP}
    print(f"\n  GATE A (all four MEASURED-variable signs match the data's): "
          + ", ".join(f"g={gv:+.2f} {'yes' if ok else 'no'}" for gv, ok in sign_ok.items()))

    # ---- 2. the sweep's other faces ---------------------------------------- #
    print(f"\n{RULE}\n2. THE SWEEP — loss under the adopted references (null {l_null:.3f}), the R50 width ratio at "
          f"fixed M* per epoch,\n   the centre (median residual at 2 kpc) per epoch, and the z=2 mh-complete "
          f"M(<103 kpc); everything but g fixed\n{RULE}")
    sn = np.load(ROOT / "experiments/exp54_unpinned_amplitude/outputs/selection.npz", allow_pickle=True)
    complete = np.isfinite(lmh_cat) & (lmh_cat >= sn["cuts"][None, :])
    i2 = int(np.argmin(np.abs(F.R_GRID - 2.0))); i103 = int(np.argmin(np.abs(F.R_GRID - 103.45)))
    print(f"  {'g':>6}{'loss':>9}" + "".join(f"{f'w{z}':>7}" for z in ANCHOR_Z)
          + "".join(f"{f'c{z}':>7}" for z in ANCHOR_Z) + f"{'z2 mh-c':>9}")
    for gval in G_SWEEP:
        th = thg0.copy(); th[jg] = gval
        loss = pr.loss(th)
        pred = M2.predict2(spec_g, th, meas, F.R_GRID)
        ok = fit_all & np.isfinite(pred).all(axis=(1, 2)) & (pred > 0).all(axis=(1, 2))
        wr, off = width_ratio_r50(pred, data, F.R_GRID, ok)
        cen = [100 * np.nanmedian((pred[ok, k, i2] - data[ok, k, i2]) / data[ok, k, i2]) for k in EPOCHS]
        mc = ok & complete[:, 4]
        z2 = 100 * np.nanmedian((pred[mc, 4, i103] - data[mc, 4, i103]) / data[mc, 4, i103])
        sweep[gval].update(loss=loss, width=wr, offset=off, centre=np.array(cen), z2_mhc=z2)
        print(f"  {gval:>+6.2f}{loss:>9.3f}" + "".join(f"{v:>7.2f}" for v in wr)
              + "".join(f"{v:>+7.1f}" for v in cen) + f"{z2:>+8.1f}%")

    # ---- figure -------------------------------------------------------------- #
    FIGDIR.mkdir(parents=True, exist_ok=True)
    gs = np.array(G_SWEEP)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    for i, v in enumerate(MEASURED_VARS):
        ax.plot(gs, [sweep[g]["corr"][i] for g in G_SWEEP], "-o", ms=4, label=qa._tex(v.replace("_", " ")))
        ax.axhline(r_data[i], ls=":", c=ax.lines[-1].get_color(), lw=1)
    ax.axhline(0, c="0.6", lw=0.8); ax.set_xlabel("g (growth-rate split)"); ax.set_ylabel(qa._tex("partial Spearman rho at fixed M$_h$"))
    ax.set_title("model compact share vs MEASURED assembly (dotted: the data's inner share)", fontsize=9); ax.legend(fontsize=7)
    ax = axes[1]
    for k in EPOCHS:
        ax.plot(gs, [sweep[g]["width"][k] for g in G_SWEEP], "-o", ms=4, label=f"z={ANCHOR_Z[k]}")
    ax.axhline(1, c="0.6", lw=0.8); ax.axhspan(0.8, 1.2, color="0.9", zorder=0)
    ax.set_xlabel("g"); ax.set_ylabel("R50 width ratio at fixed M* (model / truth)"); ax.set_title("the size gate's width sub-gate", fontsize=9); ax.legend(fontsize=7)
    ax = axes[2]
    ax.plot(gs, [sweep[g]["loss"] for g in G_SWEEP], "-o", ms=4, c="k", label="loss (adopted refs)")
    ax.set_xlabel("g"); ax.set_ylabel("loss"); ax.set_title(qa._tex(f"loss (null {l_null:.2f}); dashed: z=2 centre [{qa._pct()}] on the right axis"), fontsize=9)
    ax2 = ax.twinx(); ax2.plot(gs, [sweep[g]["centre"][4] for g in G_SWEEP], "--s", ms=4, c="C3"); ax2.set_ylabel(qa._tex(f"z=2 residual at 2 kpc [{qa._pct()}]"), color="C3")
    fig.suptitle("exp76 Stage 0 — the growth-rate split by evaluation (everything but g fixed at exp74's measured optimum)", fontsize=10)
    fig.tight_layout()
    print("wrote", qa.save_fig(fig, FIGDIR / f"exp76_stage0_sweep{tag}")[0])

    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUTDIR / f"stage0_leverage{tag}.npz", g_sweep=gs, vars=np.array(all_vars), corr_data=r_data,
             corr_model=np.array([sweep[g]["corr"] for g in G_SWEEP]),
             loss=np.array([sweep[g]["loss"] for g in G_SWEEP]), loss_null=l_null,
             width=np.array([sweep[g]["width"] for g in G_SWEEP]),
             offset=np.array([sweep[g]["offset"] for g in G_SWEEP]),
             centre=np.array([sweep[g]["centre"] for g in G_SWEEP]),
             z2_mhc=np.array([sweep[g]["z2_mhc"] for g in G_SWEEP]),
             median_share=np.array([sweep[g]["median_share"] for g in G_SWEEP]),
             share_data=share_data, gate_a=np.array([sign_ok[g] for g in G_SWEEP]),
             **{f"var_{k}": v for k, v in va.items()})
    print(f"wrote {OUTDIR / f'stage0_leverage{tag}.npz'}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
