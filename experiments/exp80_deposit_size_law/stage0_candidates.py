"""exp80 Stage 0, part C — NO FIT of the model. The candidate size-law changes
applied to the adopted baseline at FROZEN amplitude, split and kernels, and
scored on the gates (the frozen-theta test, exp74's lesson).

Everything but the size law's own constants is the baseline's: the efficiency
law (a0, a_M, a_z, a_Mz), the compact/extended split (m_half, d_split), the
two kernel shapes (n_c, c_e), the compact channel's law, the truncation C.
For each candidate the law's constants are set BY HAND from the Stage 0 A/B
table (the `start` values) and then TUNED on the radius term alone (the rms
over R20/R50/R80 x three halo-mass terciles of the tercile-median log
R_f(model)/R_f(truth), on the merged grid, summed in quadrature over the five
epochs; `size_terms.radius_term`), with everything else frozen. The CONTROL
is the baseline law itself re-tuned the same way (`control`: log_f_e and b_e
free, no knob), so a candidate is credited only with what its CHANGE buys
beyond re-tuning the constants the model already has.

Gate to proceed to Stage 1 (the plan): the candidate must recover more than
0.02 dex of the z = 2 R50 offset (the size gate's offset sub-gate at fixed
stellar mass on the standard grid, fitting sample; baseline +0.054) beyond
the control, with the z <= 1 R50 offsets kept within the 0.05 dex tolerance.
Also reported: R20/R80, the widths, the z = 2 mass-size slope (truth 0.11),
the profile at 2/10/52/103 kpc on both samples, the loss terms A, F, S, B at
the frozen theta (what the standard objective will push back against), and
the future-dependence gate at 103 kpc.

Run (one process per candidate, then --merge):
    HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 PYTHONPATH=. \\
        nohup uv run python -u experiments/exp80_deposit_size_law/stage0_candidates.py --candidate NAME [--smoke] \\
        > experiments/exp80_deposit_size_law/outputs/stage0_cand_NAME.log 2>&1 &
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

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
import selection as SEL                                  # noqa: E402
import coordinate as C                                   # noqa: E402
import size_terms as ST                                  # noqa: E402
import size_law as SL                                    # noqa: E402
from hongshao import qa                                  # noqa: E402
from hongshao.fitting import minimize_loss               # noqa: E402


def _by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S0T = _by_path("exp78_stage0_terms", ROOT / "experiments/exp78_size_aware_objective/stage0_terms.py")
RB = S0T.RB

RULE = "=" * 110
ANCHOR_Z = list(E.ANCHOR_Z)
EPOCHS = (0, 1, 2, 3, 4)
OUTDIR = HERE / "outputs"
SEL_NPZ = ROOT / "experiments/exp54_unpinned_amplitude/outputs/selection.npz"
R_SHOW = (2.0, 10.25, 52.30, 103.45)
R_LEAK = 103.45
MAX_TUNE_EVALS = {"smoke": 30, "full": 500}
GATE_RECOVER_DEX = 0.02
GATE_LOW_Z_TOL = 0.05
BOUNDS = {"log_f_e": (-2.5, 0.0), "b_e": (-3.0, 3.0), "g_e": (-1.5, 1.5), "q_e": (0.0, 1.5),
          "early_kpc": (-0.5, 1.5), "early_b": (-3.0, 1.0), "z_sw": (0.3, 4.0),
          "b_e2": (-3.0, 3.0), "z_brk_e": (0.5, 5.0), "s_floor_kpc": (0.0, 20.0)}
#: the candidates: the knobs switched on (hand-set from the table), the
#: constants the tune may move, and the extra parameters each would add to
#: the twelve if fitted in Stage 1
CANDIDATES = {
    "control": dict(law={}, free=["log_f_e", "b_e"], extra=0,
                    what="the baseline law re-tuned (log_f_e, b_e): the control"),
    "b_fixed_kpc": dict(law=dict(g_e=-1.0 / 3.0), free=["log_f_e", "b_e", "g_e"], extra=1,
                        start={"log_f_e": -1.05},
                        what="(b) g_e = -1/3: every extended deposit a fixed physical size at fixed z' (pivot moved to 10^12 by hand)"),
    "a_expand": dict(law=dict(q_e=1.0), free=["log_f_e", "b_e"], extra=0, start={"log_f_e": -1.1},
                     what="(a) q_e = 1: the extended deposit expands with the halo, sized by R200c at the OBSERVED epoch"),
    "a_expand_free": dict(law=dict(q_e=0.5), free=["log_f_e", "b_e", "q_e"], extra=1, start={"log_f_e": -0.9},
                          what="(a) with the expansion exponent q_e free"),
    "a_expand_free_g": dict(law=dict(q_e=0.13, g_e=-0.1), free=["log_f_e", "b_e", "q_e", "g_e"], extra=2,
                            start={"log_f_e": -0.58, "b_e": -1.43},
                            what="(a)+(b) both free: the expansion exponent and a halo-mass exponent at fixed z'"),
    "a_expand_qonly": dict(law=dict(q_e=0.13), free=["q_e"], extra=1,
                           what="(a) q_e alone, the baseline's own constants kept: the knob's action isolated from the re-tune"),
    "d_early_kpc": dict(law=dict(early_kpc=1.0, early_b=-0.9, z_sw=1.0), free=["early_kpc", "early_b", "z_sw", "log_f_e", "b_e"],
                        extra=3, what="(d) the extended channel's EARLY deposits (z' > z_sw) at a fixed physical size 10^early_kpc (1+z')^early_b kpc; late ones R200c(t')-scaled"),
    "c_break": dict(law=dict(b_e2=0.5, z_brk_e=2.0), free=["log_f_e", "b_e", "b_e2", "z_brk_e"], extra=2,
                    what="(c) a second (1+z') exponent for the extended channel above a break redshift"),
    "ab_expand_fixed": dict(law=dict(q_e=1.0, g_e=-1.0 / 3.0), free=["log_f_e", "b_e", "g_e"], extra=1, start={"log_f_e": -1.4},
                            what="(a)+(b): expands with the halo AND mass-independent at fixed z'"),
}


class FrozenLaw:
    """The baseline with only the size law's constants movable."""

    def __init__(self, spec2, th_b, cand, pr, Rm, truth_m, n_inner, lmh_bins, rows_m, index_of):
        self.spec2, self.th_b, self.cand = spec2, th_b, cand
        self.pr, self.Rm, self.n_inner = pr, np.asarray(Rm, float), int(n_inner)
        self.curves = pr.curves
        self.rows_m, self.index_of = rows_m, index_of
        self.truth_k = {k: truth_m[rows_m[k], k] for k in EPOCHS}
        self.lmh_k = {k: lmh_bins[rows_m[k], k] for k in EPOCHS}
        self.free = list(cand["free"])
        self.bounds = [BOUNDS[n] for n in self.free]
        law0 = SL.with_law(**cand["law"])
        p = spec2.unpack(th_b)
        x0 = []
        for n in self.free:
            v = cand.get("start", {}).get(n, p[n] if n in p else law0[n])
            x0.append(float(v))
        self.x0 = np.array(x0)
        self.n_eval = 0

    def unpack(self, x):
        th = self.th_b.copy()
        law = SL.with_law(**self.cand["law"])
        for n, v in zip(self.free, x):
            if n in self.spec2.theta_names:
                th[self.spec2.index(n)] = v
            else:
                law[n] = float(v)
        return th, law

    def predict(self, x, nodes=M2.FULL_NODES):
        th, law = self.unpack(x)
        return SL.predict_law(self.spec2, th, law, self.curves, self.Rm, epochs=EPOCHS, nodes=nodes)

    def radius_terms(self, m):
        """(5,) raw radius term per epoch and the (5, 3, 3) median tables."""
        raw, med = np.full(5, np.nan), np.full((5, 3, 3), np.nan)
        for j, k in enumerate(EPOCHS):
            raw[j], med[j], _ = ST.radius_term(m[self.index_of[self.rows_m[k]], j], self.truth_k[k], self.Rm, self.lmh_k[k])
        return raw, med

    def loss(self, x):
        self.n_eval += 1
        m = self.predict(x, nodes=M2.FIT_NODES)
        if not np.isfinite(m).all():
            return F.FAIL
        raw, _ = self.radius_terms(m)
        if not np.isfinite(raw).all():
            return F.FAIL
        return float((raw ** 2).sum())

    def describe(self, x):
        th, law = self.unpack(x)
        p = self.spec2.unpack(th)
        return f"log_f_e {p['log_f_e']:+.3f}, b_e {p['b_e']:+.3f}; law: {SL.describe(law)}"


def score(name, m_full, pr, n_inner, data, fit_all, complete, lmh_bins, lmh_cat, index_of, fl, label):
    """Everything the verdict needs for one model prediction on the merged grid (n_fit_rows, 5, nRm)."""
    raw, med = fl.radius_terms(m_full)
    # the loss terms at the frozen theta
    per = {k: pr.problems[k].score_model(m_full[pr.index[k], j, n_inner:]) for j, k in enumerate(EPOCHS)}
    lt = np.array([[per[k][0], per[k][1], per[k][2], per[k][4]] for k in EPOCHS])
    # the standard grid, all fitting rows, for the QA
    prd = m_full[index_of[fit_all], :, n_inner:]
    logms = np.log10(np.clip(data[fit_all][:, 0, -1], 1.0, None))
    out = qa.evaluate(prd, data[fit_all], F.R_GRID, ANCHOR_Z, name=f"exp80_stage0_{name}", figdir=None,
                      figures=False, verbose=False, bin_by=lmh_bins[fit_all][:, 0], bin_by_ms=logms,
                      halo_mass_epochs=lmh_cat[fit_all])
    g_ms, g_mh = out["size_gate_ms"], out["size_gate_mh"]
    sizes = out["sizes"]
    off = {(f, k): g_ms[0][(f, k)]["offset"] for f in ("R20", "R50", "R80") for k in EPOCHS}
    wid = {(f, k): g_ms[0][(f, k)]["width_ratio"] for f in ("R20", "R50", "R80") for k in EPOCHS}
    off_mh = {(f, k): g_mh[0][(f, k)]["offset"] for f in ("R20", "R50", "R80") for k in EPOCHS}
    slope_m = {k: sizes[("R50", k)][1]["slope"] for k in EPOCHS}
    slope_t = {k: sizes[("R50", k)][0]["slope"] for k in EPOCHS}
    ir = [int(np.argmin(np.abs(F.R_GRID - r))) for r in R_SHOW]
    prof = np.full((5, len(ir), 2), np.nan)
    comp = complete[fit_all]
    for k in EPOCHS:
        for a, i in enumerate(ir):
            rel = (prd[:, k, i] - data[fit_all, k, i]) / data[fit_all, k, i]
            prof[k, a] = (100 * np.nanmedian(rel), 100 * np.nanmedian(rel[comp[:, k]]))
    c = int(np.argmin(np.abs(F.R_GRID - R_LEAK)))
    leak = np.full((5, 2), np.nan)
    y_all = np.full((len(data), 5), np.nan)
    y_all[fit_all] = np.log10(np.clip(prd[:, :, c], 1.0, None)) - np.log10(np.clip(data[fit_all][:, :, c], 1.0, None))
    growth = lmh_cat[:, 0][:, None] - lmh_cat
    for k in range(1, 5):
        rho, sl, _ = SEL.partial_growth(y_all[:, k], lmh_cat[:, k], growth[:, k], fit_all & np.isfinite(lmh_cat[:, k]))
        leak[k] = (rho, sl)
    print(f"\n  --- {label} ---")
    print(f"    radius term raw [dex] per epoch: " + " ".join(f"{v:.4f}" for v in raw) + f"  (quadrature sum {np.sqrt((raw ** 2).sum()):.4f})")
    print(f"    loss terms A/F/S/B per epoch (adopted references, frozen theta):")
    for j, k in enumerate(EPOCHS):
        print(f"      z={ANCHOR_Z[k]}: " + " ".join(f"{v:.3f}" for v in lt[j]) + f" -> {(lt[j] ** 2).sum():.3f}")
    print(f"      total {(lt ** 2).sum():.3f}")
    print(f"    SIZE GATE at fixed STELLAR mass: offset {g_ms[3]} of {g_ms[2]} | width {g_ms[4]} of {g_ms[2]};  at fixed HALO mass: offset {g_mh[3]} | width {g_mh[4]}")
    print(f"    {'':<14}" + "".join(f"{f'z={z}':>16}" for z in ANCHOR_Z))
    for f in ("R20", "R50", "R80"):
        print(f"    {f + ' offset|width':<14}" + "".join(f"{off[(f, k)]:>+9.3f} |{wid[(f, k)]:>5.2f}" for k in EPOCHS))
    print(f"    {'R50 off. (Mh)':<14}" + "".join(f"{off_mh[('R50', k)]:>+16.3f}" for k in EPOCHS))
    print(f"    mass-size slope R50 vs M*(<148): model " + " / ".join(f"{slope_m[k]:.2f}" for k in EPOCHS)
          + "; truth " + " / ".join(f"{slope_t[k]:.2f}" for k in EPOCHS))
    print(f"    profile median (model-data)/data [%], fitting | mh-complete:")
    print(f"      {'':>6}" + "".join(f"{f'M(<{F.R_GRID[i]:.0f})':>20}" for i in ir))
    for k in EPOCHS:
        print(f"      {ANCHOR_Z[k]:>6}" + "".join(f"{prof[k, a, 0]:>+9.1f} |{prof[k, a, 1]:>+8.1f}" for a in range(len(ir))))
    print(f"    future-dependence at {R_LEAK:g} kpc (partial rho | dex per dex): "
          + "  ".join(f"z={ANCHOR_Z[k]}: {leak[k, 0]:+.3f} | {leak[k, 1]:+.3f}" for k in range(1, 5)))
    print(f"    radius-term tercile medians of log R_f(model)/R_f(truth) [dex], low/mid/high:")
    for i, f in enumerate(ST.FRACTIONS):
        print(f"      R{int(100 * f):<3}" + "".join("  z=" + f"{ANCHOR_Z[j]}:" + "/".join(f"{v:+.3f}" for v in med[j, i]) for j in range(5)))
    return dict(raw=raw, med=med, loss_terms=lt, offset=np.array([[off[(f, k)] for k in EPOCHS] for f in ("R20", "R50", "R80")]),
                width=np.array([[wid[(f, k)] for k in EPOCHS] for f in ("R20", "R50", "R80")]),
                offset_mh=np.array([[off_mh[(f, k)] for k in EPOCHS] for f in ("R20", "R50", "R80")]),
                slope_model=np.array([slope_m[k] for k in EPOCHS]), slope_truth=np.array([slope_t[k] for k in EPOCHS]),
                gate_counts=np.array([g_ms[3], g_ms[4], g_mh[3], g_mh[4]]), prof=prof, leak=leak)


def build(smoke):
    recs, data, mask, lmh_dm, lmh_bins, meas, fz, spec2, th_inc, pr, Rm, truth_m, n_inner, rows_m = S0T.build(smoke)
    spec_b, th_b = RB.adopted_baseline()
    index_of = np.full(len(recs), -1)
    index_of[pr.all_rows] = np.arange(len(pr.all_rows))
    fit_all = np.zeros(len(recs), bool); fit_all[pr.all_rows] = True
    fit_all &= np.isfinite(data).all(axis=(1, 2)) & (data > 0).all(axis=(1, 2))
    sn = np.load(SEL_NPZ, allow_pickle=True)
    hs = np.load(SEL.HS_NPZ, allow_pickle=True)
    lmh_cat = SEL.sample_masses(hs)[np.array([h.row for h in recs])]
    complete = np.isfinite(lmh_cat) & (lmh_cat >= sn["cuts"][None, :])
    return dict(recs=recs, data=data, lmh_bins=lmh_bins, spec2=spec2, th_b=th_b, pr=pr, Rm=Rm, truth_m=truth_m,
                n_inner=n_inner, rows_m=rows_m, index_of=index_of, fit_all=fit_all, lmh_cat=lmh_cat, complete=complete)


def run_candidate(name, smoke):
    tag = "_smoke" if smoke else ""
    cand = CANDIDATES[name]
    print(f"{RULE}\nexp80 Stage 0 C — candidate '{name}': {cand['what']}\n  extra parameters if fitted: +{cand['extra']} "
          f"(12 + {cand['extra']} = {12 + cand['extra']}); tuned here: {cand['free']} at FROZEN amplitude, split, kernels{' (SMOKE)' if smoke else ''}\n{RULE}\n")
    B = build(smoke)
    fl = FrozenLaw(B["spec2"], B["th_b"], cand, B["pr"], B["Rm"], B["truth_m"], B["n_inner"], B["lmh_bins"], B["rows_m"], B["index_of"])
    common = (B["pr"], B["n_inner"], B["data"], B["fit_all"], B["complete"], B["lmh_bins"], B["lmh_cat"], B["index_of"], fl)
    res = {}
    if name == "control":
        x_base = np.array([B["spec2"].unpack(B["th_b"])[n] for n in fl.free])
        res["baseline"] = score(name, fl.predict(x_base), *common, label="THE BASELINE (no change)")
    print(f"\n  hand-set start: {fl.describe(fl.x0)}")
    res["hand"] = score(name, fl.predict(fl.x0), *common, label=f"'{name}' HAND-SET from the table")
    t0 = time.time(); fl.n_eval = 0
    l0 = fl.loss(fl.x0)
    r = minimize_loss(fl.loss, fl.x0, method="neldermead", bounds=fl.bounds,
                      max_evals=MAX_TUNE_EVALS["smoke" if smoke else "full"])
    print(f"\n  Z-only tune: {l0:.5f} -> {r.fun:.5f} in {fl.n_eval} evaluations, {(time.time() - t0) / 60:.1f} min")
    print(f"  tuned: {fl.describe(r.x)}")
    res["tuned"] = score(name, fl.predict(r.x), *common, label=f"'{name}' TUNED on the radius term")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    save = dict(name=name, free=np.array(fl.free), x0=fl.x0, x_tuned=np.asarray(r.x), loss0=l0, loss_tuned=float(r.fun),
                n_eval=fl.n_eval, extra=cand["extra"], law_keys=np.array(list(cand["law"])),
                law_values=np.array([cand["law"][k] for k in cand["law"]], float), theta_baseline=B["th_b"])
    for lab, d in res.items():
        for k, v in d.items():
            save[f"{lab}_{k}"] = v
    np.savez(OUTDIR / f"stage0_cand_{name}{tag}.npz", **save)
    print(f"\n  wrote {OUTDIR / f'stage0_cand_{name}{tag}.npz'}")


def merge(smoke):
    tag = "_smoke" if smoke else ""
    files = {n: OUTDIR / f"stage0_cand_{n}{tag}.npz" for n in CANDIDATES}
    files = {n: f for n, f in files.items() if f.exists()}
    assert "control" in files, "run the control first"
    fs = {n: dict(np.load(f, allow_pickle=True)) for n, f in files.items()}
    base = fs["control"]
    b_r50 = base["baseline_offset"][1]
    c_r50 = fs["control"]["tuned_offset"][1]
    print(f"{RULE}\nexp80 Stage 0 C — THE VERDICT TABLE (frozen amplitude, split, kernels; the law's constants tuned on the radius term)\n{RULE}")
    print(f"  gate: recover > {GATE_RECOVER_DEX} dex of the z = 2 R50 offset (fixed M*, standard grid, fitting sample) BEYOND the control, "
          f"with |R50 offset| <= {GATE_LOW_Z_TOL} at z <= 1\n")
    hdr = f"  {'candidate':<18}{'+par':>5}{'Z sum':>8}{'loss':>8}{'off/15':>7}{'wid':>4}" + "".join(f"{f'R50 z={z}':>10}" for z in ANCHOR_Z) \
        + f"{'R80 z=2':>9}{'slope z=2':>10}{'M(<2) z=.4':>11}{'M(<10) z=2':>11}{'M(50-100)':>10}{'leak z=1':>9}{'verdict':>10}"
    print(hdr)
    rows = [("baseline", base, "baseline_")]
    for n in CANDIDATES:
        if n in fs:
            rows.append((n + " (hand)", fs[n], "hand_"))
            rows.append((n + " (tuned)", fs[n], "tuned_"))
    verdicts = {}
    for lab, d, pre in rows:
        off = d[pre + "offset"]; wid = d[pre + "width"]; lt = d[pre + "loss_terms"]; prof = d[pre + "prof"]; raw = d[pre + "raw"]
        r50 = off[1]
        extra = int(d["extra"]) if pre != "baseline_" else 0
        rec_base = b_r50[4] - r50[4]
        rec_ctrl = c_r50[4] - r50[4]
        low_ok = np.all(np.abs(r50[:3]) <= GATE_LOW_Z_TOL)
        if pre == "tuned_" and lab != "control (tuned)":
            ok = (rec_ctrl > GATE_RECOVER_DEX) and (rec_base > GATE_RECOVER_DEX) and low_ok
            verdicts[lab] = ok
            v = "PASS" if ok else "fail"
        else:
            v = "-"
        gc = d[pre + "gate_counts"]
        print(f"  {lab:<18}{extra:>5}{np.sqrt((raw ** 2).sum()):>8.4f}{(lt ** 2).sum():>8.2f}{gc[0]:>7}{gc[1]:>4}"
              + "".join(f"{r50[k]:>+10.3f}" for k in EPOCHS) + f"{off[2, 4]:>+9.3f}{d[pre + 'slope_model'][4]:>10.2f}"
              + f"{prof[0, 0, 0]:>+10.1f}%{prof[4, 1, 0]:>+10.1f}%{prof[4, 2, 0]:>+9.1f}%{d[pre + 'leak'][1, 1]:>+9.3f}{v:>10}")
    print(f"\n  truth's R50 mass-size slope at z = 2: {base['baseline_slope_truth'][4]:.2f}; the baseline's z = 2 R50 offset {b_r50[4]:+.3f}, "
          f"the control's {c_r50[4]:+.3f} (re-tuning alone recovers {b_r50[4] - c_r50[4]:+.3f} dex)")
    print(f"  tuned constants:")
    for n in CANDIDATES:
        if n in fs:
            d = fs[n]
            print(f"    {n:<18}" + ", ".join(f"{k} {v:+.3f}" for k, v in zip(d["free"], d["x_tuned"])) + f"  ({int(d['n_eval'])} evals)")
    passing = [l for l, ok in verdicts.items() if ok]
    print(f"\n  GATE: " + (f"PASS for {passing}" if passing else "no candidate passes") + "\n")


if __name__ == "__main__":
    a = sys.argv
    if "--merge" in a:
        merge("--smoke" in a)
    else:
        run_candidate(a[a.index("--candidate") + 1], "--smoke" in a)
