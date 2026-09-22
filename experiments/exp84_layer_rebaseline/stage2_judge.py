"""exp84 Stage 2 — compose the layer and judge it HELD OUT, both ways.

Protocol (exp41 / exp60): the fitting sample is split in two random halves;
the sampler is fitted on one (Stage 1's anatomy rows of that half, the
amplitude calibration on that half) and scored on the other, then the halves
swap; N_REAL realizations per swap; every number below is the mean over
realizations (and over the two swaps) with its scatter across realizations.

THE VARIANTS (the plan's a-d):
  gauss         the fitted layer: Gaussian, per-epoch widths and the 10x10
                cross-epoch / cross-axis correlation from the anatomy
  gauss-centred the same with per-axis, per-epoch OFFSETS calibrated so the
                drawn median profile stays on the mean at 4.9 / 32.6 kpc (S3
                by construction inside the engine; the offsets are printed)
  gauss-scaled  the same with ONE size scale calibrated so the R50 width at
                fixed stellar mass is the truth's on the calibration half
                (R20 and R80 are then the tests, not calibrations)
  gauss-2scale  TWO scales: the compact axis's on the R20 width, the extended
                axis's on the R80 width (R50 is then the test); added after
                the first run showed R20 over-dispersed 1.4x in every variant
                (the compact axis's anatomy width is inflated by its flat loss
                valley) and R80 under at z >= 1.5 with one scale
  gauss-2sc-cent the two scales AND the profile-centring offsets — the candidate
  rows          the nonparametric control: whole anatomy records resampled
  independent   correlation = identity (the epochs draw independently)
  persistent    correlation = 1 within an axis (a persistent trait)

THE GATES, all on the draws (`qa.evaluate_draws`, the same code path as a
mean): tier 2d R20 / R50 / R80 offset and width SEPARATELY at fixed stellar
and at fixed halo mass, all five epochs; S3 the drawn median CoG on the
mean's; S5 the amplitude width and coherence; S4 the 52-148 kpc annulus;
the persistence diagnostic; tier 2e; S1 for the record. Rows: the adopted
mean (null), v1 on its own mean (exp73's saved scores), each variant.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp84_layer_rebaseline/stage2_judge.py [--smoke] [--fast]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import layer_common as LC                                # noqa: E402
import predictor as P                                    # noqa: E402
from sampler import Sampler                              # noqa: E402
from hongshao import qa                                  # noqa: E402

RULE, THIN = P.RULE, P.THIN
OUT = P.OUTDIR / "stage2_judge.npz"
E73 = P.ROOT / "experiments/exp73_size_relative/outputs/size_gate_layer.npz"
SEED, N_REAL = 84, 8
I52, I148 = 15, 23                                       # exp60's annulus indices on R_GRID
VARIANTS = (("gauss", dict(form="gauss", corr="fitted", scale=1.0)),
            ("gauss-centred", dict(form="gauss", corr="fitted", scale=1.0, centred=True)),
            ("gauss-scaled", dict(form="gauss", corr="fitted", scale=None)),
            ("gauss-2scale", dict(form="gauss", corr="fitted", scale=1.0, two_scales=True)),
            ("gauss-2sc-cent", dict(form="gauss", corr="fitted", scale=1.0, two_scales=True, centred=True)),
            ("rows", dict(form="rows", corr="fitted", scale=1.0)),
            ("independent", dict(form="gauss", corr="identity", scale=1.0)),
            ("persistent", dict(form="gauss", corr="one", scale=1.0)))
S3_TOL, S5_TOL_SIGMA, S5_TOL_CORR, PERSIST_TOL = 0.010, 0.10, 0.05, 0.15


def width_ratio(draw_cogs, data, R, key="R50"):
    """Mean over epochs of (the draw's size scatter at fixed M*(<148) / the truth's)."""
    lm_t = np.log10(np.clip(data[:, :, -1], 1.0, None))
    lm_m = np.log10(np.clip(draw_cogs[:, :, -1], 1.0, None))
    rt = LC.size_residuals(LC.log_sizes(data, R, (key,))[key], lm_t)
    rm = LC.size_residuals(LC.log_sizes(draw_cogs, R, (key,))[key], lm_m)
    return float(np.nanmean(np.nanstd(rm, axis=0) / np.nanstd(rt, axis=0)))


def r50_width_ratio(draw_cogs, data, R):
    return width_ratio(draw_cogs, data, R, "R50")


def calibrate_two_scales(smp, predict_fn, data, R, n, rng, n_real=3, n_round=2):
    """The two-scale variant: the compact axis's scale on the R20 width, the
    extended axis's on the R80 width (alternating bisections); R50 is then
    the test of the two axes together."""
    for _ in range(n_round):
        for i_axis, key in ((0, "R20"), (1, "R80")):
            lo, hi = 0.05, 3.0
            for _ in range(10):
                smp.scale_axis[i_axis] = 0.5 * (lo + hi)
                ratio = np.mean([width_ratio(predict_fn(smp.draw_sizes(n, rng)), data, R, key) for _ in range(n_real)])
                if ratio < 1.0:
                    lo = smp.scale_axis[i_axis]
                else:
                    hi = smp.scale_axis[i_axis]
            smp.scale_axis[i_axis] = 0.5 * (lo + hi)
    return smp.scale_axis.copy()


def calibrate_scale(smp, predict_fn, data, R, n, rng, n_real=3):
    """Bisection on the size scale so the calibration half's drawn R50 width is the truth's."""
    lo, hi = 0.05, 3.0
    for _ in range(12):
        smp.scale = 0.5 * (lo + hi)
        ratio = np.mean([r50_width_ratio(predict_fn(smp.draw_sizes(n, rng)), data, R) for _ in range(n_real)])
        if ratio < 1.0:
            lo = smp.scale
        else:
            hi = smp.scale
    smp.scale = 0.5 * (lo + hi)
    return smp.scale


def direct_gates(draws, mean, data, R, t0):
    """S3, S5, S4, the persistence and S1 on one set of realizations (S, n, 5, nr)."""
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        lg = np.log10(np.clip(draws, 1.0, None))
        lmean = np.log10(np.clip(mean, 1.0, None))
        med = np.nanmedian(lg.reshape(-1, *lg.shape[2:]), axis=0)          # (5, nr), pooled over draws
        out["s3_all"] = float(np.nanmax(np.abs(med - np.nanmedian(lmean, axis=0))))
        out["s3_lowz"] = float(np.nanmax(np.abs(med - np.nanmedian(lmean, axis=0))[:3]))
        dev = lg[:, :, :, LC.F.I100] - lmean[None, :, :, LC.F.I100]         # (S, n, 5)
        out["s5_sigma"] = LC.half_width(dev.reshape(-1, 5))
        cors = [np.corrcoef(d[np.isfinite(d).all(axis=1)].T) for d in dev]
        out["s5_near"] = float(np.mean([LC.nearest_epoch(c) for c in cors]))
        ann_m = draws[:, :, 0, I148] - draws[:, :, 0, I52]
        ann_t = data[:, 0, I148] - data[:, 0, I52]
        ok = np.isfinite(ann_m) & (ann_m > 0) & (ann_t > 0)[None, :]
        out["s4"] = float(np.nanmedian(np.where(ok, np.log10(ann_m / ann_t[None, :]), np.nan)))
        pers = {k: [] for k in LC.SIZE_KEYS}
        s1 = []
        for d in draws:
            lm = np.log10(np.clip(d[:, :, -1], 1.0, None))
            ls = LC.log_sizes(d, R)
            for k in LC.SIZE_KEYS:
                pers[k].append(LC.nearest_epoch(LC.persistence(LC.size_residuals(ls[k], lm))))
            dc = np.log10(d[:, 0, LC.I5] / d[:, 4, LC.I5])
            g = np.isfinite(dc)
            s1.append((np.mean(dc[g] < 0), np.median(dc[g]), np.median(dc[g & (dc < 0)]) if (dc[g] < 0).any() else np.nan))
        out["persistence"] = {k: (float(np.mean(v)), float(np.std(v))) for k, v in pers.items()}
        out["s1"] = np.nanmean(np.array(s1), axis=0)
    return out


def average_swaps(a, b):
    """Average two evaluate_draws outputs cell by cell (pass flags recomputed)."""
    def gate(ga, gb):
        out, n_off, n_wid = {}, 0, 0
        for c in ga:
            o = 0.5 * (ga[c]["offset"] + gb[c]["offset"])
            w = 0.5 * (ga[c]["width_ratio"] + gb[c]["width_ratio"])
            po, pw = abs(o) <= qa.SIZE_GATE_OFFSET, abs(w - 1) <= qa.SIZE_GATE_WIDTH
            n_off += po; n_wid += pw
            out[c] = dict(offset=o, offset_sd=0.5 * (ga[c]["offset_sd"] + gb[c]["offset_sd"]), width_ratio=w,
                          width_sd=0.5 * (ga[c]["width_sd"] + gb[c]["width_sd"]), pass_offset=po, pass_width=pw)
        return out, (n_off, n_wid, len(out))
    gm, cm = gate(a["gate_ms"], b["gate_ms"])
    gh, ch = gate(a["gate_mh"], b["gate_mh"])
    cd = {c: {k: 0.5 * (a["cdfs"][c][k] + b["cdfs"][c][k]) for k in a["cdfs"][c]} for c in a["cdfs"]}
    return dict(gate_ms=gm, gate_mh=gh, counts_ms=cm, counts_mh=ch, cdfs=cd)


def v1_rows():
    """exp73's saved scores of the v1 layer on ITS OWN mean (the incumbent): {(label, src): gate dict}."""
    if not E73.exists():
        return {}
    z = np.load(E73, allow_pickle=True)
    out = {}
    for key, (o, w, so, sw) in zip(z["keys"], z["vals"]):
        label, k, j, src = str(key).split("|")
        out.setdefault((label, src), {})[(k, int(j))] = dict(offset=float(o), width_ratio=float(w), width_sd=float(sw))
    return out


def main(smoke=False, fast=False, only=None):
    variants = VARIANTS if only is None else tuple(v for v in VARIANTS if v[0] in only)
    assert variants, only
    out_path = OUT if only is None else OUT.with_name(OUT.stem + "_" + "-".join(v[0] for v in variants) + ".npz")
    print(f"{RULE}\nexp84 STAGE 2 — the layer composed and judged held out, both ways\n{RULE}\n")
    print(f"  variants: " + ", ".join(v[0] for v in variants))
    n_real = 2 if (smoke or fast) else N_REAL
    recs, data, keep, lmh, pred = P.build(smoke)
    rows_all = np.where(keep)[0]
    n = len(rows_all)
    R = pred.R
    d_all, lmh_all = data[rows_all], lmh[rows_all]
    mean_all = pred.predict(rows=rows_all)
    t0 = np.load(P.OUTDIR / "stage0_targets.npz")
    if smoke:                                             # the smoke anatomy is not saved: a stand-in
        rng0 = np.random.default_rng(0)
        delta_c, delta_e = 0.3 * rng0.standard_normal((n, 5)), 0.15 * rng0.standard_normal((n, 5))
    else:
        a1 = np.load(P.OUTDIR / "stage1_anatomy.npz")
        assert np.array_equal(a1["rows"], rows_all), "the anatomy's rows are not this sample's"
        delta_c, delta_e = a1["delta_c"], a1["delta_e"]
    amp_sigma, amp_corr = t0["amp_sigma"], t0["amp_corr"]

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    halves = (perm[: n // 2], perm[n // 2:])
    print(f"  {n} galaxies; halves of {len(halves[0])} / {len(halves[1])}; {n_real} realizations per swap; "
          f"amplitude target " + " ".join(f"{v:.3f}" for v in amp_sigma))

    results = {name: [] for name, _ in variants}
    direct = {name: [] for name, _ in variants}
    described = {}
    t_start = time.time()
    for swap in range(2):
        calib, score = halves[swap], halves[1 - swap]
        rows_c, rows_s = rows_all[calib], rows_all[score]
        pf_c = lambda dev, rc=rows_c: pred.predict(size_dev={a: _expand(dev[a], rc, pred.n) for a in P.AXES}, rows=rc)  # noqa: E731
        pf_s = lambda dev, rs=rows_s: pred.predict(size_dev={a: _expand(dev[a], rs, pred.n) for a in P.AXES}, rows=rs)  # noqa: E731
        mean_c, mean_s = mean_all[calib], mean_all[score]
        for name, kw in variants:
            rng_v = np.random.default_rng(SEED + 100 * swap + [v[0] for v in VARIANTS].index(name))
            smp = Sampler(delta_c, delta_e, amp_sigma, amp_corr, calib, form=kw["form"], corr=kw["corr"],
                          scale=1.0 if kw["scale"] is None else kw["scale"])
            if kw["scale"] is None:
                calibrate_scale(smp, pf_c, d_all[calib], R, len(calib), rng_v)
            if kw.get("two_scales"):
                calibrate_two_scales(smp, pf_c, d_all[calib], R, len(calib), rng_v)
            if kw.get("centred"):
                hist = smp.calibrate_offsets(pf_c, mean_c, len(calib), rng_v)
                print(f"    centring: max |median excess| at 4.9 / 32.6 kpc per iteration " + " ".join(f"{h:.3f}" for h in hist))
            induced = smp.calibrate_amplitude(pf_c, mean_c, len(calib), rng_v)
            described[name] = smp.describe()
            draws = np.stack([smp.profiles(pf_s, len(score), rng_v) for _ in range(n_real)])
            ev = qa.evaluate_draws(draws, d_all[score], R, list(P.ANCHOR_Z), halo_mass_epochs=lmh_all[score])
            dg = direct_gates(draws, mean_s, d_all[score], R, t0)
            results[name].append(ev)
            direct[name].append(dg)
            print(f"  swap {swap} {name:<13} scale {smp.scale:.3f}; induced amplitude " + " ".join(f"{v:.3f}" for v in induced)
                  + "; sig_add " + " ".join(f"{v:.3f}" for v in smp.sig_add)
                  + f"; 2d M* offset {ev['counts_ms'][0]}/{ev['counts_ms'][2]} width {ev['counts_ms'][1]}/{ev['counts_ms'][2]}"
                  f"; S3 {dg['s3_lowz']:.3f}  ({(time.time() - t_start) / 60:.1f} min)", flush=True)

    # --- the tables ------------------------------------------------------------
    ev_mean = qa.evaluate(mean_all, d_all, R, list(P.ANCHOR_Z), figdir=None, figures=False, verbose=False,
                          bin_by=lmh_all[:, 0], halo_mass_epochs=lmh_all)
    merged = {name: average_swaps(*results[name]) for name in results}
    v1 = v1_rows()
    print(f"\n{RULE}\n  THE SAMPLERS (fitted on the second calibration half)\n{RULE}")
    for name in described:
        print(f"  {name:<13} {described[name]}")
    for which, label, v1label in (("ms", "STELLAR mass", "STELLAR mass"), ("mh", "HALO mass", "HALO mass")):
        print(f"\n{RULE}\n  TIER 2d at fixed {label} — the adopted MEAN (null), v1 on the INCUMBENT (exp73's record), "
              f"the v2 variants (held out, mean over {2 * n_real} realizations, ± across them)\n{RULE}")
        rows = [("mean", ev_mean[f"size_gate_{which}"][0])]
        if (v1label, "layer") in v1:
            rows.append(("v1 (incumbent)", v1[(v1label, "layer")]))
        rows += [(name, merged[name][f"gate_{which}"]) for name in merged]
        qa.print_draw_gate(rows, P.ANCHOR_Z, label)
    print(f"\n{RULE}\n  THE OTHER GATES (held out, both swaps)\n{RULE}")
    print(f"  {'variant':<13}{'S3 z<=1':>9}{'S3 all':>8}{'S5 sigma ratio (per epoch)':>34}{'S5 near':>9}{'S4':>8}"
          f"{'persist R20/R50/R80':>22}{'S1 frac':>9}{'S1 med':>8}{'S1 dec':>8}")
    pt = {k: LC.nearest_epoch(t0[f"persistence_truth_{k}"]) for k in LC.SIZE_KEYS}
    print(f"  {'truth':<13}{'':>9}{'':>8}{'':>34}{LC.nearest_epoch(amp_corr):>+9.3f}{0.0:>8.3f}"
          f"{'/'.join(f'{pt[k]:+.2f}' for k in LC.SIZE_KEYS):>22}{100 * float(t0['s1_frac']):>8.1f}%{float(t0['s1_med']):>+8.3f}{float(t0['s1_med_dec']):>+8.3f}")
    summary = {}
    for name in merged:
        dg = direct[name]
        s3l = np.mean([g["s3_lowz"] for g in dg]); s3a = np.mean([g["s3_all"] for g in dg])
        s5 = np.mean([g["s5_sigma"] for g in dg], axis=0) / amp_sigma
        near = np.mean([g["s5_near"] for g in dg])
        s4 = np.mean([g["s4"] for g in dg])
        pers = {k: np.mean([g["persistence"][k][0] for g in dg]) for k in LC.SIZE_KEYS}
        s1 = np.mean([g["s1"] for g in dg], axis=0)
        flags = (("S3" if s3l <= S3_TOL else "s3!") + " " + ("S5" if (np.abs(s5 - 1) <= S5_TOL_SIGMA).all()
                 and abs(near - LC.nearest_epoch(amp_corr)) <= S5_TOL_CORR else "s5!")
                 + " " + ("P" if all(abs(pers[k] - pt[k]) <= PERSIST_TOL for k in LC.SIZE_KEYS) else "p!"))
        summary[name] = dict(s3_lowz=s3l, s3_all=s3a, s5_ratio=s5, s5_near=near, s4=s4, persistence=pers, s1=s1, flags=flags)
        print(f"  {name:<13}{s3l:>9.3f}{s3a:>8.3f}{' '.join(f'{v:.2f}' for v in s5):>34}{near:>+9.3f}{s4:>+8.3f}"
              f"{'/'.join(f'{pers[k]:+.2f}' for k in LC.SIZE_KEYS):>22}{100 * s1[0]:>8.1f}%{s1[1]:>+8.3f}{s1[2]:>+8.3f}   {flags}")
    print(f"    (S3 pass <= {S3_TOL} dex at z <= 1; S5 sigma within {int(100 * S5_TOL_SIGMA)}% and nearest-epoch correlation "
          f"within {S5_TOL_CORR}; P = persistence within {PERSIST_TOL} of the truth's on every size)")
    print(f"\n{RULE}\n  TIER 2e on the draws — the mean, then the variants\n{RULE}")
    qa.print_draw_cdfs([("mean", ev_mean["cdfs"])] + [(name, merged[name]["cdfs"]) for name in merged], P.ANCHOR_Z)

    print(f"\n{RULE}\n  THE COUNT — offsets / widths passing of 15 (fixed M* | fixed Mh)\n{RULE}")
    _, _, _, mo, mw = ev_mean["size_gate_ms"]; _, _, _, ho, hw = ev_mean["size_gate_mh"]
    print(f"  {'mean':<13} {mo:>2}/15 {mw:>2}/15 | {ho:>2}/15 {hw:>2}/15")
    for name in merged:
        cm, ch = merged[name]["counts_ms"], merged[name]["counts_mh"]
        print(f"  {name:<13} {cm[0]:>2}/15 {cm[1]:>2}/15 | {ch[0]:>2}/15 {ch[1]:>2}/15   {summary[name]['flags']}")
    print(f"\n  ({pred.n_call} model calls, {(time.time() - t_start) / 60:.1f} min)")

    if smoke:
        print("\n  (smoke: nothing saved)")
        return
    np.savez(out_path, variants=np.array([v[0] for v in variants]), n_real=n_real, seed=SEED,
             **{f"gate_ms_{name}_keys": np.array([f"{k}|{j}" for (k, j) in merged[name]["gate_ms"]]) for name in merged},
             **{f"gate_ms_{name}_vals": np.array([[v["offset"], v["width_ratio"], v["width_sd"]] for v in merged[name]["gate_ms"].values()]) for name in merged},
             **{f"gate_mh_{name}_vals": np.array([[v["offset"], v["width_ratio"], v["width_sd"]] for v in merged[name]["gate_mh"].values()]) for name in merged},
             **{f"cdf_{name}_vals": np.array([[v["ks_ratio"], v["w1_ratio"]] for v in merged[name]["cdfs"].values()]) for name in merged},
             **{f"summary_{name}": np.array([summary[name]["s3_lowz"], summary[name]["s3_all"], summary[name]["s5_near"], summary[name]["s4"],
                                             *summary[name]["s5_ratio"], *[summary[name]["persistence"][k] for k in LC.SIZE_KEYS],
                                             *summary[name]["s1"]]) for name in merged},
             described=np.array([f"{k}: {v}" for k, v in described.items()]))
    print(f"  saved -> {out_path.relative_to(P.ROOT)}")


def _expand(v, idx, n_total):
    """Scatter a (m, 5) deviation of the galaxies `idx` into an (n_total, 5) array."""
    out = np.zeros((n_total, 5))
    out[idx] = v
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    main(smoke="--smoke" in a, fast="--fast" in a,
         only=a[a.index("--variants") + 1].split(",") if "--variants" in a else None)
