"""exp87 Stage 3 extras — readings the report and the figures need, on the
parent at z = 0.4 (development folds only), saved to `outputs/extras.npz`.

1. TRUNCATION ON REAL DATA. The ordinary normal fit, the ordinary fit
   renormalised above the cut, and the truncated-likelihood fit, for the
   scalar stellar-mass input and for the whole 24-point profile.
2. THE SAFE ZONE (the plan's check). Galaxies whose inputs alone put less than
   1% of the latent distribution below the cut: trained and scored INSIDE the
   zone, the ordinary and the truncated fits must agree.
3. WHERE THE INFORMATION IS. Each cumulative mass M*(<R) alone; the profile
   inside R only; the profile outside R only.
4. PROJECTION. The 6-aperture masses exist along three axes at z = 0.4: a
   model trained on the xy apertures, read on xz and yz for the same
   galaxies — the scatter of the predicted halo mass between projections, and
   the score on another projection.
5. THE PRIOR. The forward relation's slope and scatter for M*(<148), the
   implied flat-prior resolution, the box mass function's log-slope near the
   cut and the mean shift it implies.

Run:  PYTHONPATH=. uv run python -u experiments/exp87_inverse_halo/stage3_extras.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C                                       # noqa: E402
import data as D                                         # noqa: E402
import harness as HN                                     # noqa: E402
import heads as H                                        # noqa: E402
import methods as M                                      # noqa: E402
import scoring as S                                      # noqa: E402

RULE = "=" * 100
CUT = C.PARENT_CUT


def crps_of(method, X, y, fold, cut):
    per, _ = HN.oof_scores(method, X, y, fold, cut)
    return per


def main():
    print(f"{RULE}\nexp87 STAGE 3 extras — the parent at z = 0.4, development folds (config {C.config_hash()})\n{RULE}")
    s = D.load_parent(verbose=False)
    y = s.targets["mh"]
    fold = np.where(~s.lockbox, s.fold, -1)
    used = fold >= 0
    X24 = s.logcog
    out, js = {}, {}

    print("\n1. THE TRUNCATION ON REAL DATA (mean CRPS, dex)")
    for name, X in (("M*(<148) alone", X24[:, [-1]]), ("the 24-point profile", X24)):
        trunc = crps_of(M.DirectLinear("t", "L1"), X, y, fold, CUT)
        naive = crps_of(M.DirectLinear("n", "L1"), X, y, fold, None)
        double = S.TruncNormal(naive["latent_m"][used], naive["latent_s"][used], CUT)
        d_crps, d_pit = double.crps(y[used]), double.pit(y[used])
        below = float(np.mean(S.TruncNormal(naive["latent_m"][used], naive["latent_s"][used], None).pit(np.full(used.sum(), CUT))))
        key = "mtot" if X.shape[1] == 1 else "raw24"
        out.update({f"{key}_trunc_crps": trunc["crps"], f"{key}_naive_crps": naive["crps"], f"{key}_trunc_pit": trunc["pit"],
                    f"{key}_naive_pit": naive["pit"], f"{key}_trunc_m": trunc["latent_m"], f"{key}_trunc_s": trunc["latent_s"],
                    f"{key}_naive_m": naive["latent_m"], f"{key}_naive_s": naive["latent_s"],
                    f"{key}_trunc_mean": trunc["pred_mean"], f"{key}_naive_mean": naive["pred_mean"]})
        ok_d, boot_d = S.significant_gain(trunc["crps"][used], naive["crps"][used], 0.01)
        js[f"{key}_truncation"] = dict(trunc=float(trunc["crps"][used].mean()), naive=float(naive["crps"][used].mean()), double=float(d_crps.mean()),
                                        gain_over_naive=-boot_d["rel"], naive_mass_below_cut=below,
                                        pit_var=dict(trunc=float(np.var(trunc["pit"][used])), naive=float(np.var(naive["pit"][used])), double=float(np.var(d_pit))),
                                        scale=dict(trunc=float(np.median(trunc["latent_s"][used])), naive=float(np.median(naive["latent_s"][used]))))
        print(f"   {name:<22} truncated likelihood {trunc['crps'][used].mean():.4f} | ordinary normal {naive['crps'][used].mean():.4f} "
              f"(the truncated fit gains {100 * -boot_d['rel']:.1f}% [{100 * -boot_d['rel_hi']:.1f}, {100 * -boot_d['rel_lo']:.1f}]) | ordinary renormalised above the cut "
              f"{d_crps.mean():.4f}; the ordinary predictive puts {100 * below:.1f}% of its mass below the cut; latent scale {np.median(trunc['latent_s'][used]):.3f} "
              f"vs the ordinary fit's {np.median(naive['latent_s'][used]):.3f} dex; PIT variance {np.var(trunc['pit'][used]):.4f} / {np.var(naive['pit'][used]):.4f} (uniform 0.0833)")

    print("\n2. THE SAFE ZONE: inputs alone put < 1% of the latent distribution below the cut; trained and scored inside the zone")
    for key, X in (("mtot", X24[:, [-1]]), ("raw24", X24)):
        zone = used & (out[f"{key}_trunc_m"] - 2.326 * out[f"{key}_trunc_s"] >= CUT)
        fz = np.where(zone, fold, -1)
        t_in = crps_of(M.DirectLinear("t", "L1"), X, y, fz, CUT)["crps"][zone].mean()
        n_in = crps_of(M.DirectLinear("n", "L1"), X, y, fz, None)["crps"][zone].mean()
        global_naive = out[f"{key}_naive_crps"][zone].mean()
        global_trunc = out[f"{key}_trunc_crps"][zone].mean()
        js[f"{key}_safe_zone"] = dict(n=int(zone.sum()), fraction=float(zone.sum() / used.sum()), trunc=float(t_in), naive=float(n_in),
                                       global_trunc=float(global_trunc), global_naive=float(global_naive))
        print(f"   {key:<6} zone: {zone.sum()} galaxies ({100 * zone.sum() / used.sum():.0f}%): truncated {t_in:.4f} vs ordinary {n_in:.4f} "
              f"(difference {100 * (n_in / t_in - 1):+.2f}%); the GLOBAL fits read in the zone: truncated {global_trunc:.4f}, ordinary {global_naive:.4f} "
              f"({100 * (global_naive / global_trunc - 1):+.1f}%: a global ordinary fit is biased everywhere)")

    print("\n3. WHERE THE INFORMATION IS (CRPS, dex)")
    single = np.array([crps_of(M.DirectLinear("r", "L0"), X24[:, [i]], y, fold, CUT)["crps"][used].mean() for i in range(24)])
    inner = np.array([crps_of(M.DirectLinear("i", "L1"), X24[:, :i + 1], y, fold, CUT)["crps"][used].mean() for i in range(24)])
    outer = np.array([crps_of(M.DirectLinear("o", "L1"), X24[:, i:], y, fold, CUT)["crps"][used].mean() for i in range(24)])
    out.update(single_radius_crps=single, inner_crps=inner, outer_crps=outer, radii=s.radii)
    print("   R [kpc]         " + " ".join(f"{r:>6.1f}" for r in s.radii[::3]))
    print("   M*(<R) alone    " + " ".join(f"{v:>6.4f}" for v in single[::3]))
    print("   profile inside R" + " ".join(f"{v:>6.4f}" for v in inner[::3]))
    print("   profile outside " + " ".join(f"{v:>6.4f}" for v in outer[::3]))
    print(f"   the best single cumulative mass is M*(<{s.radii[int(np.argmin(single))]:.0f} kpc) ({single.min():.4f}); the profile inside 30 kpc "
          f"reaches {inner[11]:.4f}, the profile outside 30 kpc {outer[11]:.4f}, all 24 points {inner[-1]:.4f}")

    print("\n4. PROJECTION: six apertures (10-150 kpc) along three axes")
    ap = s.flags["aper_proj"]                                 # (n, 3, 6)
    okp = np.isfinite(ap).all(axis=(1, 2))
    fp = np.where(okp, fold, -1)
    up = fp >= 0
    means = np.full((len(y), 3), np.nan)
    crps_proj = np.full((len(y), 3), np.nan)
    for f in range(C.N_FOLDS):
        tr, te = up & (fp != f), fp == f
        model = M.DirectLinear("xy", "L1")
        model.fit_predict(ap[tr, 0], y[tr], ap[te, 0], CUT)
        coef = np.linalg.lstsq(np.column_stack([np.ones(tr.sum()), ap[tr, 0]]), y[tr], rcond=None)[0]
        for j in range(3):
            Ds = (ap[te, j] @ coef[1:] + coef[0])[:, None]
            pred = model.model.predict(ap[te, j], Ds)
            means[te, j], crps_proj[te, j] = pred.mean(), pred.crps(y[te])
    jitter = float(np.sqrt(np.mean(np.var(means[up], axis=1, ddof=1))))
    js["projection"] = dict(n=int(up.sum()), jitter_dex=jitter, crps=[float(v) for v in np.mean(crps_proj[up], axis=0)],
                            resid_rmse=float(np.sqrt(np.mean((means[up, 0] - y[up]) ** 2))))
    out.update(projection_means=means)
    print(f"   {up.sum()} galaxies: the predicted halo mass changes between projections by {jitter:.4f} dex (rms about the three-axis mean), "
          f"against a prediction error of {js['projection']['resid_rmse']:.3f} dex; CRPS on xy / xz / yz (model trained on xy): "
          + " / ".join(f"{v:.4f}" for v in js["projection"]["crps"]))

    print("\n5. THE PRIOR")
    prior = D.box_prior()
    m148 = X24[used, -1]
    b = np.polyfit(y[used], m148, 1)
    sx = float(np.std(m148 - np.polyval(b, y[used])))
    grid = np.array([13.0, 13.3, 13.6, 14.0])
    gamma = -np.gradient(D.log_prior_density(np.linspace(12.9, 14.2, 14), 0, prior), np.linspace(12.9, 14.2, 14)) / np.log(10.0)
    g_near = float(np.mean(gamma[:6]))
    fit = H.TruncLinear(CUT).fit(m148[:, None], y[used])
    slope = float(fit.raw_coefficients()[1][0])
    scale = float(np.median(fit.latent(m148[:, None])[1]))
    js["prior"] = dict(forward_slope=float(b[0]), forward_scatter=sx, flat_prior_resolution=float(sx / b[0]), gamma_near_cut=g_near,
                       implied_shift=float(g_near * np.log(10.0) * (sx / b[0]) ** 2), latent_slope=slope, latent_scale=scale)
    print(f"   forward: M*(<148) = {b[0]:.3f} x log Mh + const, scatter {sx:.3f} dex -> flat-prior resolution sigma_x / b = {sx / b[0]:.3f} dex and "
          f"inverse slope 1/b = {1 / b[0]:.3f}; the truncated-likelihood latent line: slope {slope:.3f}, scale {scale:.3f} dex")
    print(f"   the box's mass function falls as 10^(-{g_near:.2f} log M) just above the cut, so the population's mean halo mass at fixed stellar mass "
          f"lies {g_near * np.log(10.0) * (sx / b[0]) ** 2:.3f} dex BELOW the inverted forward relation (the Eddington-type shift, prior-dependent)")
    del grid
    np.savez(C.OUTDIR / "extras.npz", y=y, fold=fold, index=s.index, mtot=X24[:, -1], **out)
    (C.OUTDIR / "extras.json").write_text(json.dumps(js, indent=1))
    print(f"\n  wrote {C.OUTDIR / 'extras.npz'} and extras.json")


if __name__ == "__main__":
    main()
