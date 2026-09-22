"""exp84 Stage 0 — the targets, frozen on THE ADOPTED MEAN before the layer is
built (exp60 Stage 0's role; the gates never read the data again).

  * S5: the mean's amplitude residual at 103 kpc on the fitting sample —
    per-epoch half 16-84 width and the 5x5 cross-epoch correlation. The
    amplitude draw must reproduce both.
  * S3 reference: the mean's median log CoG per epoch (the drawn population's
    median must stay on it).
  * THE PERSISTENCE DIAGNOSTIC: the truth's cross-epoch Spearman correlation
    of the size residual at fixed stellar mass (log R20/R50/R80 about the OLS
    line on log M*(<148 kpc), the tier 2d width's own residual). Decision 2
    (per-epoch draws with a fitted cross-epoch correlation) is tested against
    it. The mean's own persistence is printed next to it (a mean's size rank
    is the halo's, so it persists MORE than the truth's).
  * For the record: the S1 decline numbers on this sample.
  * The NULL row: the mean's own tier 2d (offset and width, at fixed M* and
    at fixed Mh) and tier 2e tables, saved cell by cell.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp84_layer_rebaseline/stage0_targets.py [--smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import layer_common as LC                                # noqa: E402
import predictor as P                                    # noqa: E402
from hongshao import qa                                  # noqa: E402

RULE, THIN = P.RULE, P.THIN
OUT = P.OUTDIR / "stage0_targets.npz"


def gate_cells(gate):
    """{(key, j): (offset, width_ratio)} from evaluate's size_gate_*[0]."""
    return {c: (v["offset"], v["width_ratio"]) for c, v in gate[0].items()}


def main(smoke=False):
    print(f"{RULE}\nexp84 STAGE 0 — the targets, frozen on the adopted mean\n{RULE}\n")
    recs, data, keep, lmh, pred = P.build(smoke)
    rows = np.where(keep)[0]
    n = len(rows)
    R = pred.R
    mean = pred.predict(rows=rows)
    d = data[rows]
    use = np.ones((n, 5), bool)

    # --- S5: the amplitude residual ------------------------------------------
    a = LC.amplitude_residual(mean, d, use)
    amp_sigma = LC.half_width(a)
    full = np.isfinite(a).all(axis=1)
    amp_corr = np.corrcoef(a[full].T)
    print(f"  S5 — the adopted mean's amplitude residual log10(model/truth) at 103 kpc ({n} galaxies):")
    print(f"    median per epoch [dex]:      " + " ".join(f"{v:+.4f}" for v in np.nanmedian(a, axis=0)))
    print(f"    half 16-84 width per epoch:  " + " ".join(f"{v:.4f}" for v in amp_sigma)
          + "   (v1 on the incumbent: 0.117-0.142)")
    LC.print_matrix(amp_corr, f"cross-epoch correlation ({int(full.sum())} galaxies with all five epochs):")
    print(f"    nearest-epoch mean {LC.nearest_epoch(amp_corr):+.3f} (v1: 0.67-0.80)")

    # --- S3 reference: the mean's median CoG --------------------------------
    with np.errstate(invalid="ignore", divide="ignore"):
        med_cog = np.nanmedian(np.log10(np.clip(mean, 1.0, None)), axis=0)

    # --- the persistence diagnostic --------------------------------------------
    lm_t = np.log10(np.clip(d[:, :, -1], 1.0, None))
    lm_m = np.log10(np.clip(mean[:, :, -1], 1.0, None))
    ls_t, ls_m = LC.log_sizes(d, R), LC.log_sizes(mean, R)
    print(f"\n{THIN}\n  THE PERSISTENCE DIAGNOSTIC — Spearman correlation across epochs of the size residual at fixed "
          f"M*(<148 kpc)\n  (the tier 2d width's own residual); the truth, then the mean (a mean's rank is the halo's)\n{THIN}")
    pers_t, pers_m, scatter_t = {}, {}, {}
    for k in LC.SIZE_KEYS:
        rt, rm = LC.size_residuals(ls_t[k], lm_t), LC.size_residuals(ls_m[k], lm_m)
        pers_t[k], pers_m[k] = LC.persistence(rt), LC.persistence(rm)
        scatter_t[k] = np.nanstd(rt, axis=0)
        LC.print_matrix(pers_t[k], f"{k} — TRUTH (nearest-epoch mean {LC.nearest_epoch(pers_t[k]):+.3f}; "
                        f"scatter per epoch " + " ".join(f"{s:.3f}" for s in scatter_t[k]) + ")")
        LC.print_matrix(pers_m[k], f"{k} — the MEAN (nearest-epoch mean {LC.nearest_epoch(pers_m[k]):+.3f})")

    # --- S1 for the record ---------------------------------------------------
    with np.errstate(invalid="ignore", divide="ignore"):
        dc = np.log10(d[:, 0, LC.I5] / d[:, 4, LC.I5])
        dc_m = np.log10(mean[:, 0, LC.I5] / mean[:, 4, LC.I5])
    g = np.isfinite(dc) & np.isfinite(dc_m)
    print(f"\n  S1 (for the record, not gated) — central change at 4.92 kpc, z=2 -> 0.4 ({int(g.sum())} galaxies):")
    print(f"    truth: declining {100 * np.mean(dc[g] < 0):.1f}%, overall median {np.median(dc[g]):+.4f}, "
          f"decliner median {np.median(dc[g & (dc < 0)]):+.4f}, 10-90 {np.percentile(dc[g], 10):+.3f} to {np.percentile(dc[g], 90):+.3f}")
    print(f"    mean:  declining {100 * np.mean(dc_m[g] < 0):.1f}%, overall median {np.median(dc_m[g]):+.4f}"
          + (f", decliner median {np.median(dc_m[g & (dc_m < 0)]):+.4f}" if (dc_m[g] < 0).any() else ""))

    # --- the NULL row: the mean's tier 2d / 2e ---------------------------------
    print(f"\n{THIN}\n  THE NULL ROW — the adopted mean's own size gate and mass distributions\n{THIN}")
    ev = qa.evaluate(mean, d, R, list(P.ANCHOR_Z), figdir=None, figures=False, verbose=False,
                     bin_by=lmh[rows][:, 0], halo_mass_epochs=lmh[rows])
    for which, label in (("size_gate_ms", "STELLAR mass"), ("size_gate_mh", "HALO mass")):
        qa.print_draw_gate([("mean", ev[which][0])], P.ANCHOR_Z, label)
        _, n_ok, n_cells, n_off, n_wid = ev[which]
        print(f"    -> offset {n_off} of {n_cells}, width {n_wid} of {n_cells}, both {n_ok}\n")
    qa.print_draw_cdfs([("mean", ev["cdfs"])], P.ANCHOR_Z)

    if smoke:
        print("\n  (smoke: nothing frozen)")
        return
    cells_ms, cells_mh = gate_cells(ev["size_gate_ms"]), gate_cells(ev["size_gate_mh"])
    np.savez(OUT, rows=rows, amp_sigma=amp_sigma, amp_corr=amp_corr, amp_median=np.nanmedian(a, axis=0),
             med_cog_mean=med_cog,
             **{f"persistence_truth_{k}": pers_t[k] for k in LC.SIZE_KEYS},
             **{f"persistence_mean_{k}": pers_m[k] for k in LC.SIZE_KEYS},
             **{f"size_scatter_truth_{k}": scatter_t[k] for k in LC.SIZE_KEYS},
             s1_frac=float(np.mean(dc[g] < 0)), s1_med=float(np.median(dc[g])),
             s1_med_dec=float(np.median(dc[g & (dc < 0)])),
             gate_ms_keys=np.array([f"{k}|{j}" for (k, j) in cells_ms]), gate_ms_vals=np.array(list(cells_ms.values())),
             gate_mh_keys=np.array([f"{k}|{j}" for (k, j) in cells_mh]), gate_mh_vals=np.array(list(cells_mh.values())),
             cdf_keys=np.array([f"{q}|{j}" for (q, j) in ev["cdfs"]]),
             cdf_vals=np.array([[v["ks_ratio"], v["w1_ratio"]] for v in ev["cdfs"].values()]),
             fform=np.array([h.f_form[-1] for h in recs])[rows], lmh=lmh[rows])
    print(f"\n  frozen -> {OUT.relative_to(P.ROOT)}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv[1:])
