"""exp58 P-B — CHARACTERISE the z=2 failure before anyone models against it.

Every enrichment since exp54 Stage 3.5 aimed at the central defect; the user's
observation is that z=2 is *clearly* wrong and nobody has asked in what way.
This script decomposes the incumbent's z=2 error along four axes, fitting
nothing:

  1. AMPLITUDE OR SHAPE. The model's error at z=2 is split into the total-mass
     part (log10 of model/measured M*(<103 kpc), the loss's own amplitude
     anchor) and the radial-shape part (the same profiles with that amplitude
     divided out). If the pinned profiles are fine at z=2, the z=2 problem is
     the amplitude — the absolute deposition law — and no radial mechanism can
     fix it.
  2. WHERE IN RADIUS. Median signed error per radius per epoch, raw and
     amplitude-pinned.
  3. WHO. The z=2 amplitude residual conditioned on everything the model can
     see (halo mass, formation time, concentration excess) and on two things
     it cannot (the halo's FUTURE growth to z=0.4 — the progenitor-selection
     channel — and the measured compactness / later central decline —
     stellar-information channels).
  4. WHAT THE LOSS SEES. Per-epoch score_A / score_F of the incumbent, and the
     per-epoch mask counts, so "the fit ignores z=2" is measured rather than
     suspected.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp58_reassessment/pb_z2_diagnosis.py [--smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP54 = ROOT / "experiments/exp54_unpinned_amplitude"
EXP57 = ROOT / "experiments/exp57_expansion_term"
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink", EXP54, EXP57):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import scoreboard as SB                                  # noqa: E402
import stage0_cost as S0                                 # noqa: E402
import stage4_target as S4                               # noqa: E402

ANCHOR_Z = (0.4, 0.7, 1.0, 1.5, 2.0)
#: radii reported in the radius-resolved tables [kpc, nearest grid point used]
R_SHOW = (2.0, 3.72, 10.25, 22.97, 52.3, 103.45, 148.22)
RULE = "=" * 96
OUT = HERE / "outputs" / "pb_z2_diagnosis.npz"


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def quartile_medians(x, y):
    """median of y in quartiles of x; for a 0/1 flag, the two group medians."""
    m = np.isfinite(x) & np.isfinite(y)
    if len(np.unique(x[m])) <= 2:
        lo, hi = (float(np.median(y[m][x[m] == v]))
                  for v in np.unique(x[m])[:2])
        return [lo, np.nan, np.nan, hi]
    q = np.nanpercentile(x[m], [25, 50, 75])
    edges = [-np.inf, *q, np.inf]
    return [float(np.median(y[m][(x[m] > lo) & (x[m] <= hi)]))
            for lo, hi in zip(edges[:-1], edges[1:])]


def partial_corr(x, y, ctrl):
    """Pearson r of x with y after regressing BOTH on ctrl (one controller)."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(ctrl)
    if m.sum() < 30:
        return np.nan
    D = np.column_stack([np.ones(m.sum()), ctrl[m]])
    rx = x[m] - D @ np.linalg.lstsq(D, x[m], rcond=None)[0]
    ry = y[m] - D @ np.linalg.lstsq(D, y[m], rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def main(smoke=False):
    print(f"{RULE}\nexp58 P-B — what, exactly, is wrong at z=2 "
          f"(incumbent, nothing fitted)\n{RULE}\n")
    recs, data, mask, lmh, base, theta, slow = S0.build(smoke)
    R = F.R_GRID
    i100 = F.I100
    pred = slow.predict(theta)
    n = len(recs)
    finite = np.isfinite(pred).all(axis=2) & (pred[:, :, i100] > 0)
    use = np.asarray(mask, bool) & finite & (data[:, :, i100] > 0)

    # ------------------------------------------------------------- axis 4 --
    print(f"  AXIS 4 — what the loss sees, per epoch (incumbent)")
    sa, sf = slow.per_epoch(theta)
    print(f"  {'':<26}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
    print(f"  {'score_A (amplitude)':<26}" + "".join(f"{v:>10.4f}" for v in sa))
    print(f"  {'score_F (shape)':<26}" + "".join(f"{v:>10.4f}" for v in sf))
    print(f"  {'galaxy-epochs in the fit':<26}"
          + "".join(f"{int(use[:, j].sum()):>10d}" for j in range(5)))
    print(f"  {'SIGMA_A benchmark [dex]':<26}"
          + "".join(f"{float(F.SIGMA_A[j]):>10.4f}" for j in range(5)))
    print(f"\n  score_F is the MEAN over epochs, so each epoch carries 1/5 of "
          f"the shape loss;\n  score_A is an RMS over all galaxy-epochs with a "
          f"PER-EPOCH benchmark sigma.")

    # ------------------------------------------------------------- axis 1 --
    a_res = np.where(use, np.log10(np.clip(pred[:, :, i100], 1.0, None))
                     - np.log10(np.clip(data[:, :, i100], 1.0, None)), np.nan)
    print(f"\n  AXIS 1 — the AMPLITUDE part: log10[model/measured M*(<"
          f"{R[i100]:.0f} kpc)], fit sample")
    print(f"  {'':<26}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
    med = np.nanmedian(a_res, axis=0)
    print(f"  {'median [dex]':<26}" + "".join(f"{v:>+10.4f}" for v in med))
    print(f"  {'16th pct [dex]':<26}" + "".join(
        f"{np.nanpercentile(a_res[:, j], 16):>+10.4f}" for j in range(5)))
    print(f"  {'84th pct [dex]':<26}" + "".join(
        f"{np.nanpercentile(a_res[:, j], 84):>+10.4f}" for j in range(5)))
    print(f"  {'scatter (half 16-84)':<26}" + "".join(
        f"{(np.nanpercentile(a_res[:, j], 84) - np.nanpercentile(a_res[:, j], 16)) / 2:>10.4f}"
        for j in range(5)))

    # ------------------------------------------------------------- axis 2 --
    idx = [int(np.argmin(np.abs(R - r))) for r in R_SHOW]
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = np.log10(pred / np.clip(data, 1e-300, None))
        pin = (pred / pred[:, :, i100][:, :, None])
        pin = np.log10(pin / (data / data[:, :, i100][:, :, None]))
    raw = np.where(use[:, :, None], raw, np.nan)
    pin = np.where(use[:, :, None], pin, np.nan)
    for lab, arr in (("RAW (amplitude included)", raw),
                     ("AMPLITUDE-PINNED (shape only)", pin)):
        print(f"\n  AXIS 2 — {lab}: median log10(model/measured) per radius")
        print(f"  {'radius':<10}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
        for r, i in zip(R_SHOW, idx):
            print(f"  {r:>6.1f} kpc"
                  + "".join(f"{np.nanmedian(arr[:, j, i]):>+10.4f}"
                            for j in range(5)))

    # ------------------------------------------------------------- axis 3 --
    print(f"\n  AXIS 3 — WHO carries the z=2 amplitude error "
          f"(residual a = log10 model/measured at z=2)")
    a2 = a_res[:, 4]
    d = SB.load()
    idx_of = {int(r): i for i, r in enumerate(d["rows"])}
    sel = np.array([idx_of.get(int(h.row), -1) for h in recs])
    ok_d = sel >= 0
    def dcol(name, k=None):
        out = np.full(n, np.nan)
        v = d[name][sel[ok_d]] if k is None else d[name][sel[ok_d], k]
        out[ok_d] = v
        return out

    fform2 = np.full(n, np.nan)
    for i, h in enumerate(recs):
        w = np.where(h.epoch_mask[4])[0]
        if len(w):
            fform2[i] = h.f_form[w[-1]]
    growth = lmh[:, 0] - lmh[:, 4]                  # future halo growth [dex]
    fell, okc = S4.decline_flag(data)
    with np.errstate(invalid="ignore", divide="ignore"):
        cstar2 = data[:, 4, idx[2]] / data[:, 4, i100]   # measured M(<10)/M(<103)
    feats = (
        ("log10 Mh at z=2 (DiffMAH)", lmh[:, 4], "model input"),
        ("formation time t_half/t at z=2", fform2, "model input"),
        ("concentration excess at z=2", dcol("c_exc_k", 4), "model input"),
        ("fz2 (Mh fraction in place by z=2)", dcol("fz2"), "model input"),
        ("FUTURE growth logMh(0.4)-logMh(2)", growth, "NOT available to model"),
        ("measured M(<10)/M(<103) at z=2", cstar2, "stellar - diagnostic only"),
        ("centre declines z=2 to 0.4 (0/1)", np.where(okc, fell.astype(float),
                                                      np.nan),
         "stellar - diagnostic only"),
    )
    def cond_table(target):
        print(f"  {'feature':<36}{'r':>7}{'r|Mh':>7}   median residual in "
              f"its quartiles (Q1..Q4; two groups for a flag) [dex]")
        for name, x, kind in feats:
            r = corr(x, target)
            rp = partial_corr(x, target, lmh[:, 4])
            qm = quartile_medians(x, target) if np.isfinite(r) else [np.nan] * 4
            print(f"  {name:<36}{r:>+7.3f}{rp:>+7.3f}   "
                  + " ".join((f"{v:>+8.4f}" if np.isfinite(v) else f"{'.':>8}")
                             for v in qm) + f"   ({kind})")

    cond_table(a2)

    # the same conditioning for the z=2 SHAPE error at 10 kpc, pinned
    print(f"\n  ... and WHO carries the z=2 pinned SHAPE error at 10 kpc")
    s2r = pin[:, 4, idx[2]]
    cond_table(s2r)

    if not smoke:
        OUT.parent.mkdir(exist_ok=True)
        np.savez_compressed(OUT, a_res=a_res, raw_med=np.nanmedian(raw, 0),
                            pin_med=np.nanmedian(pin, 0), score_A=sa,
                            score_F=sf, growth=growth, fform2=fform2,
                            lmh=lmh, use=use)
        print(f"\n  saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv[1:])
