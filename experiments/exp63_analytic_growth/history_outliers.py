"""exp63 — galaxies whose measured stellar history is not physical (2026-08-30).

The user's reading of the QA worst cases: some objects show a dramatic CoG
evolution that is very likely unphysical (a wrong progenitor in the
cross-match), and the existing filter — `selection.sane_history_mask`, which
drops an epoch only when M*(<100 kpc) is more than 3 dex below the galaxy's own
z=0.4 value — does not catch them. Two examples from the joint fit's worst
cases: row 181 (the known broken cross-match, still inside every whole-sample
QA table) and row 2372 (M*(<148) 2.1 dex below its z=0.4 value at z=2, so it
passes the 3 dex rule, then grows 1.4 dex between z=2 and z=1.5).

Three criteria, each a galaxy-epoch flag; a galaxy is a CANDIDATE if any epoch
is flagged:

  A  off the population's stellar-mass--halo-mass relation at that epoch:
     |log M*(<100) - median relation at that log Mh| > SHMR_TOL dex (the
     relation is the running median in halo-mass bins at that epoch, on the
     galaxies with a valid halo mass; the population's 16-84 half-width is
     0.12-0.15 dex, so 0.5 dex is a ~4-sigma outlier and, for a main
     progenitor, a wrong object; a strict 0.75 dex set is saved as well);
  B  an adjacent-epoch JUMP: log M*(<100) grows by more than JUMP_DEX between
     two consecutive epochs (the sample's 99th percentile is 0.68 dex);
  C  an adjacent-epoch DROP: log M*(<30) falls by more than DROP_DEX between
     two consecutive epochs (a main progenitor's centre does not lose a factor
     of two of its stars in one step; the sample's 1st percentile is -0.06).

Outputs: the census (how many galaxy-epochs and galaxies each criterion
flags, their overlap, which fit masks they sit in), a figure of the
candidates' CoGs at the five epochs with their MAH and M*/Mh track, and the
weight the candidates carry in the joint fit's per-epoch loss and in the gate
(evaluated, not refitted). Writes `outputs/history_outliers.npz` with the
per-galaxy-epoch flags for the fitting path to use.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp63_analytic_growth/history_outliers.py [--smoke] [--tag _joint_kpc_free]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term", HERE):
    sys.path.insert(0, str(p))

import engine as E                                       # noqa: E402
import fit as F                                          # noqa: E402
import model2 as M2                                      # noqa: E402
import single_epoch as SE                                # noqa: E402
import stage2_fit as S2                                  # noqa: E402

RULE = "=" * 96
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
R = F.R_GRID
Z = E.ANCHOR_Z
I30 = int(np.argmin(np.abs(R - 30.0)))
import selection as SEL                                  # noqa: E402  (the shared rule lives there)

SHMR_TOL, JUMP_DEX, DROP_DEX = SEL.SHMR_TOL_DEX, SEL.JUMP_DEX, SEL.DROP_DEX
SHMR_TOL_STRICT = 0.75
running_median_relation = SEL.running_median_relation
flags = SEL.stellar_history_flags


def candidate_figure(rows, data, lmh, curves, resid, why, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hongshao.plotting import set_style, save_fig
    from hongshao.qa import _zcolors
    set_style()
    rows = list(rows)[:10]
    nr = len(rows)
    fig, axes = plt.subplots(nr, 3, figsize=(14, 2.9 * nr), squeeze=False)
    cols = _zcolors(5)
    lt = np.linspace(np.log10(E.T_START), np.log10(E.T_ANCHOR[0]), 200)
    for a, i in zip(axes, rows):
        for k in range(5):
            a[0].plot(R, np.log10(np.clip(data[i, k], 1, None)), "-", color=cols[k], lw=1.6, label=f"z={Z[k]}")
        a[0].set_xscale("log"); a[0].set_ylabel(r"$\log M_*(<R)$"); a[0].set_title(f"row {i}: measured CoGs", fontsize=9)
        lm = E.log_mah(lt, curves[i])
        a[1].plot(10.0 ** lt, lm, "k-", lw=1.5, label="DiffMAH halo mass")
        a[1].plot(E.T_ANCHOR, lmh[i], "ko", ms=4)
        a[1].plot(E.T_ANCHOR, np.log10(np.clip(data[i, :, F.I100], 1, None)), "s-", color="#D55E00", ms=5, label=r"measured $M_*(<100)$")
        a[1].set_xlabel("cosmic time [Gyr]"); a[1].set_ylabel(r"$\log M$"); a[1].set_title("halo growth curve and the stellar history", fontsize=9)
        a[2].plot(Z, resid[i], "o-", color="#0072B2")
        a[2].axhline(0, color="k", lw=0.8); a[2].axhspan(-SHMR_TOL, SHMR_TOL, color="0.9")
        a[2].set_xlabel("z"); a[2].set_ylabel("off the M*-Mh relation [dex]")
        a[2].set_title(why[i], fontsize=8)
    axes[0, 0].legend(fontsize=7); axes[0, 1].legend(fontsize=7)
    fig.suptitle("exp63 — candidates for an unphysical measured history (each row one galaxy)", fontsize=11)
    fig.tight_layout()
    out = FIGDIR / name
    save_fig(fig, out)
    return out.with_suffix(".png")


def main(smoke, tag):
    sample = SE.Sample(smoke)
    data, lmh_all, mask, curves = sample.data, None, sample.mask, sample.curves
    # halo masses at the five epochs
    _, _, _, lmh, _, _, _ = __import__("stage0_cost").build(smoke)
    lmh = np.asarray(lmh, float)
    A, B, C, resid = flags(data, lmh)
    anyf = A | B | C
    cand = anyf.any(1)
    print(f"{RULE}\nHISTORY OUTLIERS — {len(data)} galaxies\n{RULE}")
    print(f"  A (|SHMR residual| > {SHMR_TOL} dex): {A.sum()} galaxy-epochs, {A.any(1).sum()} galaxies; per epoch "
          + " / ".join(f"{A[:, k].sum()}" for k in range(5)))
    print(f"  B (adjacent jump > {JUMP_DEX} dex at M(<100)): {B.sum()} galaxy-epochs, {B.any(1).sum()} galaxies")
    print(f"  C (adjacent drop > {DROP_DEX} dex at M(<30)): {C.sum()} galaxy-epochs, {C.any(1).sum()} galaxies")
    print(f"  candidates (any): {cand.sum()} galaxies; in the fit masks per epoch: "
          + " / ".join(f"{(cand & mask[:, k]).sum()}" for k in range(5))
          + "; flagged galaxy-epochs INSIDE the fit mask at that epoch: "
          + " / ".join(f"{(anyf[:, k] & mask[:, k]).sum()}" for k in range(5)))
    print(f"  SHMR residual scatter (16-84 half-width) per epoch: "
          + " / ".join(f"{0.5 * np.subtract(*np.nanpercentile(resid[:, k], [84, 16])):.3f}" for k in range(5)) + " dex")
    rows = np.where(cand)[0]
    why = {}
    for i in rows:
        parts = []
        if A[i].any(): parts.append("A@z=" + ",".join(f"{Z[k]}" for k in range(5) if A[i, k]) + f" (resid {', '.join(f'{resid[i, k]:+.2f}' for k in range(5) if A[i, k])})")
        if B[i].any(): parts.append("B@z=" + ",".join(f"{Z[k]}" for k in range(5) if B[i, k]))
        if C[i].any(): parts.append("C@z=" + ",".join(f"{Z[k]}" for k in range(5) if C[i, k]))
        why[i] = "; ".join(parts)
    print("\n  candidates: row, logMh(z=0.4..2), log M*(<100)(z=0.4..2), why")
    for i in rows:
        print(f"    {i:>5}  Mh " + " ".join(f"{v:5.2f}" for v in lmh[i]) + "  M* "
              + " ".join(f"{v:5.2f}" for v in np.log10(np.clip(data[i, :, F.I100], 1, None))) + f"  {why[i]}")

    # their weight in the joint fit and the gate, evaluated (no refit)
    spec, th, _ = SE.load_fit(tag)
    pr = S2.JointProblem2(spec, curves, data, mask, lmh, R, sample.th_inc, binned=True)
    full = pr.loss(th)
    print(f"\n  joint fit {tag}: loss {full:.4f} on the fit masks")
    keep = ~cand
    mask_k = mask & keep[:, None]
    pr2 = S2.JointProblem2(spec, curves, data, mask_k, lmh, R, sample.th_inc, binned=True)
    print(f"  the same theta with the {cand.sum()} candidates removed from every mask: loss {pr2.loss(th):.4f} "
          f"(the null references are recomputed on the reduced masks)")
    for k in range(5):
        s_full = pr.per_epoch(th)[k]; s_red = pr2.per_epoch(th)[k]
        print(f"    z={Z[k]}: A/F/S/B {s_full[0]:.3f}/{s_full[1]:.3f}/{s_full[2]:.3f}/{s_full[4]:.3f} -> "
              f"{s_red[0]:.3f}/{s_red[1]:.3f}/{s_red[2]:.3f}/{s_red[4]:.3f}")
    path = candidate_figure(rows, data, lmh, curves, resid, why, f"exp63_history_outliers{'_smoke' if smoke else ''}")
    print(f"\n  wrote {path.relative_to(ROOT)}")
    strict = ((np.abs(resid) > SHMR_TOL_STRICT) | B | C).any(1)
    print(f"  strict set (|SHMR residual| > {SHMR_TOL_STRICT} dex, or B, or C): {strict.sum()} galaxies")
    np.savez(OUTDIR / f"history_outliers{'_smoke' if smoke else ''}.npz", A=A, B=B, C=C, resid=resid,
             candidate=cand, candidate_strict=strict, rows=rows, thresholds=np.array([SHMR_TOL, JUMP_DEX, DROP_DEX]))


if __name__ == "__main__":
    args = sys.argv[1:]
    main("--smoke" in args, args[args.index("--tag") + 1] if "--tag" in args else "_joint_kpc_free")
