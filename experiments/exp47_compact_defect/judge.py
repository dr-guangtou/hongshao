"""exp47 step 0 — THE JUDGE: one verdict function every candidate passes.

Without a single scorer, each candidate ends up judged on whichever
battery was convenient, and exp38's lesson makes that dangerous: inner
mass accuracy and the differential-deposition physics are in genuine
tension in this family, so a candidate that buys the first by breaking
the second is not an improvement. The physics tests are therefore GATES,
not tie-breakers.

What the verdict contains, for a set of model CoGs:

  planes      tier 2b centered E/floor, FULL sample and the
              compact/extended split, each against a size-matched random
              control (exp46: E/floor falls for free as n shrinks)
  sizes       tier 2d R50/R80/R90 median offsets
  inner       M*(<5 kpc) and M*(<10 kpc) bias, reported PER
              SUB-POPULATION — the point of exp47 stage 3
  tilt        the stage-1 across-ridge statistics: rho against the
              M*(<5) bias and against compactness. A real fix should
              FLATTEN the tilt; this is the statistic that says so.
  GATE_diff   differential deposition, massive tercile z0.7->0.4
              (data 0.37/0.11; adopted band 0.39-0.40 / 0.12-0.13)
  GATE_over   outskirt overshoot T1 at z=0.4 (adopted band +0.02..+0.03)

Import and call ``verdict(model_cogs, ...)``; ``print_verdict`` renders
it. Run ``demo`` for the self-check.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/exp46_highz_ridge"))

Z_GRID = (0.4, 0.7, 1.0, 1.5, 2.0)
PLANES = (("kpc:M(<30)", "kpc:M(30-50)"), ("kpc:M(<30)", "kpc:M(50-100)"))
R_COMPACT = 5.0
N_CTRL = 16

# the adopted kernel's marks, for the "did it help?" column
BAND_DIFF = (0.37, 0.11)          # data, massive tercile z0.7->0.4
BAND_OVER = (0.02, 0.03)          # adopted outskirt band, T1 [dex]


def _plane_xy(tab, kx, ky, j, mask):
    return np.column_stack([
        np.log10(np.clip(tab[kx][mask, j], 1.0, None)),
        np.log10(np.clip(tab[ky][mask, j], 1.0, None))])


def verdict(model_cogs, data_cogs, R, truth=None, model=None,
            r50_truth=None, seed=17):
    """The full verdict dict. ``truth``/``model``/``r50_truth`` may be
    passed in when several candidates share them (measure_all is the
    expensive part)."""
    from hongshao import qa
    from unification import ridge_decompose
    from compact_anatomy import spearman

    n = len(data_cogs)
    if truth is None or model is None:
        truth, model, _, _ = qa.measure_all(model_cogs, data_cogs, R)
    if r50_truth is None:
        r50_truth = qa._safe_rhalf(data_cogs, R, 0.5)
    rng = np.random.default_rng(seed)
    out = {"n": n}

    # --- planes, full and split ------------------------------------------
    for kx, ky in PLANES:
        key = f"{kx}|{ky}"
        for j in range(len(Z_GRID)):
            def score(mask):
                return qa.plane_energy(_plane_xy(truth, kx, ky, j, mask),
                                       _plane_xy(model, kx, ky, j, mask)
                                       )["energy_ratio_centered"]
            out[("plane", key, j, "all")] = score(np.ones(n, bool))
            comp = r50_truth[:, j] < R_COMPACT
            for gname, mask in (("compact", comp), ("extended", ~comp)):
                nk = int(mask.sum())
                if nk < 40:
                    out[("plane", key, j, gname)] = (np.nan, np.nan, nk)
                    continue
                ctrl = []
                for _ in range(N_CTRL):
                    m = np.zeros(n, bool)
                    m[rng.choice(n, size=nk, replace=False)] = True
                    ctrl.append(score(m))
                out[("plane", key, j, gname)] = (score(mask),
                                                 float(np.median(ctrl)), nk)

    # --- sizes -------------------------------------------------------------
    for frac in qa.SIZE_FRACTIONS:
        rt = qa._safe_rhalf(data_cogs, R, frac)
        rm = qa._safe_rhalf(model_cogs, R, frac)
        for j in range(len(Z_GRID)):
            out[("size", int(100 * frac), j)] = float(
                np.nanmedian(np.log10(rm[:, j]) - np.log10(rt[:, j])))

    # --- inner mass, per sub-population -----------------------------------
    for rad in (5.0, 10.0):
        mt = np.array([[np.interp(rad, R, data_cogs[i, j]) for j in range(5)]
                       for i in range(n)])
        mm = np.array([[np.interp(rad, R, model_cogs[i, j]) for j in range(5)]
                       for i in range(n)])
        rel = mm / np.clip(mt, 1.0, None) - 1.0
        for j in range(len(Z_GRID)):
            comp = r50_truth[:, j] < R_COMPACT
            out[("inner", int(rad), j, "all")] = float(np.nanmedian(rel[:, j]))
            out[("inner", int(rad), j, "compact")] = float(
                np.nanmedian(rel[comp, j])) if comp.sum() else np.nan
            out[("inner", int(rad), j, "extended")] = float(
                np.nanmedian(rel[~comp, j])) if (~comp).sum() else np.nan
        if rad == 5.0:
            bias5 = np.log10(np.clip(mm, 1.0, None)) - np.log10(
                np.clip(mt, 1.0, None))

    # --- the tilt ----------------------------------------------------------
    kx, ky = PLANES[0]
    c_in = np.log10(
        np.array([[np.interp(10.0, R, data_cogs[i, j]) for j in range(5)]
                  for i in range(n)])
        / np.clip(data_cogs[:, :, -1], 1.0, None))
    for j in range(len(Z_GRID)):
        tx = np.log10(np.clip(truth[kx][:, j], 1.0, None))
        ty = np.log10(np.clip(truth[ky][:, j], 1.0, None))
        mx = np.log10(np.clip(model[kx][:, j], 1.0, None))
        my = np.log10(np.clip(model[ky][:, j], 1.0, None))
        _, across, _ = ridge_decompose(tx, ty, mx, my)
        out[("tilt", j, "vs_m5")] = spearman(across, bias5[:, j])
        out[("tilt", j, "vs_cin")] = spearman(across, c_in[:, j])
        out[("tilt", j, "median_abs")] = float(np.nanmedian(np.abs(across)))
    return out


def physics_gates(model_cogs, data_cogs, logms, e):
    """The two GATES. Reuses exp35's measured differential estimator and
    the exp38 overshoot construction so the numbers stay comparable with
    every mark in the record."""
    from hongshao.profile_emulator import density_from_cog
    ed3, rows_d = e.differential(model_cogs, data_cogs, logms)
    diff = rows_d[("model", 2, 0)][:2]          # massive tercile, z0.7->0.4
    data = rows_d[("data", 2, 0)][:2]

    ls_d, mid = density_from_cog(np.log10(data_cogs[:, 0]), e.R)
    bands = ((mid >= 30.0) & (mid < 60.0), (mid >= 60.0) & (mid <= 148.0))
    c0 = model_cogs[:, 0]
    ok = np.isfinite(c0).all(1) & (c0 > 0).all(1)
    ls_m = np.full_like(ls_d, np.nan)
    ls_m[ok] = density_from_cog(np.log10(c0[ok]), e.R)[0]
    dl = ls_m - ls_d
    edges = np.quantile(logms, [0, 1 / 3, 2 / 3, 1])
    sel = ok & (logms >= edges[0]) & (logms <= edges[1] + 1e-9)
    over = tuple(float(np.nanmedian(dl[np.ix_(sel, bm)])) for bm in bands)
    return dict(diff_model=tuple(float(v) for v in diff),
                diff_data=tuple(float(v) for v in data), over_T1=over)


def print_verdict(v, label, gates=None):
    print(f"\n=== VERDICT [{label}]  (n={v['n']}) ===")
    for kx, ky in PLANES:
        key = f"{kx}|{ky}"
        print(f"  tier 2b {kx} vs {ky} — centered E/floor")
        print(f"    {'z':>5}{'all':>7} | {'compact (ctrl, n)':>26}"
              f" | {'extended (ctrl, n)':>26}")
        for j, z in enumerate(Z_GRID):
            cells = []
            for g in ("compact", "extended"):
                s, c, nk = v[("plane", key, j, g)]
                cells.append("(n too small)".rjust(26) if not np.isfinite(s)
                             else f"{s:.1f}  ({c:.1f}, {nk})".rjust(26))
            print(f"    {z:5.1f}{v[('plane', key, j, 'all')]:7.1f} | "
                  f"{cells[0]} | {cells[1]}")
    print("  M*(<5 kpc) bias [%] by sub-population")
    print(f"    {'z':>5}{'all':>9}{'compact':>10}{'extended':>10}")
    for j, z in enumerate(Z_GRID):
        print(f"    {z:5.1f}" + "".join(
            f"{100 * v[('inner', 5, j, g)]:9.1f}%"
            for g in ("all", "compact", "extended")))
    print("  median dlog R [model-truth] and the stage-1 TILT")
    print(f"    {'z':>5}{'R50':>8}{'R80':>8}{'R90':>8}"
          f"{'rho(tilt,M5)':>14}{'rho(tilt,c_in)':>16}")
    for j, z in enumerate(Z_GRID):
        print(f"    {z:5.1f}" + "".join(f"{v[('size', f, j)]:+8.3f}"
                                        for f in (50, 80, 90))
              + f"{v[('tilt', j, 'vs_m5')]:+14.2f}"
              + f"{v[('tilt', j, 'vs_cin')]:+16.2f}")
    if gates:
        d, dm = gates["diff_data"], gates["diff_model"]
        o = gates["over_T1"]
        ok_d = 0.30 <= dm[0] <= 0.45
        ok_o = abs(o[0]) <= 0.05 and abs(o[1]) <= 0.06
        print(f"  GATE differential (massive z0.7->0.4): data "
              f"{d[0]:.2f}/{d[1]:.2f} -> model {dm[0]:.2f}/{dm[1]:.2f}  "
              f"[{'PASS' if ok_d else 'FAIL'}]")
        print(f"  GATE outskirt T1 (z=0.4, 30-60/60-148): "
              f"{o[0]:+.3f}/{o[1]:+.3f}  [{'PASS' if ok_o else 'STRAINED'}]")


def demo():
    from hongshao import qa
    rng = np.random.default_rng(0)
    from hongshao import qa
    n, nz, nr = 800, 5, 24
    R = np.geomspace(2.0, 148.0, nr)
    # truth: a size-varied population; model: identical EXCEPT a central
    # deficit applied only to compact galaxies — the exact defect exp47 is
    # chasing. The judge must localize it to the compact column and leave
    # the extended one clean. The scale parameter is NOT the half-mass
    # radius (R50 ~ 2.2 x scale here), so the split is taken from the CoG
    # exactly as the real judge does.
    scale = rng.uniform(1.0, 10.0, n)
    truth_cogs = np.stack([np.stack([1e11 * (1.0 - (1.0 + (R / s) ** 2)
                                             ** -0.4) for _ in range(nz)])
                           for s in scale])
    r50 = qa._safe_rhalf(truth_cogs, R, 0.5)
    comp = r50[:, 0] < R_COMPACT
    assert comp.sum() >= 40, f"fixture too small to test: {comp.sum()}"
    # The defect must REDISTRIBUTE mass, not delete it: scaling the inner
    # CoG points alone leaves M(<30) unchanged and nothing moves on the
    # plane. So compact galaxies get an over-predicted SIZE at the same
    # total — which is exactly what exp45 measured (dlog R50 vs the
    # M*(<5) bias, rho -0.89 to -0.97).
    model_cogs = truth_cogs.copy()
    for i in np.flatnonzero(comp):
        s = 1.25 * scale[i]
        prof = 1e11 * (1.0 - (1.0 + (R / s) ** 2) ** -0.4)
        model_cogs[i] = prof * (truth_cogs[i, 0, -1] / prof[-1])
    v = verdict(model_cogs, truth_cogs, R)
    b_c = v[("inner", 5, 0, "compact")]
    b_e = v[("inner", 5, 0, "extended")]
    assert b_c < -0.10, b_c
    assert abs(b_e) < 1e-6, b_e
    s_c = v[("plane", f"{PLANES[0][0]}|{PLANES[0][1]}", 0, "compact")][0]
    s_e = v[("plane", f"{PLANES[0][0]}|{PLANES[0][1]}", 0, "extended")][0]
    assert s_c > s_e, (s_c, s_e)
    print(f"demo OK — judge localizes a compact-only central deficit: "
          f"M*(<5) bias {100 * b_c:.0f}% compact vs {100 * b_e:.0f}% "
          f"extended; plane {s_c:.1f} vs {s_e:.1f}")


if __name__ == "__main__":
    if (sys.argv[1] if len(sys.argv) > 1 else "demo") == "demo":
        demo()
    else:
        raise SystemExit(__doc__)
