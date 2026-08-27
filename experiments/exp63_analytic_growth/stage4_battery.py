"""exp63 Stage 4 — the STANDARD QA battery on three products, and tier 2e from draws.

  baseline    the incumbent's seven numbers on the exact integral (Stage 0's
              `exact-analytic`): what hongshao v1 is, minus its numerics.
  stage2      the two-channel mean fitted at z=0.4 only (Stage 2).
  stage2+p    the same mean with the stochastic deposition process (Stage 3):
              the FIRST draw is what the battery scores in every paired tier
              (a draw is a realisation, not a prediction — exp41's lesson, so
              read only the population tiers for this product), and the eight
              draws feed the plane figure and the tier-2e-from-draws table that
              closes open question C16.

Every epoch, both samples for the bias tables, the same galaxies for all
three. Figures in `figures/qa/qa_*_exp63_stage4_*`.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp63_analytic_growth/stage4_battery.py [--smoke] [--tables-only]
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
import process3 as P3                                    # noqa: E402
import stage0_cost as S0                                 # noqa: E402
import stage3_process as S3                              # noqa: E402
from hongshao import qa                                  # noqa: E402

RULE = "=" * 92
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
QADIR = FIGDIR / "qa"
ANCHOR_Z = E.ANCHOR_Z
MODELS = ("baseline", "stage2", "stage2+p")
TABLE_KEYS = ("kpc:M(<10)", "kpc:M(<30)", "kpc:M(<100)", "kpc:M(50-100)")
N_DRAW = 8


def main(smoke=False, tables_only=False):
    tag = "_smoke" if smoke else ""
    print(f"{RULE}\nexp63 STAGE 4 — the standard battery on three products{' (SMOKE)' if smoke else ''}\n{RULE}\n")
    recs, data, mask, lmh, spec_inc, th_inc, _ = S0.build(smoke)
    curves = E.build_curves(recs, verbose=False)
    R = F.R_GRID
    spec2 = M2.Spec2()
    f2 = np.load(OUTDIR / f"stage2_fit{tag}.npz", allow_pickle=True)
    f3 = np.load(OUTDIR / f"stage3_process{tag}.npz", allow_pickle=True)
    theta = np.asarray(f2["theta_best"], float)
    noise = P3.Noise3.from_vector(f3["noise_all"])
    th_nested = np.asarray(f2["theta_nested"], float)
    print(f"  mean: {' '.join(f'{n}={v:+.3f}' for n, v in zip(spec2.theta_names, theta))}")
    print(f"  process: {dict(zip(P3.NOISE_NAMES, noise.vector().round(3)))}")
    cogs = {"baseline": M2.predict2(spec2, th_nested, curves, R),
            "stage2": M2.predict2(spec2, theta, curves, R)}
    draws = P3.draw(spec2, theta, noise, curves, R, np.random.default_rng(4040), n_draw=N_DRAW)
    cogs["stage2+p"] = draws[0]
    good = np.isfinite(data).all(axis=(1, 2)) & (data > 0).all(axis=(1, 2))
    for m in cogs.values():
        good &= np.isfinite(m).all(axis=(1, 2)) & (m > 0).all(axis=(1, 2))
    good &= np.isfinite(draws).all(axis=(0, 2, 3)) & (draws > 0).all(axis=(0, 2, 3))
    msk = np.asarray(mask, bool)[good]
    logms = np.log10(np.clip(data[good][:, 0, -1], 1.0, None))
    print(f"  {int(good.sum())} galaxies; fit mask " + "/".join(f"{int(msk[:, j].sum())}" for j in range(5)) + "\n")

    QADIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for m in MODELS:
        print(f"\n{RULE}\nSTANDARD QA — {m}\n{RULE}")
        if m == "stage2+p":
            print("  REMINDER: the paired tiers score ONE draw, a realisation; read the population tiers.")
        out[m] = qa.evaluate(cogs[m][good], data[good], R, list(ANCHOR_Z), name=f"exp63_stage4_{m}{tag}",
                             figdir=(None if tables_only else QADIR), figures=not tables_only, verbose=True,
                             bin_by=lmh[good][:, 0], bin_label=r"logM$_h$(z=0.4)",
                             bin_by_ms=logms, ms_label=r"logM$_*$ (total)",
                             draw_cogs=(draws[:, good] if m == "stage2+p" else None))

    print(f"\n{RULE}\nSIDE BY SIDE — median relative bias [%], all / fit mask\n{RULE}")
    for key in TABLE_KEYS:
        print(f"\n  {key}")
        print(f"  {'product':<12}{'sample':<12}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
        for m in MODELS:
            t, mm = out[m]["truth"][key], out[m]["model"][key]
            print(f"  {m:<12}{'all':<12}" + "".join(f"{100 * np.nanmedian(qa.relerr(mm[:, j], t[:, j])):>9.1f}%" for j in range(5)))
            print(f"  {'':<12}{'fit mask':<12}" + "".join(
                f"{100 * np.nanmedian(qa.relerr(mm[msk[:, j], j], t[msk[:, j], j])):>9.1f}%" for j in range(5)))
    print("\n  tier 3, profile max|rel| beyond 5 kpc (median) — the draw's value is a realisation")
    for m in MODELS:
        print(f"  {m:<12}" + "".join(f"{np.nanmedian(out[m]['mr_out'][:, j]):>10.4f}" for j in range(5)))

    # tier 2e FROM DRAWS (C16), and the planes from pooled draws, per epoch
    print(f"\n{RULE}\n  TIER 2e FROM DRAWS (closes C16) — 16-84 width, model/truth; W1/floor — pooled {N_DRAW} draws\n{RULE}")
    dd = data[good]; dr = draws[:, good]; mean2 = cogs["stage2"][good]; base = cogs["baseline"][good]
    widths = {}
    for j, z in enumerate(ANCHOR_Z):
        s_d = S3.pop_stats(dr, dd, R, j=j); s_m = S3.pop_stats(mean2[None], dd, R, j=j); s_b = S3.pop_stats(base[None], dd, R, j=j)
        widths[j] = (s_b, s_m, s_d)
        print(f"  z={z}:  " + "  ".join(f"{k.split(':')[1]} {s_d[f'width {k}']:.2f} (mean {s_m[f'width {k}']:.2f}, base {s_b[f'width {k}']:.2f})"
                                     for k in S3.CDF_KEYS))
        print(f"          planes E/floor: " + ", ".join(
            f"{ky.split(':')[1]}|{kx.split(':')[1]} {s_d[f'E/floor {ky}|{kx}']:.2f} (mean {s_m[f'E/floor {ky}|{kx}']:.2f}, base {s_b[f'E/floor {ky}|{kx}']:.2f})"
            for kx, ky in S3.KPC_PLANES) + f", R50|M* {s_d['E/floor R50|M*']:.2f} (mean {s_m['E/floor R50|M*']:.2f}, base {s_b['E/floor R50|M*']:.2f})")
    np.savez(OUTDIR / f"stage4_battery{tag}.npz", names=np.array(MODELS), n_galaxies=int(good.sum()),
             **{f"mr_out_{m}": out[m]["mr_out"] for m in MODELS},
             **{f"bias_{m}_{k}_{s}": np.array([np.nanmedian(qa.relerr(out[m]["model"][k][sel[:, j], j], out[m]["truth"][k][sel[:, j], j]))
                                               for j in range(5)])
                for m in MODELS for k in TABLE_KEYS for s, sel in (("all", np.ones_like(msk)), ("fitmask", msk))},
             **{f"width_{j}_{prod}_{k.replace(':', '_').replace('<', 'lt').replace('(', '').replace(')', '').replace('-', '_')}": w[i][f"width {k}"]
                for j, w in widths.items() for i, prod in enumerate(("base", "mean", "draws")) for k in S3.CDF_KEYS})
    print(f"\n  saved {(OUTDIR / f'stage4_battery{tag}.npz').relative_to(ROOT)}")
    if not tables_only:
        print(f"  battery figures -> {QADIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main("--smoke" in sys.argv[1:], "--tables-only" in sys.argv[1:])
