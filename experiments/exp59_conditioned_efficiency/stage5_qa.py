"""exp59 Stage 5 — the project's STANDARD QA battery on the Stage 2 candidate.

`hongshao.qa.evaluate` — the battery every promoted model in this programme has
been shown on — for three models on the same galaxies, so the adoption
decision can be made from the same figures every other candidate was judged
on:

  * `incumbent`      exp54's adopted model — the REFERENCE.
  * `X3-pinned`      THE STAGE 2 CANDIDATE: core-only expansion refit with the
                     size law frozen at the incumbent's (A = +1.50,
                     Rc = 1.98 kpc). Every mechanism gate passed; the unpinned
                     version's outskirt drift eliminated.
  * `X3-unpinned`    exp57's original small-core fit, for comparison — what
                     the extra size-law freedom bought (0.15 shape points) and
                     cost (the +0.075 dex outskirt drift).

**Standing caveat, stated whenever this battery is quoted** (exp57 P-C): the
battery's headline tiers cannot see a redistribution of a few per cent of the
mass inside 10 kpc — the improvement shows in the inner-aperture bias table
and in the gates, not in tier 3 or the scatters.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp59_conditioned_efficiency/stage5_qa.py [--tables-only]
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

import expand as X                                       # noqa: E402
import fit as F                                          # noqa: E402
import stage0_cost as S0                                 # noqa: E402
from hongshao import qa                                  # noqa: E402

QADIR = HERE / "figures" / "qa"
FITDIR = EXP57 / "outputs" / "stage3_fits"
ANCHOR_Z = (0.4, 0.7, 1.0, 1.5, 2.0)
#: (label in the report, fit file or None for the incumbent)
MODELS = (("incumbent", None),
          ("X3-pinned", "gompertz_log-E2-S2+X3+frozen_log_f0_b#pinned_size"),
          ("X3-unpinned", "gompertz_log-E2-S2+X3#smallcore"))
RULE = "=" * 96


def main(tables_only=False):
    print(f"{RULE}\nexp59 — the STANDARD QA battery on the Stage 2 candidate"
          f"\n{RULE}\n")
    recs, data, mask, lmh, base, theta, slow = S0.build(False)
    R = F.R_GRID

    cogs = {}
    for name, lab in MODELS:
        if lab is None:
            sp, th = X.XSpec(base, ""), theta
        else:
            z = np.load(FITDIR / f"{lab}.npz", allow_pickle=True)
            sp = X.XSpec(base, str(z["law"]))
            th = np.asarray(z["theta"], float)
        pr = X.ExpandingProblem(sp, recs, data, mask, epochs=(0, 1, 2, 3, 4))
        cogs[name] = pr.predict(th)
        print(f"  {name:<14} {sp.label:<40} "
              + (", ".join(f"{n}={v:+.4f}" for n, v in
                           zip(sp.x_names, th[base.n_theta:]))
                 or "(no expansion)"))

    good = np.isfinite(data).all(axis=(1, 2)) & (data > 0).all(axis=(1, 2))
    for m in cogs.values():
        good &= np.isfinite(m).all(axis=(1, 2)) & (m > 0).all(axis=(1, 2))
    print(f"\n  {int(good.sum())} of {len(recs)} galaxies representable by "
          f"all three models with finite positive\n  measured profiles at "
          f"every epoch; the same set is used for each report.")
    logms = np.log10(np.clip(data[good][:, 0, -1], 1.0, None))

    QADIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, _ in MODELS:
        print(f"\n{RULE}\nSTANDARD QA — {name}\n{RULE}")
        out[name] = qa.evaluate(
            cogs[name][good], data[good], R, list(ANCHOR_Z),
            name=f"exp59_{name}", figdir=(None if tables_only else QADIR),
            figures=not tables_only, verbose=True,
            bin_by=lmh[good][:, 0], bin_label=r"logM$_h$(z=0.4)",
            bin_by_ms=logms, ms_label=r"logM$_*$ (total)")

    print(f"\n{RULE}\nSIDE BY SIDE, on the same {int(good.sum())} galaxies"
          f"\n{RULE}")
    print(f"  WHERE THE CANDIDATE'S CHANGE LIVES: the inner-aperture bias "
          f"table. The headline\n  tiers below it cannot see the change "
          f"(exp57 P-C) and are printed to show that.")
    for key in ("kpc:M(<10)", "kpc:M(<30)", "kpc:M(<100)"):
        print(f"\n  {key} — median relative bias, closer to 0 is better")
        print(f"  {'model':<16}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
        for name, _ in MODELS:
            t, m = out[name]["truth"][key], out[name]["model"][key]
            print(f"  {name:<16}" + "".join(
                f"{100 * np.nanmedian(qa.relerr(m[:, j], t[:, j])):>9.1f}%"
                for j in range(5)))
    print(f"\n  tier 3, profile max|rel| beyond 5 kpc (median) — LOWER is "
          f"better")
    print(f"  {'model':<16}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
    for name, _ in MODELS:
        print(f"  {name:<16}"
              + "".join(f"{np.nanmedian(out[name]['mr_out'][:, j]):>10.4f}"
                        for j in range(5)))
    k = "kpc:M(<100)"
    print(f"\n  M*(<100 kpc) scatter [dex] — LOWER is better")
    print(f"  {'model':<16}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
    for name, _ in MODELS:
        print(f"  {name:<16}" + "".join(
            f"{qa.dex_scatter(out[name]['model'][k][:, j], out[name]['truth'][k][:, j]):>10.4f}"
            for j in range(5)))

    print(f"\n  figures -> {QADIR.relative_to(ROOT)}")
    np.savez(HERE / "outputs" / "stage5_qa.npz",
             names=np.array([m[0] for m in MODELS]),
             n_galaxies=int(good.sum()),
             **{f"mr_out_{m[0]}": out[m[0]]["mr_out"] for m in MODELS})


if __name__ == "__main__":
    main("--tables-only" in sys.argv)
