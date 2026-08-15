"""exp47 — the STANDARD qa figure sets for the adopted kernel and the
step-2 candidates, so the claimed improvement can actually be looked at.

Produces `hongshao.qa.evaluate`'s full layout (kpc and Re aperture/annulus
masses truth-vs-model per epoch, halo-mass-binned residuals, the 2-D
observational planes with energy/floor scores, the cross-epoch growth
planes, the mass-size planes, the best/worst gallery) for:

  exp47_adopted   the current adopted kernel (12 params, no core channel)
  exp47_B         core channel + DiffMAH-shape conditioning, balanced loss
  exp47_C         core channel, balanced loss, no shape conditioning

Run: PYTHONPATH=. uv run python experiments/exp47_compact_defect/\
qa_figures.py {run} [--dev]
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/exp46_highz_ridge"))

FIGDIR = HERE / "figures"
OUTDIR = HERE / "outputs"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
ANCHOR_Z = (0.4, 0.7, 1.0, 1.5, 2.0)


def cmd_run(dev=False):
    from hongshao import qa
    import core_shape as cs
    from unification import adopted_model_cogs, _w_init as u_init, _W as uW

    rows = np.load(POP_NPZ)["dev100"] if dev else None
    FIGDIR.mkdir(exist_ok=True)

    # --- the adopted kernel, through its own harness ----------------------
    u_init(rows)
    gals, R = uW["gals"], uW["e"].R
    data_cogs = np.stack([g["data"] for g in gals])
    print(f"exp47 qa figures (n={len(gals)}{', DEV' if dev else ''})",
          flush=True)
    cogs_adopted = adopted_model_cogs(data_cogs, rows, dev)
    qa.evaluate(cogs_adopted, data_cogs, R, ANCHOR_Z,
                name="exp47_adopted", figdir=str(FIGDIR))

    # --- the step-2 candidates -------------------------------------------
    cs._w_init(rows)
    gals, R = cs._W["gals"], cs._W["e"].R
    data_cogs = np.stack([g["data"] for g in gals])
    d = np.load(OUTDIR / ("core_shape_dev.npz" if dev else "core_shape.npz"))
    for key in ("B", "C"):
        if f"theta_{key}" not in d.files:
            continue
        cogs = cs._build(d[f"theta_{key}"], len(gals), len(R), rows)
        qa.evaluate(cogs, data_cogs, R, ANCHOR_Z,
                    name=f"exp47_{key}", figdir=str(FIGDIR))
    print(f"\n  figures in {FIGDIR}")


if __name__ == "__main__":
    if (sys.argv[1] if len(sys.argv) > 1 else "run") == "run":
        cmd_run("--dev" in sys.argv)
    else:
        raise SystemExit(__doc__)
