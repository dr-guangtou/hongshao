"""exp87 Stage 3 — the fixed method ladder L0-L3 and L5, no adaptivity.

Blocks (each a list of cells; a cell already on disk is skipped):

  mh-parent      halo mass at z = 0.4 on the complete parent: the full ladder
  mh-sens        the same reference methods on the curated samples at z = 0.4
                 (input-only cuts; the 2356 fitting sample): the sensitivity
  mh-highz       halo mass at z = 0.7 .. 2, two labelled populations:
                 progenitors as-is (no cut) and complete above c_k (cut c_k)
  conc           log concentration: parent at z = 0.4 (full-ish ladder), the
                 progenitors as-is at higher z; and the ORACLE extension — the
                 curve of growth plus the true halo mass as inputs
  form           formation time t50 on the parent, with the same oracle extension

Run (detached; `--block NAME` to run one; `--smoke` thins the folds):
    PYTHONPATH=. uv run python -u experiments/exp87_inverse_halo/stage3_ladder.py [--block mh-parent] [--smoke]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C                                       # noqa: E402
import generative as G                                   # noqa: E402
import harness as HN                                     # noqa: E402
import methods as M                                      # noqa: E402

RULE = "=" * 100


def spec(sample, epoch, population, target, feature, extra=None):
    return dict(sample=sample, epoch=epoch, population=population, target=target, feature=feature, extra=extra)


def reference_cells(sample, epoch, population, target):
    """L0: climatology, the scalar stellar-mass relation, the fixed aperture list."""
    cells = [(spec(sample, epoch, population, target, "mtot"), M.Climatology(target)),
             (spec(sample, epoch, population, target, "mtot"), M.scalar_method("line"))]
    for name in C.APERTURES:
        cells.append((spec(sample, epoch, population, target, name), M.scalar_method("line")))
    return cells


def full_ladder(sample, epoch, population, target, flexible=True, generative=True):
    cells = reference_cells(sample, epoch, population, target)
    cells.append((spec(sample, epoch, population, target, "mass_size"), M.DirectLinear("linear", "L1")))
    raw = spec(sample, epoch, population, target, "raw24")
    cells += [(raw, m) for m in M.linear_methods(24)]
    cells += [(raw, m) for m in M.nonlinear_direct_methods(24)]
    cells += [(raw, m) for m in M.augmented_methods()]
    if flexible:
        cells += [(raw, m) for m in M.flexible_methods(24)]
    if generative:
        cells += [(raw, m) for m in G.generative_methods(epoch, target, box=(population == "parent"))]
    return cells


def reduced_ladder(sample, epoch, population, target):
    """The reference rows plus one representative per rung, for the secondary populations."""
    cells = reference_cells(sample, epoch, population, target)[:2]
    cells.append((spec(sample, epoch, population, target, "M(50-100)"), M.scalar_method("line")))
    cells.append((spec(sample, epoch, population, target, "mass_size"), M.DirectLinear("linear", "L1")))
    raw = spec(sample, epoch, population, target, "raw24")
    cells += [(raw, M.DirectLinear("ridge", "L1", ridge_grid=M.RIDGE_GRID)), (raw, M.DirectLinear("pca4", "L1", transform=M._pca(4))),
              (raw, M.DirectLinear("pca4-poly2", "L2", transform=M._pca_poly(4), ridge_grid=M.RIDGE_GRID)),
              (raw, M.LearnerHead("gbm", "L3", M._gbm)),
              (raw, G.GenerativeInverse("gen-k4", k=4, epoch=epoch, target=target))]
    cells += [(raw, m) for m in M.augmented_methods()]
    return cells


def oracle_cells(sample, epoch, population, target):
    """The oracle extension: what the curve of growth adds once the halo mass is known."""
    return [(spec(sample, epoch, population, target, "none", extra="mh"), M.DirectLinear("poly-mh", "L0", transform=M._poly_only)),
            (spec(sample, epoch, population, target, "none", extra="mh"), M.LearnerHead("gbm", "L3", M._gbm)),
            (spec(sample, epoch, population, target, "mtot", extra="mh"), M.DirectLinear("linear", "L1")),
            (spec(sample, epoch, population, target, "mass_size", extra="mh"), M.DirectLinear("linear", "L1")),
            (spec(sample, epoch, population, target, "raw24", extra="mh"), M.DirectLinear("ridge", "L1", ridge_grid=M.RIDGE_GRID)),
            (spec(sample, epoch, population, target, "raw24", extra="mh"), M.LearnerHead("gbm", "L3", M._gbm)),
            (spec(sample, epoch, population, target, "raw24", extra="mh"),
             M.DirectLinear("pca4-poly2", "L2", transform=M._pca_poly(5), ridge_grid=M.RIDGE_GRID))] \
        + [(spec(sample, epoch, population, target, "raw24", extra="mh"), m) for m in M.augmented_methods()]


def blocks():
    out = {"mh-parent": full_ladder("parent", 0, "parent", "mh")}
    out["mh-sens"] = reduced_ladder("curated", 0, "complete", "mh") + reduced_ladder("curated", 0, "fit2356", "mh")
    out["mh-highz"] = []
    for k in range(1, 5):
        out["mh-highz"] += reduced_ladder("curated", k, "asis", "mh") + reduced_ladder("curated", k, "complete", "mh")
    out["conc"] = full_ladder("parent", 0, "parent", "logc", flexible=True, generative=False) + oracle_cells("parent", 0, "parent", "logc")
    for k in range(1, 5):
        out["conc"] += reduced_ladder("curated", k, "asis", "logc")[:8] + oracle_cells("curated", k, "asis", "logc")[:5]
    out["form"] = full_ladder("parent", 0, "parent", "t50", flexible=False, generative=False) \
        + [(spec("parent", 0, "parent", "t50", "raw24"), M.LearnerHead("gbm", "L3", M._gbm))] + oracle_cells("parent", 0, "parent", "t50")
    out["form-extra"] = []
    for target in ("t75", "t90"):
        raw = spec("parent", 0, "parent", target, "raw24")
        out["form-extra"] += reference_cells("parent", 0, "parent", target)[:2] \
            + [(raw, M.DirectLinear("ridge", "L1", ridge_grid=M.RIDGE_GRID))] + [(raw, m) for m in M.augmented_methods()] \
            + oracle_cells("parent", 0, "parent", target)
    return out


def main(block=None, smoke=False):
    gate = json.loads((C.OUTDIR / "gate_mechanics.json").read_text())
    assert gate["passed"], "the mechanics gate has not passed: no real-data score"
    t_start = time.time()
    todo = blocks()
    names = [block] if block else list(todo)
    print(f"{RULE}\nexp87 STAGE 3 — the ladder, blocks {names}{' (SMOKE)' if smoke else ''} (config {C.config_hash()})\n{RULE}", flush=True)
    for name in names:
        cells = todo[name][:6] if smoke else todo[name]
        print(f"\n--- block {name}: {len(cells)} cells ---", flush=True)
        for sp, method in cells:
            try:
                HN.run_cell(sp, method, smoke=smoke)
            except Exception as err:                      # one failing cell must not stop the ladder; it is recorded and reported
                print(f"  FAILED {sp} {method.name}: {type(err).__name__}: {err}", flush=True)
                with open(C.OUTDIR / "failed_cells.jsonl", "a") as fh:
                    fh.write(json.dumps(dict(sp, method=method.name, error=f"{type(err).__name__}: {err}")) + "\n")
        print(f"--- block {name} done at {time.time() - t_start:.0f} s ---", flush=True)
    print(f"\nSTAGE 3 blocks {names} finished in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    a = sys.argv
    main(block=(a[a.index("--block") + 1] if "--block" in a else None), smoke="--smoke" in a)
