"""exp53 — the fit driver

NAMED `fit.py`, not `run.py`: exp30/exp35 path-load a module called `run`,
and a second `run` on `sys.path` shadows it and breaks every downstream import.: the profiled `q` curve and the family shootout.

Two stages, one storage format, everything resumable at the level of a single
fit ("cell"), so a run can be sharded across independent processes and stopped
or resumed without losing work.

  qprofile   stage A. Fix `q` on a grid with `q = 0` as the endpoint and
             RE-OPTIMIZE every other parameter at each point. exp49 measured
             that a frozen-parameter SLICE and a profile disagree in SIGN in
             this model, so nothing here freezes anything but `q`.
  shootout   stage B. Each profile family at `q = 0` and at `q` free.
             `gauss` is the falsification control -- it is the wingless
             incumbent-of-record and should lose.

Multi-start is not optional: this loss has at least two basins (exp38 stage 2's
bounds-stress found the second one, and exp49 reproduced it from a start at
g = 2). Every cell is fitted from THREE starts and the best is kept; the
per-start losses are stored so a start-to-start spread can be reported rather
than assumed away.

Fits are SERIAL inside a process. Parallelism comes from `--shard i/N`, which
partitions the cell list -- the in-loss process pool has hung on this codebase
before, and a hung pool inside a 4-hour fit is far more expensive than running
four processes.

Run (full sample, four shards)::

    HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python \
      experiments/exp53_deposition_only/fit.py qprofile --shard 0/4

`--dev` uses the 100-galaxy subsample for WIRING ONLY. The dev100 subsample has
inverted the full-sample answer on `g` twice; no exp53 number is reported from
it.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/exp38_deposit_rethink"))

import families                                      # noqa: E402
import kernel as K                                   # noqa: E402
import stage2_multiepoch as s2                       # noqa: E402
from hongshao.fitting import minimize_loss           # noqa: E402

OUTDIR = HERE / "outputs"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
MAX_EVALS = 4000
SEED = 20260820

#: stage A grid. Spans the adopted q = 0.895 on both sides so the profiled
#: minimum is bracketed rather than read off an endpoint.
Q_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2)


# --------------------------------------------------------------------------- #
# starts                                                                       #
# --------------------------------------------------------------------------- #
def project(spec, theta_moffat_qfree):
    """Project an exp53 moffat-qfree theta into ``spec``'s layout.

    This is only meaningful BECAUSE the scale coordinate is a half-mass radius:
    `log_R50` denotes the same physical quantity in every family, so the
    adopted deposit size transfers across the shootout unchanged and each
    family starts from the same physical place rather than from whatever its
    own scale parameter happens to mean.
    """
    t = np.asarray(theta_moffat_qfree, float)
    log_r50, gg, q, mu, sig = t[0], t[1], t[2], t[3], t[4]
    out = [log_r50, gg] + ([q] if spec.q_free else []) + [mu, sig]
    if spec.family == "moffat":
        out.append(t[5])                       # carry the fitted gamma across
    else:
        out.extend(families.shape_defaults(spec.family))
    return _clip(spec, np.asarray(out + list(t[6:]), float))


def _box(spec):
    lo = np.array([-np.inf if b[0] is None else b[0] for b in spec.bounds])
    hi = np.array([np.inf if b[1] is None else b[1] for b in spec.bounds])
    return lo, hi


def _clip(spec, theta):
    """Clip strictly INSIDE the box: L-BFGS-B refuses a start on the wrong
    side of a bound, and a start exactly ON one starts the fit at a kink."""
    lo, hi = _box(spec)
    pad = np.where(np.isfinite(lo) & np.isfinite(hi), 1e-6 * (hi - lo), 0.0)
    return np.clip(np.asarray(theta, float), lo + pad, hi - pad)


def starts(spec, base):
    """Three starts per cell: the adopted basin, a deliberately COMPACT and
    early-forming alternative, and a seeded random displacement.

    The compact start is not arbitrary. exp38 stage 2 and exp49 both found the
    second basin by moving the deposit scale and the size-growth exponent
    together, so that is the direction displaced here.
    """
    # a DETERMINISTIC per-spec seed: Python's str hash is salted per process,
    # so `hash(label)` would give a different random start in every shard
    tag = int(hashlib.sha256(spec.label.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(SEED + tag % 10_000)
    out = {"adopted": _clip(spec, base)}
    c = np.asarray(base, float).copy()
    c[0] -= 0.45                                 # smaller deposits
    c[1] = max(c[1] - 1.8, 0.1)                  # weaker size growth in time
    i = 2 + (1 if spec.q_free else 0)            # index of `mu`; sig follows
    sig_hi = K.BASE_BOX["sig"][1]
    c[i + 1] = min(c[i + 1] * 2.2, 0.95 * sig_hi)   # a far wider window
    out["compact"] = _clip(spec, c)
    r = np.asarray(base, float).copy()
    n_base = len(r) - 6
    r[:n_base] *= 1.0 + rng.normal(0, 0.15, n_base)
    r[-6:] = rng.normal(0, 0.04, 6)
    out[f"random{SEED}"] = _clip(spec, r)
    return out


# --------------------------------------------------------------------------- #
# storage                                                                      #
# --------------------------------------------------------------------------- #
#: ALL stages share ONE store. Stage A's moffat cells and stage B's moffat
#: cells are the same fits; splitting the store would refit them.
def _store(stage=None, dev=False):
    return OUTDIR / f"fits{'_dev' if dev else ''}.npz"


def _load(store):
    return dict(np.load(store, allow_pickle=True)) if store.exists() else {}


def _save(store, done, cell, start, theta, loss, res, minutes):
    key = f"{cell}||{start}"
    done[f"theta::{key}"] = np.asarray(theta, float)
    done[f"loss::{key}"] = np.array([float(loss)])
    done[f"meta::{key}"] = np.array([json.dumps(dict(
        nfev=int(res.nfev), success=bool(res.success), minutes=round(minutes, 2),
        status=str(res.message)[:120]))])
    OUTDIR.mkdir(exist_ok=True)
    # write-then-rename, with a PER-PROCESS temp name: shards share the store,
    # so a fixed temp name would let two of them clobber each other's write.
    # numpy appends '.npz' unless the name already ends in it -- hence the
    # suffix order here.
    tmp = store.with_name(f"{store.stem}.tmp{os.getpid()}.npz")
    np.savez(tmp, **done)
    tmp.replace(store)                     # atomic: a shard never reads a torn file


def best_cell(done, cell):
    """``(theta, loss, start)`` of the best start for ``cell``, or None."""
    keys = [k.split("::", 1)[1] for k in done
            if k.startswith("theta::") and k.split("::", 1)[1].startswith(cell + "||")]
    if not keys:
        return None
    losses = [done[f"loss::{k}"][0] for k in keys]
    j = int(np.argmin(losses))
    return (done[f"theta::{keys[j]}"], float(losses[j]),
            keys[j].split("||", 1)[1])


def cell_spread(done, cell):
    """Best-to-worst loss spread across the starts of one cell."""
    keys = [k for k in done if k.startswith("loss::")
            and k.split("::", 1)[1].startswith(cell + "||")]
    if len(keys) < 2:
        return np.nan
    v = np.array([done[k][0] for k in keys])
    return float(v.max() - v.min())


def _bound_audit(spec, theta):
    hits = []
    for j, (lo, hi) in enumerate(spec.bounds):
        if lo is None or hi is None:
            continue
        d = min(theta[j] - lo, hi - theta[j]) / (hi - lo)
        if d < 0.02:
            hits.append(f"{spec.theta_names[j]}={theta[j]:.4f} "
                        f"({100*d:.2f}% from a bound)")
    return hits


# --------------------------------------------------------------------------- #
# the cell lists                                                               #
# --------------------------------------------------------------------------- #
def qprofile_cells():
    """Stage A: the profiled `q` curve, on the incumbent moffat family."""
    specs = [K.Spec("moffat", q_free=True)]
    specs += [K.Spec("moffat", q_free=False, q_fixed=float(q)) for q in Q_GRID]
    return [(sp.label, sp) for sp in specs]


#: The generalized-sigmoid family, added 2026-08-21 after the user pointed out
#: it kept being noted and never run. DEPOSITION-ONLY only (`q` = 0), which is
#: this branch's scope. `richards` nests the other two, so its fitted `nu`
#: measures which sigmoid the data want rather than assuming one.
SIGMOID_CELLS = ("gompertz_log", "loglogistic", "richards")


def sigmoid_cells():
    return [(K.Spec(f, q_free=False, q_fixed=0.0).label,
             K.Spec(f, q_free=False, q_fixed=0.0)) for f in SIGMOID_CELLS]


def shootout_cells():
    """Stage B: every family at `q = 0` and at `q` free.

    The two moffat cells are SHARED with stage A rather than refitted -- the
    cell names come from `Spec.label`, so `moffat-qfree` and `moffat-q0` are
    the same cell in the same store and are fitted exactly once.
    """
    specs = []
    for fam in families.FAMILIES:
        specs.append(K.Spec(fam, q_free=True))
        specs.append(K.Spec(fam, q_free=False, q_fixed=0.0))
    return [(sp.label, sp) for sp in specs]


def all_cells():
    """The union, de-duplicated, in run order: stage A first."""
    seen, out = set(), []
    for name, spec in qprofile_cells() + shootout_cells() + sigmoid_cells():
        if name not in seen:
            seen.add(name)
            out.append((name, spec))
    return out


# --------------------------------------------------------------------------- #
CELL_LISTS = dict(main=all_cells, qprofile=qprofile_cells,
                  shootout=shootout_cells, sigmoid=sigmoid_cells)


def _setup(dev):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    s2._w_init(rows)
    return s2._W["gals"], s2._W["e"].R


def run_stage(stage, dev=False, shard=(0, 1), max_hours=24.0, only=None):
    gals, R = _setup(dev)
    cells = CELL_LISTS[stage]()
    if only:
        cells = [c for c in cells if c[0] in only]
    base = K.from_exp38_theta(K.adopted_theta())
    store = _store(stage, dev)
    idx, nshard = shard
    jobs = [(name, spec, sname)
            for name, spec in cells
            for sname in starts(spec, project(spec, base))]
    jobs = [j for i, j in enumerate(jobs) if i % nshard == idx]
    print(f"exp53 {stage} — n={len(gals)}{' DEV' if dev else ''}, "
          f"shard {idx}/{nshard}: {len(jobs)} fits, budget {max_hours:.1f} h",
          flush=True)
    t0 = time.time()
    for name, spec, sname in jobs:
        done = _load(store)                    # re-read: other shards write too
        key = f"{name}||{sname}"
        if f"theta::{key}" in done:
            print(f"  [{key}] cached", flush=True)
            continue
        if (time.time() - t0) / 3600 > max_hours:
            print(f"  BUDGET REACHED — stopping before [{key}]", flush=True)
            break
        p0 = starts(spec, project(spec, base))[sname]
        prob = K.Problem(spec, gals, R)
        t1 = time.time()
        try:
            r = minimize_loss(prob.loss, p0, method="lbfgsb",
                              bounds=spec.bounds, max_evals=MAX_EVALS)
        except Exception as exc:               # one bad cell must not kill a
            print(f"  [{key}] FAILED: {type(exc).__name__}: {exc}",  # 4-h shard
                  flush=True)
            continue
        mins = (time.time() - t1) / 60
        done = _load(store)                    # re-read again before writing
        _save(store, done, name, sname, r.x, r.fun, r, mins)
        hits = _bound_audit(spec, r.x)
        print(f"  [{key:<28}] loss {r.fun:.8f}  ({r.nfev} evals, {mins:.1f} min,"
              f" conv={r.success})\n"
              f"      theta {np.round(r.x, 4).tolist()}\n"
              f"      at bound: {'; '.join(hits) if hits else 'NONE'}",
              flush=True)
    summarize(stage, dev, cells)


def summarize(stage, dev, cells=None):
    done = _load(_store(stage, dev))
    cells = cells or CELL_LISTS[stage]()
    print(f"\n  === {stage}: cell summary ===")
    print(f"    {'cell':<20}{'best loss':>14}{'start':>14}{'spread':>12}"
          f"{'starts':>8}   bounds")
    for name, spec in cells:
        b = best_cell(done, name)
        if b is None:
            print(f"    {name:<20}{'(not run)':>14}")
            continue
        theta, loss, sname = b
        nst = sum(1 for k in done if k.startswith(f"loss::{name}||"))
        hits = _bound_audit(spec, theta)
        print(f"    {name:<20}{loss:14.8f}{sname:>14}"
              f"{cell_spread(done, name):12.2e}{nst:8d}   "
              f"{'; '.join(hits)[:48] if hits else '-'}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "qprofile"
    dev = "--dev" in sys.argv
    shard = (0, 1)
    hours = 24.0
    only = None
    if "--shard" in sys.argv:
        a, b = sys.argv[sys.argv.index("--shard") + 1].split("/")
        shard = (int(a), int(b))
    if "--max-hours" in sys.argv:
        hours = float(sys.argv[sys.argv.index("--max-hours") + 1])
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1].split(",")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, cwd=ROOT).stdout.strip()
    print(f"# git {sha[:10]}  stage={stage}  dev={dev}  shard={shard}",
          flush=True)
    if stage in ("main", "qprofile", "shootout", "sigmoid"):
        run_stage(stage, dev, shard, hours, only)
    elif stage == "summary":
        summarize("main", dev)
    else:
        raise SystemExit(__doc__)
