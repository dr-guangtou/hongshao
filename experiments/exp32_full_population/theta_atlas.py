"""exp32 step 1b — the per-galaxy 7-param theta atlas for the full population.

Fits the phase-3 parametric-mass transport emulator (param_emulator.fit, 12
starts, zero free masses) to every galaxy in the population cache, per MAH
config. Output per config: theta (n,7), in-sample per-epoch profile max|rel|
(n,5) — the per-galaxy floor across the whole mass range, and the raw material
for the anatomy/conditioning/scatter steps.

Run:  PYTHONPATH=. uv run python experiments/exp32_full_population/theta_atlas.py \
        {real|diffmah} [--dev] [--workers K]
Demo: ... theta_atlas.py demo   (one galaxy must reproduce the exp30 fit exactly)
"""
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP29 = ROOT / "experiments" / "exp29_outer_deposit"
EXP30 = ROOT / "experiments" / "exp30_transport_kernel"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP29))
sys.path.insert(0, str(EXP30))
import param_emulator as pe                                                          # noqa: E402
from run import dipfree_mah                                                          # noqa: E402
from real_mah_test import real_mah                                                   # noqa: E402

OUTDIR = HERE / "outputs"
POP_NPZ = OUTDIR / "population.npz"
MAHFUN = {"real": real_mah, "diffmah": dipfree_mah}
_POP = None


def _init():
    global _POP
    _POP = np.load(POP_NPZ)


def fit_one(args):
    """(row, config) -> (row, theta (7,), maxrel (5,)); NaNs if the MAH is missing."""
    row, config = args
    gi = int(_POP["index"][row])
    mah = MAHFUN[config](gi)
    data = [_POP["data"][row][k] for k in range(5)]
    if mah is None:
        return row, np.full(7, np.nan), np.full(5, np.nan)
    cogs, _, theta = pe.fit(mah, data)
    if cogs is None:                       # all starts invalid (seen at the low-mass end)
        return row, np.full(7, np.nan), np.full(5, np.nan)
    return row, theta, pe.tf.maxrel(cogs, data)


def build(config, rows, workers):
    n = len(rows)
    theta = np.full((n, 7), np.nan)
    mr = np.full((n, 5), np.nan)
    t0 = time.time()
    with Pool(workers, initializer=_init) as pool:
        for k, (row, th, m) in enumerate(
                pool.imap_unordered(fit_one, [(r, config) for r in rows], chunksize=8)):
            i = int(np.searchsorted(rows, row))
            theta[i], mr[i] = th, m
            if (k + 1) % 200 == 0:
                el = time.time() - t0
                print(f"  {k+1}/{n}  ({el/60:.1f} min, ETA {el/(k+1)*(n-k-1)/60:.0f} min)",
                      flush=True)
    return theta, mr


def main():
    config = sys.argv[1]
    dev = "--dev" in sys.argv
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv \
        else max(os.cpu_count() - 2, 2)
    pop = np.load(POP_NPZ)
    rows = pop["dev100"] if dev else np.arange(len(pop["index"]))
    tag = "_dev" if dev else ""
    out_npz = OUTDIR / f"theta_atlas_{config}{tag}.npz"
    print(f"theta atlas [{config}{tag}]: {len(rows)} galaxies, {workers} workers")
    t0 = time.time()
    theta, mr = build(config, np.asarray(rows), workers)
    np.savez(out_npz, rows=rows, index=pop["index"][rows], logms=pop["logms"][rows],
             theta=theta, maxrel=mr)
    ok = np.isfinite(theta).all(1)
    print(f"wrote {out_npz}  ({ok.sum()}/{len(rows)} fits ok, "
          f"{(time.time()-t0)/60:.1f} min)")
    print(f"  per-galaxy floor, median max|rel| per epoch: "
          + " ".join(f"{100*np.nanmedian(mr[:, k]):.1f}%" for k in range(5)))


def demo():
    """One exp30 galaxy (real config) must reproduce the stored phase-3 fit."""
    _init()
    d30 = np.load(EXP30 / "outputs" / "param_emulator.npz")
    row = int(np.where(_POP["index"] == d30["index"][0])[0][0])
    _, theta, mr = fit_one((row, "real"))
    assert np.allclose(theta, d30["params"][0], rtol=1e-6), (theta, d30["params"][0])
    print(f"theta_atlas.demo OK: gal {int(d30['index'][0])} reproduces the exp30 fit "
          f"(epoch-avg max|rel| {100*mr.mean():.1f}%)")


if __name__ == "__main__":
    if sys.argv[1:2] == ["demo"]:
        demo()
    else:
        sys.exit(main())
