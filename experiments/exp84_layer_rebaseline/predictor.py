"""exp84 — THE PREDICTOR the stochastic layer is re-baselined on: the adopted
mean (exp82, `exp74/rebaseline.adopted_mean()`) on the MEASURED halo history,
with per-galaxy, per-epoch deviations of the two deposit sizes
(`size_law.predict_law(size_dev=)`).

This is the half-day rebuild `rebaseline.py`'s docstring records as owed
since exp74: the v1 layer (exp60) ran on exp57's X3 problem and the
official-curve step engine, neither of which the adopted mean uses.

`build(smoke)` returns the fitting sample (THE FITTING SAMPLE rule:
`selection.fitting_sample_mask`, printed with what each criterion removed)
and a `Predictor` whose `predict(size_dev)` is the one model call every
stage makes. Nothing here fits anything.

Run the smoke (it MEASURES one call's wall time and peak resident size, the
numbers every stage's budget is set from — never estimate):
    HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 \\
    PYTHONPATH=. uv run python -u experiments/exp84_layer_rebaseline/predictor.py [--smoke]
"""
from __future__ import annotations

import importlib.util
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp63_analytic_growth", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import model2 as M2                                      # noqa: E402


def _by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RB = _by_path("exp74_rebaseline", ROOT / "experiments/exp74_c19_history_leak/rebaseline.py")   # exp54 shadows it
SL = _by_path("exp80_size_law", ROOT / "experiments/exp80_deposit_size_law/size_law.py")

RULE, THIN = "=" * 100, "-" * 100
ANCHOR_Z = (0.4, 0.7, 1.0, 1.5, 2.0)
EPOCHS = (0, 1, 2, 3, 4)
OUTDIR = HERE / "outputs"
#: the two size axes of the layer and the engine parameter each one deviates
AXES = ("c", "e")
AXIS_PARAMETER = {"c": "log_f_c", "e": "log_f_e"}


class Predictor:
    """The adopted mean on the measured history for one fixed set of galaxies.

    `predict(size_dev=None, nodes=FULL_NODES, rows=None)` -> M*(<R) [Msun],
    (n, 5, len(R)) on the standard grid; `size_dev` = dict(c=(n, 5), e=(n, 5))
    in dex (see `size_law.predict_law`). `rows` restricts the call to a subset
    of galaxies (positions in this predictor's sample) and the deviation
    arrays are then indexed by the SAME positions.
    """

    def __init__(self, curves, adopted, R=F.R_GRID):
        self.curves = list(curves)
        self.adopted = adopted
        self.spec, self.theta13, self.law = adopted["spec"], adopted["theta13"], adopted["law"]
        self.R = np.asarray(R, float)
        self.n = len(self.curves)
        self.n_call = 0

    def theta_with(self, **changes):
        """theta13 with named parameters replaced (shared sweeps of an axis)."""
        th = np.asarray(self.theta13, float).copy()
        for name, value in changes.items():
            th[self.spec.index(name)] = value
        return th

    def predict(self, size_dev=None, nodes=M2.FULL_NODES, rows=None, theta13=None):
        self.n_call += 1
        curves = self.curves if rows is None else [self.curves[i] for i in rows]
        dev = None
        if size_dev is not None:
            dev = {a: np.asarray(size_dev[a], float) for a in AXES}
            if rows is not None:
                dev = {a: dev[a][rows] for a in AXES}
        th = self.theta13 if theta13 is None else theta13
        return SL.predict_law(self.spec, th, self.law, curves, self.R, epochs=EPOCHS, nodes=nodes, size_dev=dev)


def build(smoke=False, verbose=True):
    """(recs, data, keep, lmh, predictor): the fitting sample as a (n,) bool
    `keep` over the records, the measured halo masses per epoch `lmh` (n, 5),
    and the predictor over ALL records (so indices match `data`)."""
    recs, data, mask, lmh_dm, lmh_bins, meas, fz, spec2, th_inc, pr = RB.build(smoke)
    keep = np.asarray(mask, bool).all(axis=1)
    adopted = RB.adopted_mean()
    if verbose:
        print(f"\n  THE MEAN: the adopted mean (exp82), {adopted['file'].name}; "
              f"{len(adopted['theta_full'])} parameters (tau_d held at {adopted['theta13'][adopted['spec'].index('tau_d')]:.2f}, "
              f"q_e = {adopted['law']['q_e']:.3f}); input = the MEASURED history")
        print(f"  THE SAMPLE: {int(keep.sum())} of {len(recs)} galaxies (the fitting sample; the layer is calibrated on it "
              f"and scored held out inside it)")
    pred = Predictor(meas, adopted)
    return recs, data, keep, np.asarray(lmh_bins, float), pred


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3


def smoke_timing(pred, keep):
    """Measure one call under each condition the stages use."""
    rows = np.where(keep)[0]
    n = pred.n
    rng = np.random.default_rng(0)
    dev = dict(c=0.2 * rng.standard_normal((n, 5)), e=0.2 * rng.standard_normal((n, 5)))
    print(f"\n{THIN}\n  TIMING one predict call on {len(rows)} galaxies (the budgets are set from these numbers)\n{THIN}")
    ref = {}
    for label, nodes, sd in (("FIT nodes, no deviation", M2.FIT_NODES, None),
                             ("FIT nodes, size_dev", M2.FIT_NODES, dev),
                             ("FULL nodes, no deviation", M2.FULL_NODES, None),
                             ("FULL nodes, size_dev", M2.FULL_NODES, dev)):
        t0 = time.time()
        m = pred.predict(size_dev=sd, nodes=nodes, rows=rows)
        dt = time.time() - t0
        ref[label] = m
        fin = np.isfinite(m).all(axis=(1, 2)).mean()
        print(f"  {label:<28} {dt:7.1f} s   peak RSS so far {peak_rss_gb():.2f} GB   finite {100 * fin:.1f}%")
    d = np.log10(ref["FIT nodes, no deviation"] / ref["FULL nodes, no deviation"])
    print(f"  FIT vs FULL nodes: max |dlog M| {np.nanmax(np.abs(d)):.4f} dex, median at 5 kpc {np.nanmedian(np.abs(d[:, :, 3])):.4f}")


def main(smoke=False):
    print(f"{RULE}\nexp84 — THE PREDICTOR: the adopted mean on the measured history, with size deviations\n{RULE}\n")
    recs, data, keep, lmh, pred = build(smoke)
    smoke_timing(pred, keep)
    print(f"\n  peak resident size of this process: {peak_rss_gb():.2f} GB")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv[1:])
