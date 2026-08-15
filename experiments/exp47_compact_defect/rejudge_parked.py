"""exp47 step 1 — re-judge the PARKED exp38 models under compact-aware scoring.

Free: every theta is on disk, so this is pure evaluation.

The record's standing puzzle. exp38's core channel (f_core = 0.185,
rc_core = 1.55 kpc, no parameter at a bound) and its retention floor
(f_ret = 0.084) both HALVED the inner-mass deficit and were then rejected
for "breaking the differential physics" (0.47-0.53 / 0.16-0.19 against
data 0.37/0.11). The conclusion drawn was structural: "inner-mass accuracy
and the differential-deposition physics are in genuine tension in this
family."

exp47 stage 3 gives a concrete reason to re-open it. At z=0.4 the compact
galaxies those components were built to fix are **6.3%** of the sample, so
both the loss and the physics tests are dominated by the 94% the model
already fits well. A component that helps the minority while mildly
disturbing the majority would produce exactly the observed signature.

PRE-REGISTERED READING (fixed before running):
  - the differential failure is concentrated in the EXTENDED
    sub-population while compact galaxies improve -> the rejection was a
    scoring artifact and the core channel returns as a live candidate;
  - it breaks for compact galaxies too -> the rejection stands, the
    tension is structural as exp38 concluded, and step 2 proceeds
    unencumbered.

Run: PYTHONPATH=. uv run python \
experiments/exp47_compact_defect/rejudge_parked.py {demo|run} [--dev]
"""
import importlib.util
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/exp46_highz_ridge"))

OUTDIR = HERE / "outputs"
E38D = ROOT / "experiments/exp38_deposit_rethink/outputs"
E38 = ROOT / "experiments/exp38_deposit_rethink"
E40 = ROOT / "experiments/exp40_epoch_objective/outputs/latestart.npz"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
KS_ALL = [0, 1, 2, 3, 4]
R_COMPACT = 5.0

_W = {}


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _w_init(rows):
    s3 = _load_by_path("exp38_stage3", E38 / "stage3_inner.py")
    s3._w_init(rows)
    _W.update(s3._W)
    _W["s3"] = s3
    _W["s2"] = s3._W["s2"]


def zoo():
    """[(label, npz key, forward-function name)] — every parked candidate."""
    return [
        ("adopted z<=1.5 (baseline)", E40, "theta_z15", "model_cogs"),
        ("exp38 stage2 (z<=2.0)", E38D / "stage2_multiepoch.npz",
         "theta_1ch-mof", "model_cogs"),
        ("core channel", E38D / "stage3_core.npz", "theta",
         "model_cogs_core"),
        ("retention floor", E38D / "stage3_retention.npz", "theta",
         "model_cogs_ret"),
        ("combined (inner-aware)", E38D / "stage3_combined.npz", "theta",
         "model_cogs_comb"),
        ("combined (plain loss)", E38D / "stage3_combined_plain.npz",
         "theta", "model_cogs_comb"),
        ("quenched core", E38D / "stage3_qcore.npz", "theta",
         "model_cogs_qcore"),
        ("re-aimed fiducial", E38D / "stage3_capacity.npz",
         "theta_inner_1ch-mof", "model_cogs"),
    ]


def _cogs_chunk(args):
    lo, hi, theta, fn_name = args
    s3, s2, gals = _W["s3"], _W["s2"], _W["gals"]
    fn = getattr(s3, fn_name, None) or getattr(s2, fn_name)
    out = np.full((hi - lo, len(KS_ALL), len(_W["e"].R)), np.nan)
    for i in range(lo, hi):
        c = fn(theta, gals[i], KS_ALL, "1ch-mof")
        if c is not None:
            out[i - lo] = c
    return lo, out


def build_cogs(theta, fn_name, n, nr, rows):
    workers = max(os.cpu_count() - 2, 2)
    step = max(n // (4 * workers) + 1, 1)
    jobs = [(lo, min(lo + step, n), theta, fn_name)
            for lo in range(0, n, step)]
    out = np.full((n, len(KS_ALL), nr), np.nan)
    with Pool(workers, initializer=_w_init, initargs=(rows,)) as pool:
        for lo, blk in pool.map(_cogs_chunk, jobs):
            out[lo:lo + len(blk)] = blk
    return out


def cmd_run(dev=False):
    from hongshao import qa
    from judge import verdict, print_verdict, physics_gates, PLANES

    rows = np.load(POP_NPZ)["dev100"] if dev else None
    tag = "_dev" if dev else ""
    _w_init(rows)
    gals, e = _W["gals"], _W["e"]
    R = e.R
    n = len(gals)
    data_cogs = np.stack([g["data"] for g in gals])
    pop = np.load(POP_NPZ)
    logms = np.array([pop["logms"][g["row"]] for g in gals])
    r50_t = qa._safe_rhalf(data_cogs, R, 0.5)

    print(f"exp47 step 1 — re-judging the PARKED models "
          f"(n={n}{', DEV' if dev else ''})", flush=True)

    rowsout = {}
    for label, path, key, fn_name in zoo():
        if not Path(path).exists():
            print(f"  [skip] {label}: {path} missing")
            continue
        d = np.load(path)
        if key not in d.files:
            print(f"  [skip] {label}: no key {key}")
            continue
        t0 = time.time()
        cogs = build_cogs(d[key], fn_name, n, len(R), rows)
        good = np.isfinite(cogs).all(axis=(1, 2)).mean()
        v = verdict(cogs, data_cogs, R, r50_truth=r50_t)
        gates = physics_gates(cogs, data_cogs, logms, e)
        print_verdict(v, f"{label}  ({100 * good:.0f}% usable, "
                         f"{time.time() - t0:.0f}s)", gates)
        rowsout[label] = (v, gates)

    # --- the pre-registered comparison ------------------------------------
    kx, ky = PLANES[0]
    key = f"{kx}|{ky}"
    print("\n\n=== THE PRE-REGISTERED COMPARISON ===")
    print("  M*(<5 kpc) bias at z=0.4 [%] and the z=0.4 plane score, by "
          "sub-population,\n  against the differential-deposition GATE:\n")
    print(f"  {'model':<26}{'M5 all':>8}{'M5 cmp':>8}{'M5 ext':>8}"
          f"{'plane cmp':>11}{'plane ext':>11}{'differential':>15}")
    for label, (v, g) in rowsout.items():
        sc = v[("plane", key, 0, "compact")]
        se = v[("plane", key, 0, "extended")]
        print(f"  {label:<26}"
              f"{100 * v[('inner', 5, 0, 'all')]:7.1f}%"
              f"{100 * v[('inner', 5, 0, 'compact')]:7.1f}%"
              f"{100 * v[('inner', 5, 0, 'extended')]:7.1f}%"
              f"{sc[0]:11.1f}{se[0]:11.1f}"
              f"{g['diff_model'][0]:8.2f}/{g['diff_model'][1]:.2f}")

    OUTDIR.mkdir(exist_ok=True)
    np.savez(OUTDIR / f"rejudge_parked{tag}.npz",
             labels=np.array(list(rowsout)),
             m5=np.array([[rowsout[l][0][("inner", 5, 0, g)]
                           for g in ("all", "compact", "extended")]
                          for l in rowsout]),
             diff=np.array([rowsout[l][1]["diff_model"] for l in rowsout]),
             over=np.array([rowsout[l][1]["over_T1"] for l in rowsout]))
    print(f"\n  wrote {OUTDIR / f'rejudge_parked{tag}.npz'}")


def demo():
    z = zoo()
    assert len(z) >= 6
    missing = [(l, str(p)) for l, p, k, _ in z if not Path(p).exists()]
    assert not missing, f"parked thetas absent: {missing}"
    for label, path, key, fn in z:
        d = np.load(path)
        assert key in d.files, (label, key, d.files)
    print(f"demo OK — all {len(z)} parked thetas present and keyed "
          "(pure evaluation, no fitting needed)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "demo":
        demo()
    elif cmd == "run":
        cmd_run(dev="--dev" in sys.argv)
    else:
        raise SystemExit(__doc__)
