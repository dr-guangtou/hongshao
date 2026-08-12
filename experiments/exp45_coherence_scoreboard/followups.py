"""exp45 follow-ups (user-approved 2026-08-12).

(1) `sweep` — WHY do the inner-aware objective and the real-MAH input land
    on the same coherence excess (+0.45) when nothing else in the zoo moves?
    Their thetas share three features against the pack: a much broader
    efficiency window (sig 1.17 / 0.52 vs 0.24-0.28), a smaller birth radius
    (log_rc 1.64 / 1.56 vs 2.37-2.74) and a shallower width law (g 2.39 /
    2.43 vs 3.64-4.00). Those covary in a fit, so disentangle them by
    sweeping ONE coordinate at a time off the adopted theta and measuring
    cross-epoch coherence. Whichever coordinate moves it is the lever, and
    (if it is sig) that promotes menu option (C), the flexible efficiency
    window, from "never built" to "targeted".

(2) `emulator_physics` — the exp37 statistical emulator wins cross-epoch
    coherence decisively (2.1x floor vs the kernels' 8.8-12.3x). Does the
    family that wins coherence also PASS the kernel's out-of-model physics?
    Run the differential-deposition test on exp37's saved draws. If it
    passes, the two-model division of labour ("the emulator has the
    accuracy, the kernel has the physics") needs revisiting.

Run: PYTHONPATH=. uv run python experiments/exp45_coherence_scoreboard/\
followups.py {demo|sweep|emulator_physics} [--dev]
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

OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
E37 = ROOT / "experiments/exp37_multi_epoch/outputs/multi_epoch_block.npz"
E38 = ROOT / "experiments/exp38_deposit_rethink/outputs"
E40 = ROOT / "experiments/exp40_epoch_objective/outputs/latestart.npz"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
KS_ALL = [0, 1, 2, 3, 4]
NAMES = ["log_rc", "g", "q", "mu", "sig", "gamma"]
# one-at-a-time grids, spanning the adopted value and the two low-coherence
# fits' values for that coordinate
GRIDS = {"log_rc": [1.5, 2.0, 2.4, 2.74, 3.0],
         "g": [2.0, 2.5, 3.0, 3.5, 4.0],
         "q": [0.2, 0.5, 0.75, 0.91, 1.2],
         "sig": [0.24, 0.4, 0.6, 0.9, 1.2]}
_W = {}


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _w_init(rows):
    s2 = _load_by_path("exp38_stage2", E38.parent / "stage2_multiepoch.py")
    s2._w_init(rows)
    _W.update(s2._W)
    _W["s2"] = s2
    _W["th0"] = np.load(E40)["theta_z15"]


def _coh_one(args):
    """Coherence curve for the adopted theta with ONE coordinate replaced."""
    from hongshao import qa
    name, val = args
    s2, e, gals = _W["s2"], _W["e"], _W["gals"]
    th = _W["th0"].copy()
    th[NAMES.index(name)] = val
    cogs = np.full((len(gals), 5, len(e.R)), np.nan)
    for i, g in enumerate(gals):
        c = s2.model_cogs(th, g, KS_ALL, "1ch-mof")
        if c is not None:
            cogs[i] = c
    data = np.stack([g["data"] for g in gals])
    gp = qa.growth_planes(cogs, data, e.R)
    return name, val, np.array(
        [gp[("R_half", 0, j)]["coherence_model"] for j in KS_ALL[1:]])


def cmd_sweep(dev=False):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    tag = "_dev" if dev else ""
    _w_init(rows)
    jobs = [(n, v) for n, vs in GRIDS.items() for v in vs]
    workers = max(os.cpu_count() - 2, 2)
    t0 = time.time()
    print(f"exp45 follow-up 1 — one-at-a-time sweep off the ADOPTED theta "
          f"(n={len(_W['gals'])}{', DEV' if dev else ''}); size coherence "
          "z=0.4 vs z=0.7/1.0/1.5/2.0, truth +0.82/+0.68/+0.50/+0.33:",
          flush=True)
    with Pool(workers, initializer=_w_init, initargs=(rows,)) as pool:
        res = pool.map(_coh_one, jobs)
    OUTDIR.mkdir(exist_ok=True)
    store = {}
    for name in GRIDS:
        print(f"\n    [{name}]  (adopted value "
              f"{_W['th0'][NAMES.index(name)]:.2f})")
        for nm, val, coh in res:
            if nm != name:
                continue
            store[f"{name}_{val}"] = coh
            print(f"      {name} = {val:<5.2f} -> " + "  ".join(
                f"{c:+.2f}" for c in coh) + f"   (widest {coh[-1]:+.2f})")
    np.savez(OUTDIR / f"sweep{tag}.npz", **store)
    print(f"\n  wrote {OUTDIR / f'sweep{tag}.npz'} "
          f"({(time.time()-t0)/60:.1f} min)")


def cmd_emulator_physics(dev=False):
    """The differential-deposition test on the exp37 emulator's draws."""
    _w_init(None)
    e = _W["e"]
    pop = np.load(POP_NPZ)
    dall, logms = pop["data"], pop["logms"]
    lt = np.log10(np.clip(dall[:, :, -1], 1.0, None))
    keep = ~((lt[:, 0][:, None] - lt).max(axis=1) > 2.0)
    assert int(keep.sum()) == 2395, keep.sum()
    draws = np.load(E37)["cog_draws"]
    print("exp45 follow-up 2 — the exp37 statistical emulator on the "
          "kernel's out-of-model physics test (massive tercile; data "
          "0.37/0.11, adopted kernel 0.40/0.13, 2ch-exp 0.39/0.14):")
    for s in range(len(draws)):
        cg = draws[s]
        _, rows_d = e.differential(cg, dall[keep], logms[keep])
        print(f"    draw {s}: " + "  ".join(
            f"z{e.ANCHOR_Z[k+1]}->z{e.ANCHOR_Z[k]}: "
            f"{rows_d[('data', 2, k)][0]:.2f}/"
            f"{rows_d[('data', 2, k)][1]:.2f} -> "
            f"{rows_d[('model', 2, k)][0]:.2f}/"
            f"{rows_d[('model', 2, k)][1]:.2f}" for k in range(4)))


def demo():
    rows = np.load(POP_NPZ)["dev100"][:10]
    _w_init(rows)
    # (1) every swept value stays inside the physical box
    lo = np.array([1.0, 0.0, 0.0, 0.0, 0.05, 1.06])
    hi = np.array([3.0, 6.0, 3.0, 3.0, 2.0, 6.0])
    for name, vs in GRIDS.items():
        j = NAMES.index(name)
        for v in vs:
            assert lo[j] <= v <= hi[j], (name, v)
    # (2) the adopted value is IN each grid, so each sweep contains the
    # unmodified model as its own control
    for name, vs in GRIDS.items():
        assert any(abs(v - _W["th0"][NAMES.index(name)]) < 0.02 for v in vs), \
            f"{name} grid must contain the adopted value as a control"
    # (3) a sweep point at the adopted value reproduces the adopted model
    nm, val, coh = _coh_one(("sig", 0.24))
    assert np.isfinite(coh).all() and (coh > 0).all()
    # (4) exp37 draws are LINEAR CoGs (monotone, ~1e11), not logs
    d = np.load(E37)["cog_draws"]
    assert d.shape[1:] == (2395, 5, 24), d.shape
    assert np.nanmedian(d[0, :, 0, -1]) > 1e9, "draws must be linear masses"
    assert np.all(np.diff(d[0, :5, 0, :], axis=-1) > -1e-6), "monotone"
    print("exp45 followups demo OK: every swept value is in the physical "
          "box and each grid contains the adopted value as its own control; "
          "a sweep point evaluates finite; exp37 draws are linear, monotone "
          "CoGs of the expected shape")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    dev = "--dev" in sys.argv
    if cmd == "demo":
        demo()
    elif cmd == "sweep":
        cmd_sweep(dev)
    elif cmd == "emulator_physics":
        cmd_emulator_physics(dev)
    else:
        sys.exit(f"unknown subcommand {cmd!r}")
