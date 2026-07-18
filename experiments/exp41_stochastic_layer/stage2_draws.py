"""exp41 stage 2 — generative draws, judged on the pre-registered criteria.

The layer: per-galaxy deviations applied to the official kernel theta
(sig alone = 1-D; sig + q jointly = 2-D), drawn from a distribution
calibrated on ONE half of the sample and scored against the OTHER half
(symmetrized), so the plane criterion is held-out by construction.

Distribution variants per dimensionality:
  emp    empirical resampling of the train-half deviation pairs,
         MEAN-CENTERED (criterion 2: the mean model is untouched) —
         preserves the measured heavy tails, skew, and the -0.31
         (sig, q) correlation with zero parametric assumptions;
  gauss  zero-mean Gaussian with the train-half covariance — the
         minimal parametric reference.

Draws are clipped to the physical box (stage 1: only ~2% pile-up).

Judged (pre-registered): centered plane energy / split-half floor ~ 1
on the three qa tier-2b observational planes (M(<30) vs M(30-50),
M(<30) vs M(50-100), M(<2Re) vs M(2-4Re)), held-out; the differential
and overshoot tests on the DRAWN population inside the adopted band;
the drawn theta mean equal to the official theta.

Run: PYTHONPATH=. uv run python experiments/exp41_stochastic_layer/\
stage2_draws.py {demo|run|physics} [--dev]
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

OUTDIR = HERE / "outputs"
E38_DIR = ROOT / "experiments/exp38_deposit_rethink"
E40_DIR = ROOT / "experiments/exp40_epoch_objective"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
KS_ALL = [0, 1, 2, 3, 4]
I_SIG, I_Q = 4, 2
LO6 = np.array([1.0, 0.0, 0.0, 0.0, 0.05, 1.06])
HI6 = np.array([3.0, 6.0, 3.0, 3.0, 2.0, 6.0])
N_REAL = 8                                   # realizations per direction
VARIANTS = ("1d-emp", "1d-gauss", "2d-emp", "2d-gauss")
_W = {}


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _w_init(rows):
    s3 = _load_by_path("exp38_stage3", E38_DIR / "stage3_inner.py")
    s3._w_init(rows)
    _W.update(s3._W)
    _W["th0"] = np.load(E40_DIR / "outputs/latestart.npz")["theta_z15"]


def draw_deltas(variant, train_d, size, rng):
    """(size, 2) deviations [d_sig, d_q]; mean-centered on the train
    half; 1-D variants zero the q column."""
    D = train_d - train_d.mean(axis=0, keepdims=True)
    if variant.endswith("emp"):
        out = D[rng.integers(0, len(D), size)].copy()
    else:
        cov = np.cov(D.T)
        out = rng.multivariate_normal(np.zeros(2), cov, size)
    if variant.startswith("1d"):
        out[:, 1] = 0.0
    return out


def drawn_cogs(gal_idx, deltas):
    """Model CoGs (len(gal_idx), 5, nR) with per-galaxy deviated theta,
    clipped to the physical box."""
    s2 = _W["s2"]
    e = _W["e"]
    th0 = _W["th0"]
    out = np.full((len(gal_idx), 5, len(e.R)), np.nan)
    for i, (gi, d) in enumerate(zip(gal_idx, deltas)):
        p = th0.copy()
        p[I_SIG] = np.clip(p[I_SIG] + d[0], LO6[I_SIG], HI6[I_SIG])
        p[I_Q] = np.clip(p[I_Q] + d[1], LO6[I_Q], HI6[I_Q])
        c = s2.model_cogs(p, _W["gals"][gi], KS_ALL, "1ch-mof")
        if c is not None:
            out[i] = c
    return out


def _real_one(args):
    """One (variant, direction, realization): plane scores on the
    held-out half."""
    from hongshao import qa
    variant, seed, train_idx, score_idx, d_pairs = args
    rng = np.random.default_rng(seed)
    deltas = draw_deltas(variant, d_pairs, len(score_idx), rng)
    cogs = drawn_cogs(score_idx, deltas)
    data = np.stack([_W["gals"][i]["data"] for i in score_idx])
    truth, model, _, _ = qa.measure_all(cogs, data, _W["e"].R)
    res = {}
    for kx, ky in qa.PLANES:
        for j in KS_ALL:
            tx = np.log10(np.clip(truth[kx][:, j], 1.0, None))
            ty = np.log10(np.clip(truth[ky][:, j], 1.0, None))
            mx = np.log10(np.clip(model[kx][:, j], 1.0, None))
            my = np.log10(np.clip(model[ky][:, j], 1.0, None))
            pe = qa.plane_energy(np.column_stack([tx, ty]),
                                 np.column_stack([mx, my]))
            res[(kx, ky, j)] = (pe["energy_ratio"],
                                pe["energy_ratio_centered"])
    return variant, res


def cmd_run(dev=False):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    tag = "_dev" if dev else ""
    from hongshao import qa
    d1 = np.load(OUTDIR / f"stage1_dist{tag}.npz")
    gals = _W["gals"]
    n = len(gals)
    row_to_i = {g["row"]: i for i, g in enumerate(gals)}
    # deviation pairs aligned to the gal list
    d_sig = np.zeros(n)
    d_q = np.zeros(n)
    for r, s, q in zip(d1["row"], d1["d_sig_2d"], d1["d_q_2d"]):
        if int(r) in row_to_i:
            i = row_to_i[int(r)]
            d_sig[i], d_q[i] = s, q
    pairs = np.column_stack([d_sig, d_q])
    rng0 = np.random.default_rng(41)
    perm = rng0.permutation(n)
    halves = (perm[: n // 2], perm[n // 2:])
    jobs = []
    seed = 0
    for variant in VARIANTS:
        for a, b in ((0, 1), (1, 0)):
            for s in range(N_REAL):
                seed += 1
                jobs.append((variant, seed, halves[a], halves[b],
                             pairs[halves[a]]))
    workers = max(os.cpu_count() - 2, 2)
    t0 = time.time()
    print(f"exp41 stage 2 — drawn populations, held-out plane scores "
          f"(n={n}{', DEV' if dev else ''}; {N_REAL} realizations x 2 "
          "directions x 4 variants):", flush=True)
    with Pool(workers, initializer=_w_init, initargs=(rows,)) as pool:
        results = pool.map(_real_one, jobs)
    # the mean-model reference, same held-out protocol (no deviations)
    ref = {}
    for a, b in ((0, 1), (1, 0)):
        variant, res = _real_one(("1d-emp", 999, halves[a], halves[b],
                                  np.zeros((len(halves[a]), 2))))
        for k, v in res.items():
            ref.setdefault(k, []).append(v)
    agg = {v: {} for v in VARIANTS}
    for variant, res in results:
        for k, v in res.items():
            agg[variant].setdefault(k, []).append(v)
    print("  centered plane-energy / split-half floor (median over "
          "realizations; the pre-registered target is ~1; 'mean' = the "
          "deterministic kernel, the exp32-era 3.5-4.8x problem):")
    e = _W["e"]
    for kx, ky in qa.PLANES:
        print(f"    plane {kx} vs {ky}:")
        hdr = "      " + " " * 10 + "  ".join(
            f"z={e.ANCHOR_Z[j]}".rjust(7) for j in KS_ALL)
        print(hdr)
        cells = [f"{np.median([v[1] for v in ref[(kx, ky, j)]]):7.1f}"
                 for j in KS_ALL]
        print("      mean    : " + "  ".join(cells))
        for variant in VARIANTS:
            cells = [f"{np.median([v[1] for v in agg[variant][(kx, ky, j)]]):7.1f}"
                     for j in KS_ALL]
            print(f"      {variant:8s}: " + "  ".join(cells))
    np.savez(OUTDIR / f"stage2_scores{tag}.npz",
             **{f"{v}_{kx}_{ky}_{j}": np.array(agg[v][(kx, ky, j)])
                for v in VARIANTS for kx, ky in qa.PLANES for j in KS_ALL},
             **{f"ref_{kx}_{ky}_{j}": np.array(ref[(kx, ky, j)])
                for kx, ky in qa.PLANES for j in KS_ALL})
    print(f"  wrote {OUTDIR / f'stage2_scores{tag}.npz'} "
          f"({(time.time()-t0)/60:.1f} min)")


def cmd_physics(dev=False, variant="2d-emp"):
    """The physics band check on a full drawn population (criterion 2)."""
    from hongshao.profile_emulator import density_from_cog
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    tag = "_dev" if dev else ""
    d1 = np.load(OUTDIR / f"stage1_dist{tag}.npz")
    gals = _W["gals"]
    e = _W["e"]
    n = len(gals)
    row_to_i = {g["row"]: i for i, g in enumerate(gals)}
    d_sig = np.zeros(n)
    d_q = np.zeros(n)
    for r, s, q in zip(d1["row"], d1["d_sig_2d"], d1["d_q_2d"]):
        if int(r) in row_to_i:
            i = row_to_i[int(r)]
            d_sig[i], d_q[i] = s, q
    pairs = np.column_stack([d_sig, d_q])
    rng = np.random.default_rng(7)
    deltas = draw_deltas(variant, pairs, n, rng)
    cogs = drawn_cogs(np.arange(n), deltas)
    data = np.stack([g["data"] for g in gals])
    logms = np.array([g["logms"] for g in gals])
    print(f"exp41 stage 2 physics on ONE drawn population [{variant}] "
          "(criterion: within the adopted kernel's band — differential "
          "0.39-0.41/0.12-0.13, overshoot T1 +0.02-0.05):")
    ed3, rows_d = e.differential(cogs, data, logms)
    cells = [f"z{e.ANCHOR_Z[k+1]}->z{e.ANCHOR_Z[k]}: "
             f"{rows_d[('data', 2, k)][0]:.2f}/"
             f"{rows_d[('data', 2, k)][1]:.2f} -> "
             f"{rows_d[('model', 2, k)][0]:.2f}/"
             f"{rows_d[('model', 2, k)][1]:.2f}" for k in range(4)]
    print("    differential massive tercile: " + "  ".join(cells))
    cogs0 = cogs[:, 0]
    ok = np.isfinite(cogs0).all(1) & (cogs0 > 0).all(1)
    ls_d, mid = density_from_cog(np.log10(data[:, 0]), e.R)
    ls_m = np.full_like(ls_d, np.nan)
    ls_m[ok] = density_from_cog(np.log10(cogs0[ok]), e.R)[0]
    dl = ls_m - ls_d
    bands = ((mid >= 30.0) & (mid < 60.0), (mid >= 60.0) & (mid <= 148.0))
    edges = np.quantile(logms, [0, 1 / 3, 2 / 3, 1])
    cells = []
    for b in range(3):
        sel = ok & (logms >= edges[b]) & (logms <= edges[b + 1] + 1e-9)
        cells.append(" ".join(f"{np.nanmedian(dl[np.ix_(sel, bm)]):+.3f}"
                              for bm in bands))
    print("    overshoot terciles (z=0.4, [30-60 / 60-148 kpc]): "
          + "  |  ".join(f"T{b+1} {c}" for b, c in enumerate(cells)))
    dm = deltas.mean(axis=0)
    print(f"    drawn-theta mean offset: d_sig {dm[0]:+.4f}, d_q "
          f"{dm[1]:+.4f} (criterion: ~0 — the mean model untouched)")


def demo():
    rows = np.load(POP_NPZ)["dev100"][:12]
    _w_init(rows)
    rng = np.random.default_rng(1)
    pairs = rng.normal(0, [0.08, 0.28], (12, 2))
    # (1) draws are mean-centered and 1-D variants zero the q column
    for variant in VARIANTS:
        d = draw_deltas(variant, pairs, 2000, rng)
        assert abs(d[:, 0].mean()) < 0.02
        if variant.startswith("1d"):
            assert np.all(d[:, 1] == 0.0)
    # (2) zero deviations reproduce the official model exactly
    c0 = drawn_cogs(np.arange(3), np.zeros((3, 2)))
    s2 = _W["s2"]
    for i in range(3):
        ref = s2.model_cogs(_W["th0"], _W["gals"][i], KS_ALL, "1ch-mof")
        assert np.abs(c0[i] / np.asarray(ref) - 1.0).max() < 1e-15
    # (3) drawn cogs are finite and monotone
    d = draw_deltas("2d-emp", pairs, 3, rng)
    c = drawn_cogs(np.arange(3), d)
    assert np.isfinite(c).all()
    assert (np.diff(c, axis=2) > -1e-9).all()
    print("exp41 stage2 demo OK: draws mean-centered (1-D zeroes q); "
          "zero deviation == the official model exactly; drawn CoGs "
          "finite and monotone")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    dev = "--dev" in sys.argv
    if cmd == "demo":
        demo()
    elif cmd == "run":
        cmd_run(dev)
    elif cmd == "physics":
        cmd_physics(dev, sys.argv[sys.argv.index("--variant") + 1]
                    if "--variant" in sys.argv else "2d-emp")
    else:
        sys.exit(f"unknown subcommand {cmd!r}")
