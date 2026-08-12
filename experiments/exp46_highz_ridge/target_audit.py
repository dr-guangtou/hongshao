"""exp46 stage 1 — is the z >= 1.5 plane target real?

The adopted product breaks on the inner-vs-outer stellar mass planes at
z >= 1.5 (centered energy/split-half-floor 3.5 at z=1.5, 4.8 at z=2.0, vs
2.2 at z=0.4). The 2026-08-12 discussion falsified the fit-scope
explanation twice and left the deposit machinery as the leading cause —
but nobody has checked whether the TARGET is real structure a model
should chase.

That matters because ``qa.plane_energy`` compares two POPULATIONS and is
blind to which halo produced which galaxy, so a model can always be made
to match a wide target by injecting scatter. The question is therefore
not "is the spread reachable" but "is the spread REAL".

Stage 1, no fitting anywhere:
  1a  where the truth's plane scatter lives (by inner-mass quartile)
  1b  the resolution audit: truth R50 against qa's own 2 kpc grid start
      and 5 kpc marginally-resolved band; the stellar shot-noise floor
  1c  the near-empty progenitor census (exp37's >2 dex criterion)
  1d  the trimmed re-score of the adopted kernel, each targeted trim
      PAIRED with a random trim of the same size (a smaller sample has a
      larger split-half floor, which lowers E/floor for free)

Run: PYTHONPATH=. uv run python experiments/exp46_highz_ridge/\
target_audit.py {demo|run|figures} [--dev]
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
E38 = ROOT / "experiments/exp38_deposit_rethink/outputs"
E40 = ROOT / "experiments/exp40_epoch_objective/outputs/latestart.npz"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"

KS_ALL = [0, 1, 2, 3, 4]
Z_GRID = (0.4, 0.7, 1.0, 1.5, 2.0)

# TNG300-1 baryon (gas/star) particle mass. The shot-noise conclusion is
# insensitive to it: closing the z=2 gap (0.027 -> 0.338 dex) would need a
# particle mass ~150x larger.
M_PARTICLE = 1.1e7

# qa's own radial disclaimers, restated here as the thresholds under test.
R_GRID_MIN = 2.0        # CoG grid starts here; smaller R50 is extrapolated
R_MARGINAL = 5.0        # qa.RMIN_KPC — "inner 2-5 kpc marginally resolved"
PROG_DEX = 2.0          # exp37's broken-progenitor criterion
N_REP = 24              # random same-n trims per cell (see random_control)

# The two kpc planes that carry the failure (qa.PLANES[:2]).
PLANE_KEYS = [("kpc:M(<30)", "kpc:M(30-50)"),
              ("kpc:M(<30)", "kpc:M(50-100)")]

_W = {}


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _w_init(rows):
    """Worker state — the exp39-45 spawn-safe pattern (path-load exp38
    stage2, which owns the forward model and the galaxy list)."""
    s2 = _load_by_path("exp38_stage2", E38.parent / "stage2_multiepoch.py")
    s2._w_init(rows)
    _W.update(s2._W)
    _W["s2"] = s2


def _cogs_chunk(bounds):
    """Adopted-theta model CoGs for gals[lo:hi]. Every worker builds the
    same galaxy list in the same order, so slicing by index is safe."""
    lo, hi = bounds
    s2, gals = _W["s2"], _W["gals"]
    theta = np.load(E40)["theta_z15"]
    out = np.full((hi - lo, len(KS_ALL), len(_W["e"].R)), np.nan)
    for i in range(lo, hi):
        c = s2.model_cogs(theta, gals[i], KS_ALL, "1ch-mof")
        if c is not None:
            out[i - lo] = c
    return lo, out


# --------------------------------------------------------------------------- #
# stage 1a-1c — the audit of the TRUTH alone                                   #
# --------------------------------------------------------------------------- #
def shot_noise_dex(mass):
    """1-sigma log10 scatter from Poisson counting of stellar particles."""
    n = np.maximum(np.asarray(mass, float) / M_PARTICLE, 1.0)
    return 1.0 / (np.log(10.0) * np.sqrt(n))


def audit_scatter(truth, keys_xy):
    """1a — transverse scatter about the plane's fitted line, total and by
    inner-mass quartile. Returns {(plane, j): dict}."""
    kx, ky = keys_xy
    out = {}
    for j in range(truth[kx].shape[1]):
        x = np.log10(np.clip(truth[kx][:, j], 1.0, None))
        y = np.log10(np.clip(truth[ky][:, j], 1.0, None))
        ok = np.isfinite(x) & np.isfinite(y)
        slope, icpt = np.polyfit(x[ok], y[ok], 1)
        res = y - (slope * x + icpt)
        edges = np.quantile(x[ok], [0.0, 0.25, 0.5, 0.75, 1.0])
        byq = [float(np.std(res[ok & (x >= edges[q]) & (x <= edges[q + 1])]))
               for q in range(4)]
        out[j] = dict(slope=float(slope), scatter=float(np.std(res[ok])),
                      by_quartile=byq)
    return out


def audit_resolution(rh_truth):
    """1b — how much of the TRUTH sits inside qa's own resolution
    disclaimers, per epoch."""
    return [dict(median=float(np.nanmedian(rh_truth[:, j])),
                 frac_below_grid=float(np.mean(rh_truth[:, j] < R_GRID_MIN)),
                 frac_marginal=float(np.mean(rh_truth[:, j] < R_MARGINAL)))
            for j in range(rh_truth.shape[1])]


def audit_progenitor(data_cogs):
    """1c — log M*(<148 kpc) below the SAME galaxy at z=0.4, per epoch."""
    tot = np.log10(np.clip(data_cogs[:, :, -1], 1.0, None))
    down = tot[:, [0]] - tot
    return [dict(median=float(np.median(down[:, j])),
                 frac_1dex=float(np.mean(down[:, j] > 1.0)),
                 frac_2dex=float(np.mean(down[:, j] > PROG_DEX)))
            for j in range(down.shape[1])]


# --------------------------------------------------------------------------- #
# stage 1d — the trimmed re-score, with the sample-size control                #
# --------------------------------------------------------------------------- #
def trim_masks(rh_truth, data_cogs, truth_x):
    """Nested per-epoch keep-masks, cheapest cut first. Each is (n, nz)."""
    tot = np.log10(np.clip(data_cogs[:, :, -1], 1.0, None))
    keep_prog = (tot[:, [0]] - tot) <= PROG_DEX
    keep_grid = keep_prog & (rh_truth >= R_GRID_MIN)
    keep_marg = keep_grid & (rh_truth >= R_MARGINAL)
    lx = np.log10(np.clip(truth_x, 1.0, None))
    q75 = np.nanquantile(lx, 0.75, axis=0, keepdims=True)
    return [("all", np.ones_like(keep_prog)),
            (f"prog <={PROG_DEX:g}dex down", keep_prog),
            (f"+ R50 >= {R_GRID_MIN:g} kpc (on grid)", keep_grid),
            (f"+ R50 >= {R_MARGINAL:g} kpc (resolved)", keep_marg),
            ("top inner-mass quartile", lx >= q75)]


def _plane_score(truth, model, keys_xy, j, mask):
    from hongshao import qa
    kx, ky = keys_xy
    t = np.column_stack([np.log10(np.clip(truth[kx][mask, j], 1.0, None)),
                         np.log10(np.clip(truth[ky][mask, j], 1.0, None))])
    m = np.column_stack([np.log10(np.clip(model[kx][mask, j], 1.0, None)),
                         np.log10(np.clip(model[ky][mask, j], 1.0, None))])
    return qa.plane_energy(t, m)["energy_ratio_centered"]


def random_control(score_fn, n_total, n_keep, n_rep=N_REP, seed=0):
    """The mandatory control: the SAME n, drawn at random, n_rep times.

    Two things make a raw (targeted - random-median) delta uninterpretable,
    and both are absorbed by comparing against the whole random DISTRIBUTION
    instead: (1) trimming raises the split-half floor, lowering E/floor even
    for an unchanged model; (2) ``plane_energy`` estimates that floor from 8
    random half-splits and standardizes by the TRUTH's per-axis std, so a
    single influential galaxy moves the score. Measured on the first full
    run: dropping ONE galaxy shifted z=1.5 from 3.5 to 4.1.

    Returns the n_rep scores. The statistic to read is where the targeted
    score falls inside them, not the gap between medians.
    """
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_rep):
        m = np.zeros(n_total, bool)
        m[rng.choice(n_total, size=min(n_keep, n_total), replace=False)] = True
        vals.append(score_fn(m))
    return np.array(vals, float)


def _cell(targeted, rand):
    """'E [p]' — p is the fraction of same-n RANDOM trims scoring at or below
    the targeted trim. p near 0 means the targeted cut genuinely helped
    beyond shrinking the sample; p near 0.5 means it did nothing."""
    p = float(np.mean(rand <= targeted))
    return f"{targeted:.1f}[p={p:.2f}]", p


def _size_score(rh_t, rh_m, data_cogs, model_cogs, j, mask, frac_key):
    """Tier 2d for one epoch on a subsample. Mirrors ``qa.size_planes``'
    construction, but takes the enclosed radii precomputed (they are the
    expensive part and do not depend on the trim)."""
    from hongshao import qa
    tx = np.log10(np.clip(data_cogs[mask, j, -1], 1.0, None))
    mx = np.log10(np.clip(model_cogs[mask, j, -1], 1.0, None))
    t = np.column_stack([tx, np.log10(rh_t[frac_key][mask, j])])
    m = np.column_stack([mx, np.log10(rh_m[frac_key][mask, j])])
    return qa.plane_energy(t, m)["energy_ratio_centered"]


# --------------------------------------------------------------------------- #
# commands                                                                     #
# --------------------------------------------------------------------------- #
def cmd_run(dev=False):
    from hongshao import qa
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    tag = "_dev" if dev else ""
    _w_init(rows)
    gals, R = _W["gals"], _W["e"].R
    n = len(gals)
    data_cogs = np.stack([g["data"] for g in gals])

    print(f"exp46 stage 1 — is the z>=1.5 plane target real? "
          f"(n={n}{', DEV' if dev else ''})", flush=True)

    t0 = time.time()
    workers = max(os.cpu_count() - 2, 2)
    step = max(n // (4 * workers) + 1, 1)
    bounds = [(lo, min(lo + step, n)) for lo in range(0, n, step)]
    model_cogs = np.full_like(data_cogs, np.nan)
    with Pool(workers, initializer=_w_init, initargs=(rows,)) as pool:
        for lo, blk in pool.map(_cogs_chunk, bounds):
            model_cogs[lo:lo + len(blk)] = blk
    print(f"  adopted-theta model CoGs built ({time.time() - t0:.0f}s)",
          flush=True)

    truth, model, _, _ = qa.measure_all(model_cogs, data_cogs, R)
    rh_t = {f: qa._safe_rhalf(data_cogs, R, f) for f in qa.SIZE_FRACTIONS}
    rh_m = {f: qa._safe_rhalf(model_cogs, R, f) for f in qa.SIZE_FRACTIONS}
    zs = "  ".join(f"z={z:<5.1f}" for z in Z_GRID)

    # --- 1a ---------------------------------------------------------------
    print("\n  1a — TRUTH plane scatter about the fitted line "
          "[dex], total and by inner-mass quartile Q1..Q4:")
    sc = {}
    for kx, ky in PLANE_KEYS:
        sc[(kx, ky)] = audit_scatter(truth, (kx, ky))
        print(f"    {kx} vs {ky}")
        for j, z in enumerate(Z_GRID):
            s = sc[(kx, ky)][j]
            print(f"      z={z:3.1f}  slope {s['slope']:+.2f}  "
                  f"total {s['scatter']:.3f}  |  "
                  + " ".join(f"{v:.3f}" for v in s['by_quartile']))

    # --- 1b ---------------------------------------------------------------
    res = audit_resolution(rh_t[0.5])
    print("\n  1b — resolution audit (TRUTH R50 vs qa's own disclaimers):")
    print(f"    {'z':>5} {'med R50':>9} {'< 2 kpc':>9} {'< 5 kpc':>9}   "
          "(2 kpc = CoG grid start; 5 kpc = qa.RMIN_KPC)")
    for j, z in enumerate(Z_GRID):
        r = res[j]
        print(f"    {z:5.1f} {r['median']:8.2f}k "
              f"{100 * r['frac_below_grid']:8.1f}% "
              f"{100 * r['frac_marginal']:8.1f}%")
    print("\n    stellar shot-noise floor [dex] vs the measured scatter:")
    print(f"    {'z':>5} {'M(<30)':>9} {'M(30-50)':>9} {'M(50-100)':>10}   "
          f"{'measured 30-50':>15}")
    sn = {}
    for j, z in enumerate(Z_GRID):
        vals = [float(np.median(shot_noise_dex(truth[k][:, j])))
                for k in ("kpc:M(<30)", "kpc:M(30-50)", "kpc:M(50-100)")]
        p95 = float(np.quantile(shot_noise_dex(truth["kpc:M(30-50)"][:, j]),
                                0.95))
        sn[j] = vals + [p95]
        meas = sc[PLANE_KEYS[0]][j]["scatter"]
        print(f"    {z:5.1f} {vals[0]:9.3f} {vals[1]:9.3f} {vals[2]:10.3f}   "
              f"{meas:15.3f}   (95th pct 30-50: {p95:.3f})")

    # --- 1c ---------------------------------------------------------------
    prog = audit_progenitor(data_cogs)
    print("\n  1c — near-empty progenitor census "
          "(log M*(<148) below the SAME galaxy at z=0.4):")
    print(f"    {'z':>5} {'median':>8} {'>1 dex':>9} {'>2 dex':>9}")
    for j, z in enumerate(Z_GRID):
        p = prog[j]
        print(f"    {z:5.1f} {p['median']:8.2f} "
              f"{100 * p['frac_1dex']:8.1f}% {100 * p['frac_2dex']:8.1f}%")

    # --- 1d ---------------------------------------------------------------
    masks = trim_masks(rh_t[0.5], data_cogs, truth["kpc:M(<30)"])
    print(f"\n  1d — trimmed re-score, adopted deterministic kernel."
          f"\n       Cells are centered E/floor [p], where p = the fraction of"
          f"\n       {N_REP} RANDOM same-n trims scoring at or below it."
          f"\n       p ~ 0 = the cut really helped; p ~ 0.5 = it only shrank n."
          f"\n       The raw value alone is NOT interpretable (trimming lowers"
          f"\n       E/floor for free).")
    trim_out, zw = {}, "  ".join(f"z={z:<10.1f}" for z in Z_GRID)

    def _emit(title, score_at, tag):
        print(f"\n    {title}\n      {'trim':<28s}{zw}")
        for label, mk in masks:
            cells, rec = [], []
            for j in range(len(Z_GRID)):
                m = mk[:, j]
                e = score_at(j, m)
                nk = int(m.sum())
                if label == "all" or nk == len(m):
                    # nothing was cut: a same-n "random" trim is the same
                    # sample, so p would be a meaningless 1.00.
                    rec.append((e, np.nan, nk))
                    cells.append(f"{e:.1f}         ")
                    continue
                rnd = random_control(lambda mm, _j=j: score_at(_j, mm),
                                     len(m), nk, seed=1000 + j)
                txt, p = _cell(e, rnd)
                rec.append((e, p, nk))
                cells.append(txt)
                trim_out[(tag, label, j, "rand")] = rnd
            trim_out[(tag, label)] = np.array(rec)
            ns = "  ".join(f"n={int(v):<8d}" for v in np.array(rec)[:, 2])
            print(f"      {label:<28s}" + "  ".join(f"{c:<12s}" for c in cells)
                  + f"\n      {'':<28s}{ns}")

    for i, (kx, ky) in enumerate(PLANE_KEYS):
        _emit(f"tier 2b — {kx} vs {ky}",
              lambda j, m, _k=(kx, ky): _plane_score(truth, model, _k, j, m),
              f"2b{i}")
    _emit("tier 2d — the mass-size plane (R50)",
          lambda j, m: _size_score(rh_t, rh_m, data_cogs, model_cogs,
                                   j, m, 0.5),
          "2d")

    OUTDIR.mkdir(exist_ok=True)
    np.savez(OUTDIR / f"target_audit{tag}.npz",
             scatter=np.array([[sc[p][j]["scatter"] for j in range(5)]
                               for p in PLANE_KEYS]),
             by_quartile=np.array([[sc[p][j]["by_quartile"]
                                    for j in range(5)] for p in PLANE_KEYS]),
             shot_noise=np.array([sn[j] for j in range(5)]),
             r50_median=np.array([r["median"] for r in res]),
             frac_below_grid=np.array([r["frac_below_grid"] for r in res]),
             frac_marginal=np.array([r["frac_marginal"] for r in res]),
             prog_1dex=np.array([p["frac_1dex"] for p in prog]),
             trim_labels=np.array([la for la, _ in masks]),
             **{f"trim_{tag}_{li}": trim_out[(tag, la)]
                for tag in ("2b0", "2b1", "2d")
                for li, (la, _) in enumerate(masks) if (tag, la) in trim_out},
             **{f"rand_{tag}_{li}_{j}": trim_out[(tag, la, j, "rand")]
                for tag in ("2b0", "2b1", "2d")
                for li, (la, _) in enumerate(masks)
                for j in range(len(Z_GRID))
                if (tag, la, j, "rand") in trim_out})
    print(f"\n  wrote {OUTDIR / f'target_audit{tag}.npz'} "
          f"({time.time() - t0:.0f}s total)")


def demo():
    """Self-check on synthetics with known answers."""
    from hongshao import qa
    rng = np.random.default_rng(0)

    # 1. shot noise: the helper reproduces 1/(ln10 sqrt(N)) exactly.
    got = shot_noise_dex(np.array([100.0 * M_PARTICLE]))[0]
    assert abs(got - 1.0 / (np.log(10) * 10.0)) < 1e-12, got

    # 2. THE PAIRING-BLINDNESS ASSERTION — this is what killed the
    #    twin-matching design. A "model" that emits truth galaxies drawn
    #    WITH REPLACEMENT (a genuinely different multiset, exactly what a
    #    twin-draw predictor produces) still scores at the floor, so the
    #    plane metric cannot measure how predictable a galaxy is from its
    #    halo — every feature-conditioned resampler passes by construction.
    t = rng.normal(size=(1200, 2)) @ np.array([[1.0, 0.9], [0.0, 0.5]])
    resampled = t[rng.choice(len(t), size=len(t), replace=True)]
    assert not np.array_equal(np.sort(resampled, 0), np.sort(t, 0))
    r = qa.plane_energy(t, resampled)["energy_ratio_centered"]
    assert r < 1.6, f"a twin-draw resample should sit at the floor, got {r}"

    # 3. ... while a genuinely too-tight population is caught. This is the
    #    failure mode the model actually has at z=2.
    tight = t * np.array([1.0, 0.3])
    r_tight = qa.plane_energy(t, tight)["energy_ratio_centered"]
    assert r_tight > 3, f"under-dispersion must be caught, got {r_tight}"

    # 4. the trim control behaves: for a PERFECT model, both a targeted
    #    trim and a random trim of the same size sit at the floor, and the
    #    targeted-minus-random difference is ~0 (no free credit).
    keep = t[:, 0] > np.quantile(t[:, 0], 0.5)
    e_t = qa.plane_energy(t[keep], t[keep])["energy_ratio_centered"]
    idx = rng.choice(len(t), size=int(keep.sum()), replace=False)
    e_r = qa.plane_energy(t[idx], t[idx])["energy_ratio_centered"]
    assert abs(e_t - e_r) < 0.5, (e_t, e_r)

    # 5. ... and the control is NOT vacuous: shrinking n really does lower
    #    a fixed model's E/floor, which is why the pairing is mandatory.
    big = qa.plane_energy(t, tight)["energy_ratio_centered"]
    sub = rng.choice(len(t), size=200, replace=False)
    small = qa.plane_energy(t[sub], tight[sub])["energy_ratio_centered"]
    assert small < big, (small, big)

    # 6. the audit helpers run and are self-consistent on a toy population.
    cogs = np.cumsum(rng.lognormal(20, 0.3, size=(40, 5, 24)), axis=-1)
    prog = audit_progenitor(cogs)
    assert len(prog) == 5 and abs(prog[0]["median"]) < 1e-9, prog[0]
    rh = qa._safe_rhalf(cogs, np.asarray(qa_R()), 0.5)
    resx = audit_resolution(rh)
    assert len(resx) == 5 and 0.0 <= resx[0]["frac_marginal"] <= 1.0

    print("demo OK — including the pairing-blindness assertion "
          f"(a twin-draw resample of the truth scores {r:.2f}x floor; "
          f"a 0.3x-too-tight "
          f"population scores {r_tight:.1f}x; shrinking n moves a fixed "
          f"model {big:.1f} -> {small:.1f})")


def qa_R():
    from hongshao.tng_data import COG_RAD_KPC
    return COG_RAD_KPC


def cmd_figures(dev=False):
    raise SystemExit("figures: run `run` first; not implemented until the "
                     "stage-1 numbers are in (readouts are pre-registered "
                     "in README.md).")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    is_dev = "--dev" in sys.argv
    if cmd == "demo":
        demo()
    elif cmd == "run":
        cmd_run(dev=is_dev)
    elif cmd == "figures":
        cmd_figures(dev=is_dev)
    else:
        raise SystemExit(__doc__)
