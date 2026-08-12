"""exp44 stage 3 — the AR(1)-in-epoch stochastic layer, as an OPTION.

exp44 stage 2 measured the per-galaxy individuality to decorrelate across
epochs as AR(1) with rank correlation rho ~ 0.56, while the ADOPTED exp41
layer applies one deviation unchanged at every epoch — implicitly rho = 1.
This module implements both, selectable, with the adopted behavior as the
DEFAULT so the fiducial product is unchanged unless the evidence says
otherwise (user decision 2026-07-21).

  MODE_PERSISTENT ("rho=1", default)  one deviation per galaxy, all epochs
                                      — reproduces the adopted exp41 layer
  MODE_AR1        ("rho measured")    a per-epoch chain generated as a
                                      Gaussian AR(1) then rank-mapped onto
                                      the measured empirical pool

The rank-mapping (a Gaussian copula) is what keeps the change honest: the
MARGINAL deviation distribution at every epoch stays exactly the measured
heavy-tailed pool — whose tails exp41 showed carry real plane signal, and
which a naive Gaussian AR(1) would sand off — while only the cross-epoch
LINKAGE changes. Because a Gaussian copula compresses rank correlation
(Spearman = (6/pi) arcsin(rho_gauss/2)), the Gaussian AR(1) constant is
inverted from the measured Spearman target, rho_gauss = 2 sin(pi rho_s/6),
and the demo asserts the realized rank correlation hits the target.

Judged against exp41's pre-registered criteria, same held-out protocol
(pool calibrated on one half of the sample, applied to the other,
symmetrized): centered plane energy / split-half floor, plus the two gates
that an epoch-varying theta specifically threatens — the drawn population
must still conserve mass between epochs (the differential-deposition test
is INHERENTLY cross-epoch) and stay inside the adopted physics band.

Run: PYTHONPATH=. uv run python experiments/exp44_lowz_anchoring/\
ar1_layer.py {demo|run|physics} [--dev]
"""
import importlib.util
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

OUTDIR = HERE / "outputs"
E38_DIR = ROOT / "experiments/exp38_deposit_rethink"
E40_DIR = ROOT / "experiments/exp40_epoch_objective"
E41_DIR = ROOT / "experiments/exp41_stochastic_layer"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
KS_ALL = [0, 1, 2, 3, 4]
I_SIG = 4
LO6 = np.array([1.0, 0.0, 0.0, 0.0, 0.05, 1.06])
HI6 = np.array([3.0, 6.0, 3.0, 3.0, 2.0, 6.0])
MODES = ("persistent", "ar1")            # persistent = the adopted default
N_REAL = 4
# exp41's recorded held-out marks for the adopted layer, plane
# kpc:M(<30) vs kpc:M(30-50): mean 1.6/1.8/2.1/2.3/2.9, 1d-emp
# 1.1/1.4/1.6/1.8/2.6 — the persistent mode must reproduce the latter.
_W = {}


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _w_init(rows):
    s2 = _load_by_path("exp38_stage2", E38_DIR / "stage2_multiepoch.py")
    s2._w_init(rows)
    _W.update(s2._W)
    _W["s2"] = s2
    _W["th0"] = np.load(E40_DIR / "outputs/latestart.npz")["theta_z15"]
    _W["rho_s"] = float(np.load(
        HERE / "outputs/decorr_report.npz")["rho_sig_spearman"][0])


# --------------------------------------------------------------------------- #
# the two draw modes                                                           #
# --------------------------------------------------------------------------- #
def _empirical_quantile(pool_sorted, u):
    """Inverse CDF of the empirical pool at probabilities ``u`` (plotting
    positions (i+0.5)/n, linear interpolation between order statistics)."""
    n = len(pool_sorted)
    return np.interp(u, (np.arange(n) + 0.5) / n, pool_sorted)


def draw_deltas(mode, train_pool, size, rng, rho_s=None, n_epoch=5):
    """(size, n_epoch) per-epoch deviations, mean-centered on the train pool.

    persistent: one draw per galaxy, repeated at every epoch (rho = 1) —
                the adopted exp41 layer.
    ar1:        a Gaussian AR(1) chain rank-mapped onto the SAME empirical
                pool, so every epoch's marginal is identical to persistent
                and only the cross-epoch linkage differs.
    """
    pool = np.sort(np.asarray(train_pool, float)
                   - np.mean(train_pool))
    if mode == "persistent":
        d = pool[rng.integers(0, len(pool), size)]
        return np.repeat(d[:, None], n_epoch, axis=1)
    if mode != "ar1":
        raise ValueError(f"unknown mode {mode!r}")
    rho_g = 2.0 * np.sin(np.pi * float(rho_s) / 6.0)     # copula inversion
    z = np.empty((size, n_epoch))
    z[:, 0] = rng.standard_normal(size)
    s = np.sqrt(max(1.0 - rho_g ** 2, 0.0))
    for k in range(1, n_epoch):
        z[:, k] = rho_g * z[:, k - 1] + s * rng.standard_normal(size)
    return _empirical_quantile(pool, ndtr(z))


def drawn_cogs(gal_idx, deltas):
    """(len(gal_idx), 5, nR) model CoGs with a PER-EPOCH deviation applied
    to the sig coordinate (deltas is (n, 5); a persistent draw simply has
    identical columns, so this one path serves both modes)."""
    s2, e, th0 = _W["s2"], _W["e"], _W["th0"]
    out = np.full((len(gal_idx), 5, len(e.R)), np.nan)
    for i, (gi, d) in enumerate(zip(gal_idx, deltas)):
        g = _W["gals"][gi]
        for k in KS_ALL:
            p = th0.copy()
            p[I_SIG] = np.clip(p[I_SIG] + d[k], LO6[I_SIG], HI6[I_SIG])
            c = s2.model_cogs(p, g, [k], "1ch-mof")
            if c is not None:
                out[i, k] = c[0]
    return out


# --------------------------------------------------------------------------- #
# the held-out plane comparison (exp41's protocol)                             #
# --------------------------------------------------------------------------- #
def _real_one(args):
    from hongshao import qa
    mode, seed, train_pool, score_idx = args
    rng = np.random.default_rng(seed)
    deltas = (np.zeros((len(score_idx), 5)) if mode == "mean"
              else draw_deltas(mode, train_pool, len(score_idx), rng,
                               rho_s=_W["rho_s"]))
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
            res[(kx, ky, j)] = pe["energy_ratio_centered"]
    return mode, res


def cmd_run(dev=False):
    from hongshao import qa
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    tag = "_dev" if dev else ""
    d1 = np.load(E41_DIR / f"outputs/stage1_dist{tag}.npz")
    gals = _W["gals"]
    n = len(gals)
    row_to_i = {g["row"]: i for i, g in enumerate(gals)}
    pool = np.zeros(n)
    for r, s in zip(d1["row"], d1["d_sig_2d"]):
        if int(r) in row_to_i:
            pool[row_to_i[int(r)]] = s
    rng0 = np.random.default_rng(41)                 # exp41's split seed
    perm = rng0.permutation(n)
    halves = (perm[: n // 2], perm[n // 2:])
    jobs, seed = [], 0
    for mode in ("mean",) + MODES:
        for a, b in ((0, 1), (1, 0)):
            for _ in range(1 if mode == "mean" else N_REAL):
                seed += 1
                jobs.append((mode, seed, pool[halves[a]], halves[b]))
    workers = max(os.cpu_count() - 2, 2)
    t0 = time.time()
    print(f"exp44 stage 3 — AR(1) vs persistent layer, held-out planes "
          f"(n={n}{', DEV' if dev else ''}; measured rho_s="
          f"{_W['rho_s']:.3f}; {N_REAL} realizations x 2 directions):",
          flush=True)
    with Pool(workers, initializer=_w_init, initargs=(rows,)) as pool_:
        results = pool_.map(_real_one, jobs)
    agg = {}
    for mode, res in results:
        for k, v in res.items():
            agg.setdefault((mode, *k), []).append(v)
    e = _W["e"]
    print("  centered plane energy / split-half floor (target ~1; exp41 "
          "recorded for the adopted layer on plane 1: mean "
          "1.6/1.8/2.1/2.3/2.9, 1d-emp 1.1/1.4/1.6/1.8/2.6):")
    store = {}
    for kx, ky in qa.PLANES:
        print(f"    plane {kx} vs {ky}:")
        print("      " + " " * 12 + "  ".join(f"z={e.ANCHOR_Z[j]}".rjust(7)
                                              for j in KS_ALL))
        for mode in ("mean",) + MODES:
            cells = []
            for j in KS_ALL:
                v = np.median(agg[(mode, kx, ky, j)])
                store[f"{mode}_{kx}_{ky}_{j}"] = np.array(
                    agg[(mode, kx, ky, j)])
                cells.append(f"{v:7.1f}")
            lab = mode + (" (adopted)" if mode == "persistent" else "")
            print(f"      {lab:<12s}" + "  ".join(cells))
    OUTDIR.mkdir(exist_ok=True)
    np.savez(OUTDIR / f"ar1_planes{tag}.npz", rho_s=_W["rho_s"], **store)
    print(f"  wrote {OUTDIR / f'ar1_planes{tag}.npz'} "
          f"({(time.time()-t0)/60:.1f} min)")


def cmd_physics(dev=False):
    """The two gates an epoch-varying theta specifically threatens:
    mass growth between epochs must stay positive (the drawn object is no
    longer one consistent history), and the differential-deposition test —
    which is INHERENTLY cross-epoch — must stay in the adopted band."""
    from hongshao.profile_emulator import density_from_cog
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    tag = "_dev" if dev else ""
    d1 = np.load(E41_DIR / f"outputs/stage1_dist{tag}.npz")
    gals = _W["gals"]
    e = _W["e"]
    n = len(gals)
    row_to_i = {g["row"]: i for i, g in enumerate(gals)}
    pool = np.zeros(n)
    for r, s in zip(d1["row"], d1["d_sig_2d"]):
        if int(r) in row_to_i:
            pool[row_to_i[int(r)]] = s
    data = np.stack([g["data"] for g in gals])
    logms = np.array([g["logms"] for g in gals])
    print(f"exp44 stage 3 physics gates (n={n}{', DEV' if dev else ''}; "
          "marks: data 0.37/0.11, adopted layer 0.40/0.12, overshoot T1 "
          "+0.027/+0.031):")
    for mode in MODES:
        rng = np.random.default_rng(7)
        deltas = draw_deltas(mode, pool, n, rng, rho_s=_W["rho_s"])
        cogs = drawn_cogs(np.arange(n), deltas)
        ok = np.isfinite(cogs).all(axis=(1, 2))
        # GATE 1 — mass conservation between epochs on the drawn object
        growth = cogs[:, :-1, :] - cogs[:, 1:, :]        # later minus earlier
        frac_neg = float(np.mean(growth[ok] < 0))
        tot_neg = float(np.mean(growth[ok][:, :, -1] < 0))
        # GATE 2 — the cross-epoch physics
        ed3, rows_d = e.differential(cogs, data, logms)
        cells = [f"z{e.ANCHOR_Z[k+1]}->z{e.ANCHOR_Z[k]}: "
                 f"{rows_d[('data', 2, k)][0]:.2f}/"
                 f"{rows_d[('data', 2, k)][1]:.2f} -> "
                 f"{rows_d[('model', 2, k)][0]:.2f}/"
                 f"{rows_d[('model', 2, k)][1]:.2f}" for k in range(4)]
        c0 = cogs[:, 0]
        ok0 = np.isfinite(c0).all(1) & (c0 > 0).all(1)
        ls_d, mid = density_from_cog(np.log10(data[:, 0]), e.R)
        ls_m = np.full_like(ls_d, np.nan)
        ls_m[ok0] = density_from_cog(np.log10(c0[ok0]), e.R)[0]
        dl = ls_m - ls_d
        bands = ((mid >= 30.0) & (mid < 60.0), (mid >= 60.0) & (mid <= 148.0))
        edges = np.quantile(logms, [0, 1 / 3, 2 / 3, 1])
        oc = []
        for b in range(3):
            sel = ok0 & (logms >= edges[b]) & (logms <= edges[b + 1] + 1e-9)
            oc.append(" ".join(f"{np.nanmedian(dl[np.ix_(sel, bm)]):+.3f}"
                               for bm in bands))
        print(f"\n  [{mode}]"
              + ("  (adopted default)" if mode == "persistent" else
                 f"  (rho_s={_W['rho_s']:.2f})"))
        print(f"    GATE 1 mass growth between epochs: "
              f"{100*frac_neg:.2f}% of (galaxy, pair, radius) cells "
              f"negative; {100*tot_neg:.2f}% of TOTAL-mass growths "
              "negative")
        print("    GATE 2 differential: " + "  ".join(cells))
        print("    overshoot terciles (z=0.4, [30-60 / 60-148 kpc]): "
              + "  |  ".join(f"T{b+1} {c}" for b, c in enumerate(oc)))


def cmd_growth(dev=False):
    """The test that can actually SEE rho: a CROSS-EPOCH statistic.

    Every qa tier-2b plane is evaluated at ONE epoch, and the two modes
    share their single-epoch marginals exactly by construction — so those
    planes are structurally blind to rho. The size-growth plane is not:
    log R_half(z=0.4) vs log R_half(z=2.0) for the SAME drawn object asks
    how coherently a galaxy's relative compactness persists, which is
    precisely what rho controls. (The total-MASS growth plane cannot be
    used: every epoch is pinned to the measured M(<500 kpc), so drawn
    totals are data and identical across modes — the exp36 caveat.)"""
    from hongshao import qa
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    tag = "_dev" if dev else ""
    d1 = np.load(E41_DIR / f"outputs/stage1_dist{tag}.npz")
    gals, e = _W["gals"], _W["e"]
    n = len(gals)
    row_to_i = {g["row"]: i for i, g in enumerate(gals)}
    pool = np.zeros(n)
    for r, s_ in zip(d1["row"], d1["d_sig_2d"]):
        if int(r) in row_to_i:
            pool[row_to_i[int(r)]] = s_
    data = np.stack([g["data"] for g in gals])
    T = np.column_stack([
        np.log10([qa.half_mass_radius(data[i, 0], e.R) for i in range(n)]),
        np.log10([qa.half_mass_radius(data[i, 4], e.R) for i in range(n)])])
    print(f"exp44 stage 3 — the CROSS-EPOCH size-growth plane "
          f"(n={n}{', DEV' if dev else ''}; log R_half at z=0.4 vs z=2.0; "
          "centered energy / split-half floor, target ~1):")
    out = {}
    for mode in MODES:
        vals = []
        for s_ in range(N_REAL):
            rng = np.random.default_rng(100 + s_)
            deltas = draw_deltas(mode, pool, n, rng, rho_s=_W["rho_s"])
            cogs = drawn_cogs(np.arange(n), deltas)
            ok = np.isfinite(cogs).all(axis=(1, 2))
            M = np.column_stack([
                np.log10([qa.half_mass_radius(cogs[i, 0], e.R)
                          for i in np.where(ok)[0]]),
                np.log10([qa.half_mass_radius(cogs[i, 4], e.R)
                          for i in np.where(ok)[0]])])
            pe = qa.plane_energy(T[ok], M)
            vals.append([pe["energy_ratio"], pe["energy_ratio_centered"]])
        v = np.median(np.array(vals), axis=0)
        out[mode] = v
        # how coherent is the drawn size evolution vs the truth's?
        rng = np.random.default_rng(100)
        deltas = draw_deltas(mode, pool, n, rng, rho_s=_W["rho_s"])
        cogs = drawn_cogs(np.arange(n), deltas)
        ok = np.isfinite(cogs).all(axis=(1, 2))
        from scipy.stats import spearmanr
        rm = spearmanr(
            [qa.half_mass_radius(cogs[i, 0], e.R) for i in np.where(ok)[0]],
            [qa.half_mass_radius(cogs[i, 4], e.R) for i in np.where(ok)[0]]
        ).statistic
        lab = mode + (" (adopted)" if mode == "persistent" else
                      f" (rho_s={_W['rho_s']:.2f})")
        print(f"    {lab:<24s} energy/floor {v[0]:5.1f} (centered "
              f"{v[1]:5.1f})   size rank-corr z0.4 x z2.0: {rm:+.2f}")
    rt = spearmanr(T[:, 0], T[:, 1]).statistic
    print(f"    {'TRUTH':<24s} {'':>25s}   size rank-corr z0.4 x z2.0: "
          f"{rt:+.2f}")
    np.savez(OUTDIR / f"ar1_growth{tag}.npz",
             **{k: v for k, v in out.items()}, truth_corr=rt)
    print(f"  wrote {OUTDIR / f'ar1_growth{tag}.npz'}")


def cmd_figures(dev=False):
    """The stage-3 verdict in one figure: (a) the cross-epoch size cloud
    where the modes differ, (b) the coherence decay against the truth's,
    (c) the single-epoch plane scores, which are identical because those
    metrics are structurally blind to rho."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hongshao import qa
    from hongshao.plotting import save_fig, set_style
    from hongshao.qa import _tex
    set_style()
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    tag = "_dev" if dev else ""
    d1 = np.load(E41_DIR / f"outputs/stage1_dist{tag}.npz")
    gals, e = _W["gals"], _W["e"]
    n = len(gals)
    row_to_i = {g["row"]: i for i, g in enumerate(gals)}
    pool = np.zeros(n)
    for r, s_ in zip(d1["row"], d1["d_sig_2d"]):
        if int(r) in row_to_i:
            pool[row_to_i[int(r)]] = s_
    data = np.stack([g["data"] for g in gals])
    cols = {"persistent": "#D55E00", "ar1": "#0072B2"}
    growth, sizes = {}, {}
    for mode in MODES:
        rng = np.random.default_rng(100)
        cogs = drawn_cogs(np.arange(n), draw_deltas(mode, pool, n, rng,
                                                    rho_s=_W["rho_s"]))
        growth[mode] = qa.growth_planes(cogs, data, e.R)
        sizes[mode] = qa._safe_rhalf(cogs, e.R)
    rh_t = qa._safe_rhalf(data, e.R)

    FIGDIR = HERE / "figures"
    FIGDIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.3))
    ax = axes[0]
    ax.scatter(np.log10(rh_t[:, 0]), np.log10(rh_t[:, 4]), s=5, c="0.65",
               alpha=0.5, lw=0, label="truth")
    for mode in MODES:
        ax.scatter(np.log10(sizes[mode][:, 0]), np.log10(sizes[mode][:, 4]),
                   s=5, c=cols[mode], alpha=0.5, lw=0, label=mode)
    ax.set(xlabel=_tex(r"$\log_{10} R_{\rm half}$ at $z=0.4$ [kpc]"),
           ylabel=_tex(r"$\log_{10} R_{\rm half}$ at $z=2.0$ [kpc]"),
           title="cross-epoch size cloud")
    ax.legend(fontsize=8, markerscale=2.5)

    bx = axes[1]
    zs = [e.ANCHOR_Z[j] for j in KS_ALL[1:]]
    ct = [growth["persistent"][("R_half", 0, j)]["coherence_truth"]
          for j in KS_ALL[1:]]
    bx.plot(zs, ct, "-s", color="0.2", lw=2.2, ms=7, label="truth")
    for mode in MODES:
        cm = [growth[mode][("R_half", 0, j)]["coherence_model"]
              for j in KS_ALL[1:]]
        bx.plot(zs, cm, "-o", color=cols[mode], lw=1.8, ms=6, label=mode)
    bx.set(xlabel=r"later epoch $z$", ylim=(0, 1.05),
           ylabel=r"size rank coherence vs $z=0.4$",
           title="how loyal a galaxy stays to its size")
    bx.legend(fontsize=8)

    cx = axes[2]
    pl = np.load(OUTDIR / f"ar1_planes{tag}.npz")
    kx, ky = qa.PLANES[1]
    zall = [e.ANCHOR_Z[j] for j in KS_ALL]
    for mode in ("mean",) + MODES:
        y = [np.median(pl[f"{mode}_{kx}_{ky}_{j}"]) for j in KS_ALL]
        cx.plot(zall, y, "-o", lw=1.8, ms=6,
                color="0.2" if mode == "mean" else cols[mode], label=mode)
    cx.axhline(1.0, color="0.8", lw=1.0, ls="--", zorder=0)
    cx.set(xlabel=r"epoch $z$", ylabel="centered energy / split-half floor",
           title=_tex(f"single-epoch plane ({kx} vs {ky})"))
    cx.text(0.04, 0.06, "the two layers overlap: single-epoch\nmetrics "
            "cannot see " + r"$\rho$", transform=cx.transAxes, fontsize=8,
            color="0.35")
    cx.legend(fontsize=8)
    fig.suptitle("exp44 stage 3 — the AR(1) layer fixes cross-epoch "
                 "coherence, which no single-epoch metric scores",
                 fontsize=11)
    fig.tight_layout()
    print("wrote", save_fig(fig, FIGDIR / f"exp44_ar1_verdict{tag}")[0])
    plt.close(fig)


def demo():
    rows = np.load(POP_NPZ)["dev100"][:10]
    _w_init(rows)
    rng = np.random.default_rng(3)
    pool = np.load(E41_DIR / "outputs/stage1_dist.npz")["d_sig_2d"]
    rho_s = _W["rho_s"]
    big = 40000
    # (1) the AR(1) marginal at EVERY epoch equals the persistent marginal
    # (that is the whole point of the rank mapping)
    da = draw_deltas("ar1", pool, big, rng, rho_s=rho_s)
    dp = draw_deltas("persistent", pool, big, rng)
    for k in range(5):
        for q in (5, 25, 50, 75, 95):
            a, b = np.percentile(da[:, k], q), np.percentile(dp[:, k], q)
            assert abs(a - b) < 0.01, (k, q, a, b)
    # (2) the realized RANK correlation hits the measured target, and
    # decays as rho^sep (the copula inversion is doing its job)
    from scipy.stats import spearmanr
    C = spearmanr(da).statistic
    adj = np.mean([C[k, k + 1] for k in range(4)])
    assert abs(adj - rho_s) < 0.02, (adj, rho_s)
    assert abs(C[0, 4] - rho_s ** 4) < 0.03, (C[0, 4], rho_s ** 4)
    # (3) rho -> 1 nests the persistent mode (same rank ordering at every
    # epoch), rho -> 0 gives independence
    d1_ = draw_deltas("ar1", pool, 8000, rng, rho_s=0.999)
    assert np.mean([spearmanr(d1_[:, 0], d1_[:, k]).statistic
                    for k in range(1, 5)]) > 0.99
    d0 = draw_deltas("ar1", pool, 8000, rng, rho_s=0.0)
    assert abs(spearmanr(d0[:, 0], d0[:, 4]).statistic) < 0.05
    # (4) persistent draws are constant across epochs by construction
    assert np.allclose(dp, dp[:, :1])
    # (5) the per-epoch CoG path reproduces the all-epoch call when the
    # deviation is constant — so 'persistent' here IS the exp41 layer
    s2, th0 = _W["s2"], _W["th0"]
    dv = np.full((3, 5), 0.05)
    got = drawn_cogs(np.arange(3), dv)
    for i in range(3):
        p = th0.copy()
        p[I_SIG] = np.clip(p[I_SIG] + 0.05, LO6[I_SIG], HI6[I_SIG])
        ref = s2.model_cogs(p, _W["gals"][i], KS_ALL, "1ch-mof")
        assert np.abs(got[i] / np.asarray(ref) - 1.0).max() < 1e-12
    # (6) drawn CoGs are finite and monotone in radius in BOTH modes
    for mode in MODES:
        dd = draw_deltas(mode, pool, 3, rng, rho_s=rho_s)
        c = drawn_cogs(np.arange(3), dd)
        assert np.isfinite(c).all() and (np.diff(c, axis=2) > -1e-9).all()
    print("exp44 stage3 demo OK: the AR(1) marginal is identical to the "
          "adopted layer's at every epoch (rank mapping preserves the "
          "heavy tails); the realized rank correlation hits the measured "
          "target and decays as rho^sep; rho->1 nests the persistent "
          "mode and rho->0 gives independence; the per-epoch CoG path "
          "reproduces the exp41 all-epoch call exactly for a constant "
          "deviation; draws finite and monotone in both modes")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    dev = "--dev" in sys.argv
    if cmd == "demo":
        demo()
    elif cmd == "run":
        cmd_run(dev)
    elif cmd == "physics":
        cmd_physics(dev)
    elif cmd == "growth":
        cmd_growth(dev)
    elif cmd == "figures":
        cmd_figures(dev)
    else:
        sys.exit(f"unknown subcommand {cmd!r}")
