"""exp45 — retro-score the model zoo on qa tier 2c (cross-epoch coherence).

Pure evaluation, NO fitting: every parameter vector is already on disk
(exp38/40/42/43), plus the exp41/exp44 stochastic-layer modes and — as a
cross-FAMILY reference — the exp37 statistical emulator's saved draws.
For each, build model CoGs and run ``qa.growth_planes``.

The question (exp44): the deterministic kernel holds galaxies at size
rank coherence +1.00/+1.00/+1.00/+0.96 against a truth of
+0.82/+0.68/+0.50/+0.33. Is that universal to the family, or does some
structural choice — channel count, fit scope, objective, MAH input, or
an AR(1) latent — already evolve sizes more realistically?

Run: PYTHONPATH=. uv run python experiments/exp45_coherence_scoreboard/\
scoreboard.py {demo|run|figures} [--dev]
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
E41 = ROOT / "experiments/exp41_stochastic_layer/outputs/stage1_dist.npz"
E42 = ROOT / "experiments/exp42_real_mah/outputs/realfit.npz"
E43 = ROOT / "experiments/exp43_extrapolation/outputs/ladder.npz"
E44 = ROOT / "experiments/exp44_lowz_anchoring"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
KS_ALL = [0, 1, 2, 3, 4]
_W = {}


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _w_init(rows, config="diffmah"):
    """Worker state. ``config='real'`` swaps the loader to the de-dipped
    real MAH (the exp42 pattern) so that row is scored on its own input."""
    s2 = _load_by_path("exp38_stage2", E38.parent / "stage2_multiepoch.py")
    if config == "diffmah":
        s2._w_init(rows)
    else:
        e = _load_by_path("exp35_run",
                          ROOT / "experiments/exp35_total_norm/run.py")
        mh_scale = tuple(np.load(
            ROOT / "experiments/exp32_full_population/outputs/"
            "um_slope_diffmah.npz")["mh_scale"])
        e._init(config, mh_scale, None if rows is None else np.asarray(rows))
        pop = np.load(POP_NPZ)
        z_all = np.column_stack([pop["logmh"], pop["c200c"], pop["fz2"]])
        mu, sd = np.nanmean(z_all, 0), np.nanstd(z_all, 0)
        for g in e._G["gals"]:
            z = (z_all[g["row"]] - mu) / sd
            g["cond"] = np.where(np.isfinite(z), z, 0.0)
        s2._W["e"], s2._W["gals"] = e, e._G["gals"]
    _W.update(s2._W)
    _W["s2"] = s2


# --------------------------------------------------------------------------- #
# the zoo                                                                      #
# --------------------------------------------------------------------------- #
def zoo():
    """[(label, variant, theta, axis)] for every recorded kernel fit."""
    d38 = np.load(E38 / "stage2_multiepoch.npz")
    d38s = np.load(E38 / "stage3_single.npz")
    d38c = np.load(E38 / "stage3_capacity.npz")
    d40, d43 = np.load(E40), np.load(E43)
    out = [
        ("1ch z20", "1ch-mof", d38["theta_1ch-mof"], "scope"),
        ("2ch z20", "2ch-exp", d38["theta_2ch-exp"], "channels"),
        ("1ch z15 ADOPTED", "1ch-mof", d40["theta_z15"], "scope"),
        ("1ch z10", "1ch-mof", d40["theta_z10"], "scope"),
        ("1ch z07", "1ch-mof", d43["theta_1ch-mof_z07"], "scope"),
        ("1ch z04-only", "1ch-mof", d38s["theta_1ch-mof_k0"], "scope"),
        ("1ch z2.0-only", "1ch-mof", d38s["theta_1ch-mof_k4"], "scope"),
        ("2ch z15", "2ch-exp", d43["theta_2ch-exp_z15"], "channels"),
        ("2ch z10", "2ch-exp", d43["theta_2ch-exp_z10"], "channels"),
        ("2ch z07", "2ch-exp", d43["theta_2ch-exp_z07"], "channels"),
        ("2ch z04-only", "2ch-exp", d38s["theta_2ch-exp_k0"], "channels"),
        ("1ch inner-aware", "1ch-mof", d38c["theta_inner_1ch-mof"],
         "objective"),
        ("2ch inner-aware", "2ch-exp", d38c["theta_inner_2ch-exp"],
         "objective"),
    ]
    return out


def _cogs_one(args):
    label, variant, theta = args
    s2, e, gals = _W["s2"], _W["e"], _W["gals"]
    out = np.full((len(gals), 5, len(e.R)), np.nan)
    for i, g in enumerate(gals):
        c = s2.model_cogs(theta, g, KS_ALL, variant)
        if c is not None:
            out[i] = c
    return label, out


def _score(cogs, data, R):
    from hongshao import qa
    g = qa.growth_planes(cogs, data, R)
    coh = [g[("R_half", 0, j)]["coherence_model"] for j in KS_ALL[1:]]
    ct = [g[("R_half", 0, j)]["coherence_truth"] for j in KS_ALL[1:]]
    en = [g[("R_half", 0, j)]["energy_ratio_centered"] for j in KS_ALL[1:]]
    return np.array(coh), np.array(ct), np.array(en)


def cmd_run(dev=False):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    tag = "_dev" if dev else ""
    _w_init(rows)
    e = _W["e"]
    R = e.R
    data = np.stack([g["data"] for g in _W["gals"]])
    workers = max(os.cpu_count() - 2, 2)
    t0 = time.time()
    OUTDIR.mkdir(exist_ok=True)
    res = {}
    print(f"exp45 — cross-epoch coherence scoreboard "
          f"(n={len(_W['gals'])}{', DEV' if dev else ''}; size rank "
          "coherence z=0.4 vs later, EXCESS over truth in brackets):",
          flush=True)
    entries = zoo()
    with Pool(workers, initializer=_w_init, initargs=(rows,)) as pool:
        cogs_all = dict(pool.map(_cogs_one,
                                 [(la, va, th) for la, va, th, _ in entries]))
    truth_coh = None
    for label, variant, theta, axis in entries:
        coh, ct, en = _score(cogs_all[label], data, R)
        truth_coh = ct
        res[label] = np.stack([coh, en])
        print(f"    {label:<18s} " + "  ".join(
            f"{c:+.2f}[{c-t:+.2f}]" for c, t in zip(coh, ct))
            + f"   | E/floor {np.mean(en):.1f}   ({axis})")

    # --- the real-MAH row: its own loader ---
    if E42.exists():
        _w_init(rows, config="real")
        data_r = np.stack([g["data"] for g in _W["gals"]])
        _, cg = _cogs_one(("1ch z15 REAL-MAH", "1ch-mof",
                           np.load(E42)["theta_real_z15"]))
        coh, ct, en = _score(cg, data_r, _W["e"].R)
        res["1ch z15 REAL-MAH"] = np.stack([coh, en])
        print(f"    {'1ch z15 REAL-MAH':<18s} " + "  ".join(
            f"{c:+.2f}[{c-t:+.2f}]" for c, t in zip(coh, ct))
            + f"   | E/floor {np.mean(en):.1f}   (MAH input)")
        _w_init(rows)

    # --- the stochastic-layer rows (exp44 modes on the adopted theta) ---
    a1 = _load_by_path("exp44_ar1", E44 / "ar1_layer.py")
    a1._W.update(_W)
    a1._W["th0"] = np.load(E40)["theta_z15"]
    a1._W["rho_s"] = float(np.load(E44 / "outputs/decorr_report.npz")
                           ["rho_sig_spearman"][0])
    d1 = np.load(E41)
    row_to_i = {g["row"]: i for i, g in enumerate(_W["gals"])}
    pool_v = np.zeros(len(_W["gals"]))
    for r, s in zip(d1["row"], d1["d_sig_2d"]):
        if int(r) in row_to_i:
            pool_v[row_to_i[int(r)]] = s
    for mode in ("persistent", "ar1"):
        rng = np.random.default_rng(100)
        dl = a1.draw_deltas(mode, pool_v, len(_W["gals"]), rng,
                            rho_s=a1._W["rho_s"])
        cg = a1.drawn_cogs(np.arange(len(_W["gals"])), dl)
        coh, ct, en = _score(cg, data, R)
        res[f"+layer {mode}"] = np.stack([coh, en])
        print(f"    {'+layer ' + mode:<18s} " + "  ".join(
            f"{c:+.2f}[{c-t:+.2f}]" for c, t in zip(coh, ct))
            + f"   | E/floor {np.mean(en):.1f}   (scatter layer)")

    # --- CROSS-FAMILY: the exp37 statistical emulator's saved draws ---
    if E37.exists() and not dev:
        pop = np.load(POP_NPZ)
        dall = pop["data"]
        lt = np.log10(np.clip(dall[:, :, -1], 1.0, None))
        keep = ~((lt[:, 0][:, None] - lt).max(axis=1) > 2.0)
        assert int(keep.sum()) == 2395, f"exp37 mask mismatch: {keep.sum()}"
        draws = np.load(E37)["cog_draws"]
        cg = 10.0 ** draws[0] if np.nanmax(draws[0]) < 100 else draws[0]
        coh, ct, en = _score(cg, dall[keep], R)
        res["exp37 emulator"] = np.stack([coh, en])
        print(f"    {'exp37 emulator':<18s} " + "  ".join(
            f"{c:+.2f}[{c-t:+.2f}]" for c, t in zip(coh, ct))
            + f"   | E/floor {np.mean(en):.1f}   (CROSS-FAMILY, AR(1) by design)")

    print(f"\n    {'TRUTH':<18s} " + "  ".join(f"{t:+.2f}" for t in truth_coh))
    np.savez(OUTDIR / f"scoreboard{tag}.npz", truth=truth_coh, **res)
    print(f"  wrote {OUTDIR / f'scoreboard{tag}.npz'} "
          f"({(time.time()-t0)/60:.1f} min)")


def cmd_figures(dev=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hongshao.plotting import save_fig, set_style
    set_style()
    tag = "_dev" if dev else ""
    d = np.load(OUTDIR / f"scoreboard{tag}.npz")
    truth = d["truth"]
    labels = [k for k in d.files if k != "truth"]
    zs = np.array([0.7, 1.0, 1.5, 2.0])
    FIGDIR.mkdir(exist_ok=True)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.4, 4.6))
    cmap = plt.get_cmap("viridis")
    for i, la in enumerate(labels):
        c = cmap(i / max(len(labels) - 1, 1))
        style = "-o" if "ADOPTED" in la or "exp37" in la else "-"
        lw = 2.4 if ("ADOPTED" in la or "exp37" in la) else 1.2
        ax.plot(zs, d[la][0], style, color=c, lw=lw, ms=5, label=la,
                alpha=0.9)
    ax.plot(zs, truth, "-s", color="k", lw=3.0, ms=8, label="TRUTH", zorder=5)
    ax.set(xlabel=r"later epoch $z$", ylabel=r"size rank coherence vs $z=0.4$",
           ylim=(0, 1.05), title="every model is too coherent — by how much?")
    ax.legend(fontsize=6.5, ncol=2, loc="lower left")
    exc = [float(d[la][0][-1] - truth[-1]) for la in labels]
    order = np.argsort(exc)
    bx.barh(range(len(labels)), [exc[i] for i in order],
            color=["#0072B2" if "exp37" in labels[i] else
                   ("#D55E00" if "ADOPTED" in labels[i] else "0.6")
                   for i in order])
    bx.set_yticks(range(len(labels)),
                  [labels[i] for i in order], fontsize=7)
    bx.axvline(0.0, color="k", lw=1.2)
    bx.set(xlabel="coherence EXCESS over truth at " r"$z=0.4$ vs $z=2.0$",
           title="0 = realistic; positive = too loyal")
    fig.suptitle("exp45 — cross-epoch size coherence across the model zoo",
                 fontsize=11)
    fig.tight_layout()
    print("wrote", save_fig(fig, FIGDIR / f"exp45_scoreboard{tag}")[0])
    plt.close(fig)


def demo():
    rows = np.load(POP_NPZ)["dev100"][:12]
    _w_init(rows)
    s2 = _W["s2"]
    entries = zoo()
    # (1) every recorded theta loads with the right layout for its variant
    for label, variant, th, _ in entries:
        want = 12 if variant == "1ch-mof" else 16
        assert th.shape == (want,), (label, th.shape)
    assert len(entries) >= 12, len(entries)
    # (2) each runs finite and monotone on a real galaxy
    g = _W["gals"][0]
    for label, variant, th, _ in entries:
        c = s2.model_cogs(th, g, KS_ALL, variant)
        assert c is not None, label
        for ck in c:
            assert np.isfinite(ck).all() and np.all(np.diff(ck) > -1e-9), label
    # (3) the exp37 mask reconstruction is EXACT (its documented indices)
    pop = np.load(POP_NPZ)
    lt = np.log10(np.clip(pop["data"][:, :, -1], 1.0, None))
    drop = np.where((lt[:, 0][:, None] - lt).max(axis=1) > 2.0)[0]
    assert list(drop) == [181, 2372], drop
    # (4) the real-MAH loader yields the same galaxy count on these rows
    _w_init(rows, config="real")
    assert len(_W["gals"]) == 12, len(_W["gals"])
    _w_init(rows)
    print("exp45 demo OK: all 13 recorded thetas load with the right layout "
          "and run finite/monotone; the exp37 progenitor mask reconstructs "
          "to exactly [181, 2372] (n=2395, its documented value); the "
          "real-MAH loader returns the same galaxies")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    dev = "--dev" in sys.argv
    if cmd == "demo":
        demo()
    elif cmd == "run":
        cmd_run(dev)
    elif cmd == "figures":
        cmd_figures(dev)
    else:
        sys.exit(f"unknown subcommand {cmd!r}")
