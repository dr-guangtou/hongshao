"""exp44 stage 2 — how fast does the per-galaxy individuality decorrelate
across epochs?

exp44 stage 1 measured that low-z anchoring transfers about ONE epoch.
Two very different mechanisms produce that decay, and stage 1 cannot
tell them apart:

  (i) the individuality genuinely EVOLVES — the per-galaxy correction
      needed at z=0.4 is not the one needed at z=1.5, in which case no
      amount of low-z data extends the reach; or
  (ii) the individuality is PERSISTENT but our single-epoch ESTIMATE of
      it is noisy, so it transfers poorly — in which case better
      anchoring (more radii, more epochs, better observables) buys
      reach directly.

This script separates them. Per galaxy, the same single coordinate is
fitted INDEPENDENTLY at each of the five epochs, giving a per-galaxy
vector; the cross-epoch correlation matrix of those vectors is the
decorrelation curve. To remove mechanism (ii) it also fits each epoch
on two disjoint halves of the radial grid (odd/even radii): the
within-epoch correlation between the two halves measures the RELIABILITY
of a single-epoch estimate (Spearman-Brown corrected to full length),
and dividing the cross-epoch correlations by sqrt(rel_a * rel_b) — the
standard disattenuation — recovers the correlation the underlying
quantity would have if measured without noise.

Readouts: the raw and disattenuated correlation matrices, an AR(1) decay
constant rho fitted with the SAME estimator exp37 used for the
statistical emulator (`hongshao.multi_epoch.fit_rho`, so the two numbers
are directly comparable — exp37 measured rho = 0.62), and the
population-level drift of the fitted values with epoch (a systematic
drift indicates the BASE theta is epoch-misspecified, which is a mean-
model issue, not individuality).

Run: PYTHONPATH=. uv run python experiments/exp44_lowz_anchoring/\
decorrelation.py {demo|fit|report|figures} [--dev]
"""
import importlib.util
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
E38_DIR = ROOT / "experiments/exp38_deposit_rethink"
E40_DIR = ROOT / "experiments/exp40_epoch_objective"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
KS_ALL = [0, 1, 2, 3, 4]
AXES = {"sig": 4, "rc": 0}
NAMES = ["log_rc", "g", "q", "mu", "sig", "gamma"]
LO6 = np.array([1.0, 0.0, 0.0, 0.0, 0.05, 1.06])
HI6 = np.array([3.0, 6.0, 3.0, 3.0, 2.0, 6.0])
HALVES = ("all", "odd", "even")
RHO_EXP37 = 0.62                 # the statistical emulator's measured AR(1)
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
    nr = len(_W["e"].R)
    _W["ridx"] = {"all": np.arange(nr), "odd": np.arange(1, nr, 2),
                  "even": np.arange(0, nr, 2)}


def gal_loss_radii(th, g, k, ridx):
    """Single-epoch relative-RMS loss restricted to a radius subset —
    the exp38 gal_loss restricted to `ridx` (identical on the full set)."""
    cogs = _W["s2"].model_cogs(th, g, [k], "1ch-mof")
    if cogs is None:
        return 4.0
    c, d = cogs[0][ridx], g["data"][k][ridx]
    return float(np.sqrt(np.mean(((c - d) / d) ** 2)))


def _fit_epoch_one(args):
    """One galaxy: the axis fitted independently at every epoch, on the
    full radial grid and on each disjoint half."""
    gi, axis = args
    j = AXES[axis]
    g = _W["gals"][gi]
    th0 = _W["th0"]
    out = np.full((len(HALVES), 5), np.nan)
    moved = np.zeros((len(HALVES), 5), bool)
    for hi, half in enumerate(HALVES):
        ridx = _W["ridx"][half]
        for k in KS_ALL:
            l_base = gal_loss_radii(th0, g, k, ridx)

            def f(x):
                th = th0.copy()
                th[j] = x
                return gal_loss_radii(th, g, k, ridx)

            r = minimize_scalar(f, bounds=(LO6[j], HI6[j]),
                                method="bounded",
                                options=dict(maxiter=80, xatol=1e-4))
            if r.fun < l_base:
                out[hi, k], moved[hi, k] = r.x, True
            else:
                out[hi, k] = th0[j]
    return gi, out, moved


def cmd_fit(dev=False):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    tag = "_dev" if dev else ""
    n = len(_W["gals"])
    workers = max(os.cpu_count() - 2, 2)
    OUTDIR.mkdir(exist_ok=True)
    store = {}
    print(f"exp44 stage 2 — per-epoch independent axis fits (n={n}"
          f"{', DEV' if dev else ''}; 5 epochs x {{all,odd,even}} radii):",
          flush=True)
    with Pool(workers, initializer=_w_init, initargs=(rows,)) as pool:
        for axis in AXES:
            t0 = time.time()
            res = pool.map(_fit_epoch_one, [(i, axis) for i in range(n)])
            vals = np.full((n, len(HALVES), 5), np.nan)
            moved = np.zeros((n, len(HALVES), 5), bool)
            for gi, v, m in res:
                vals[gi], moved[gi] = v, m
            store[f"vals_{axis}"] = vals
            store[f"moved_{axis}"] = moved
            j = AXES[axis]
            med = np.median(vals[:, 0, :], axis=0) - _W["th0"][j]
            print(f"  [{axis}] median delta {NAMES[j]} by epoch: "
                  + " ".join(f"{v:+.3f}" for v in med)
                  + f"  | refit moved in {100*moved[:, 0].mean():.0f}% of "
                  f"(galaxy, epoch) cells  ({(time.time()-t0)/60:.1f} min)",
                  flush=True)
    np.savez(OUTDIR / f"decorr{tag}.npz", th0=_W["th0"], **store)
    print(f"  wrote {OUTDIR / f'decorr{tag}.npz'}")


# --------------------------------------------------------------------------- #
# correlation, reliability, disattenuation                                     #
# --------------------------------------------------------------------------- #
def _corr_matrix(v, kind):
    """(n, 5) -> (5, 5) correlation matrix; kind in {pearson, spearman}."""
    ok = np.isfinite(v).all(1)
    if kind == "spearman":
        return spearmanr(v[ok]).statistic
    return np.corrcoef(v[ok].T)


def _reliability(v_odd, v_even, kind):
    """Split-half correlation per epoch, Spearman-Brown corrected to the
    reliability of the FULL-grid estimate: rel = 2 r / (1 + r)."""
    rel = np.empty(5)
    for k in range(5):
        ok = np.isfinite(v_odd[:, k]) & np.isfinite(v_even[:, k])
        if kind == "spearman":
            r = spearmanr(v_odd[ok, k], v_even[ok, k]).statistic
        else:
            r = np.corrcoef(v_odd[ok, k], v_even[ok, k])[0, 1]
        rel[k] = 2.0 * r / (1.0 + r) if r > -1 else np.nan
    return rel


def cmd_report(dev=False):
    from hongshao.multi_epoch import fit_rho
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    tag = "_dev" if dev else ""
    d = np.load(OUTDIR / f"decorr{tag}.npz")
    e = _W["e"]
    zs = [e.ANCHOR_Z[k] for k in KS_ALL]
    store = {}
    print(f"exp44 stage 2 report (n={len(_W['gals'])}"
          f"{', DEV' if dev else ''}):")
    for axis in AXES:
        vals = d[f"vals_{axis}"]
        v_all, v_odd, v_even = vals[:, 0], vals[:, 1], vals[:, 2]
        print(f"\n  ===== axis {NAMES[AXES[axis]]} =====")
        for kind in ("spearman", "pearson"):
            C = _corr_matrix(v_all, kind)
            rel = _reliability(v_odd, v_even, kind)
            Cd = C / np.sqrt(np.outer(rel, rel))
            np.fill_diagonal(Cd, 1.0)
            store[f"C_{axis}_{kind}"] = C
            store[f"Cd_{axis}_{kind}"] = Cd
            store[f"rel_{axis}_{kind}"] = rel
            print(f"\n    [{kind}] single-epoch estimate RELIABILITY "
                  "(split-half, Spearman-Brown to full grid; 1 = noiseless):")
            print("      " + "  ".join(f"z={z:g}: {r:.2f}"
                                       for z, r in zip(zs, rel)))
            print(f"    [{kind}] cross-epoch correlation "
                  "(raw | disattenuated):")
            for a in range(5):
                cells = "  ".join(f"{C[a, b]:+.2f}|{min(Cd[a, b], 1.0):+.2f}"
                                  for b in range(5))
                print(f"      z={zs[a]:<4g} {cells}")
            rho_raw = fit_rho(np.clip(C, 1e-6, None))
            rho_dis = fit_rho(np.clip(np.minimum(Cd, 1.0), 1e-6, None))
            store[f"rho_{axis}_{kind}"] = np.array([rho_raw, rho_dis])
            print(f"    [{kind}] AR(1) decay rho: raw {rho_raw:.3f} -> "
                  f"disattenuated {rho_dis:.3f}   "
                  f"(exp37 statistical emulator: {RHO_EXP37})")
        # adjacent vs distant, the headline pair
        C = store[f"C_{axis}_spearman"]
        Cd = np.minimum(store[f"Cd_{axis}_spearman"], 1.0)
        adj = np.mean([C[k, k + 1] for k in range(4)])
        adj_d = np.mean([Cd[k, k + 1] for k in range(4)])
        print(f"\n    adjacent-epoch correlation: raw {adj:+.2f} -> "
              f"disattenuated {adj_d:+.2f};  z=0.4 vs z=2.0: raw "
              f"{C[0, 4]:+.2f} -> disattenuated {Cd[0, 4]:+.2f}")
    np.savez(OUTDIR / f"decorr_report{tag}.npz", **store)
    print(f"\n  wrote {OUTDIR / f'decorr_report{tag}.npz'}")


def cmd_figures(dev=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hongshao.plotting import save_fig, set_style
    set_style()
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    tag = "_dev" if dev else ""
    r = np.load(OUTDIR / f"decorr_report{tag}.npz")
    e = _W["e"]
    zs = np.array([e.ANCHOR_Z[k] for k in KS_ALL])
    FIGDIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1),
                             gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    for ax, key, ttl in ((axes[0], "C_sig_spearman", "raw"),
                         (axes[1], "Cd_sig_spearman", "disattenuated")):
        M = np.minimum(r[key], 1.0)
        im = ax.imshow(M, cmap="cividis", vmin=0, vmax=1)
        for a in range(5):
            for b in range(5):
                ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center",
                        fontsize=8, color="w" if M[a, b] < 0.55 else "k")
        ax.set_xticks(range(5), [f"{z:g}" for z in zs])
        ax.set_yticks(range(5), [f"{z:g}" for z in zs])
        ax.set(xlabel=r"$z$", ylabel=r"$z$",
               title=f"individuality correlation ({ttl})")
    fig.colorbar(im, ax=axes[1], shrink=0.85)
    bx = axes[2]
    sep = np.arange(5)
    for key, color, lab in (("C_sig_spearman", "#0072B2", "raw"),
                            ("Cd_sig_spearman", "#D55E00",
                             "disattenuated")):
        M = np.minimum(r[key], 1.0)
        y = [np.mean([M[a, a + s] for a in range(5 - s)]) for s in sep]
        bx.plot(sep, y, "-o", color=color, lw=1.8, ms=6, label=lab)
        rho = r[key.replace("C_", "rho_").replace("Cd_", "rho_")][
            0 if key.startswith("C_") else 1]
        bx.plot(sep, rho ** sep, "--", color=color, lw=1.2,
                label=rf"AR(1) $\rho={rho:.2f}$")
    bx.plot(sep, RHO_EXP37 ** sep, ":", color="0.35", lw=1.6,
            label=rf"exp37 emulator $\rho={RHO_EXP37}$")
    bx.set(xlabel="epoch separation [snapshots]",
           ylabel="mean correlation", ylim=(0, 1.05),
           title="how fast individuality decorrelates")
    bx.legend(fontsize=8)
    fig.suptitle("exp44 stage 2 — is the per-galaxy axis persistent "
                 "(and noisily measured) or genuinely epoch-local?",
                 fontsize=11)
    fig.tight_layout()
    print("wrote", save_fig(fig, FIGDIR / f"exp44_decorrelation{tag}")[0])
    plt.close(fig)


def demo():
    rows = np.load(POP_NPZ)["dev100"][:8]
    _w_init(rows)
    s2, th0 = _W["s2"], _W["th0"]
    g = _W["gals"][0]
    nr = len(_W["e"].R)
    # (1) the radius halves are disjoint and exhaust the grid
    odd, even = _W["ridx"]["odd"], _W["ridx"]["even"]
    assert len(np.intersect1d(odd, even)) == 0
    assert len(odd) + len(even) == nr
    # (2) the masked loss on the FULL grid == the exp38 single-epoch loss
    for k in (0, 3):
        mine = gal_loss_radii(th0, g, k, _W["ridx"]["all"])
        theirs = s2.gal_loss(th0, g, [k], "1ch-mof")
        assert abs(mine - theirs) < 1e-12, (k, mine, theirs)
    # (3) the per-epoch refit never worsens its own epoch's loss, stays
    # in the box, and the halves really do give independent estimates
    gi, vals, moved = _fit_epoch_one((0, "sig"))
    j = AXES["sig"]
    assert np.isfinite(vals).all()
    assert np.all(vals >= LO6[j] - 1e-9) and np.all(vals <= HI6[j] + 1e-9)
    for k in KS_ALL:
        th = th0.copy()
        th[j] = vals[0, k]
        assert gal_loss_radii(th, g, k, _W["ridx"]["all"]) <= \
            gal_loss_radii(th0, g, k, _W["ridx"]["all"]) + 1e-12
    # (4) CROSS-SCRIPT consistency: the k=0 full-grid fit here is exactly
    # the quantity exp44 stage 1 stored as the z04 anchoring arm
    a1 = _load_by_path("exp44_anchor", HERE / "anchor.py")
    a1._W.update(_W)
    _, _, val_stage1, _, _ = a1._fit_one((0, "sig_z04_prof"))
    assert abs(val_stage1 - vals[0, 0]) < 1e-6, (val_stage1, vals[0, 0])
    print("exp44 stage2 demo OK: radius halves disjoint and exhaustive; "
          "the masked full-grid loss reproduces the exp38 single-epoch "
          "loss exactly; per-epoch refits are non-worsening and in-box; "
          "and the k=0 fit reproduces exp44 stage 1's z04 anchoring "
          "value (cross-script consistency)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    dev = "--dev" in sys.argv
    if cmd == "demo":
        demo()
    elif cmd == "fit":
        cmd_fit(dev)
    elif cmd == "report":
        cmd_report(dev)
    elif cmd == "figures":
        cmd_figures(dev)
    else:
        sys.exit(f"unknown subcommand {cmd!r}")
