"""exp36 (C) — the statistical dressing on the two-channel kernel spine.

The multi-epoch 2ch-fa kernel CoG (held-out, from the exp36 10-fold CV) is
the MEAN; on top sits an exp37-style compressed residual layer: pooled-PCA
modes of the log-CoG residual (one basis over all five epochs), per-epoch
heteroscedastic cores (`hongshao.emulator`) predicting the mode scores from
the portable features [DiffMAH(4), c200c], and an AR(1)-in-epoch latent for
generative draws. The spine carries the physics (differential deposition,
mass conservation of the mean); the dressing pays the consistency tax and
makes the model generative.

ABLATION GUARD (pre-registered, README §5): the same dressing machinery on a
FLAT spine (per-epoch train-median log CoG — no kernel, no per-galaxy
physics) must do WORSE on the held-out shape and the planes, else the spine
earns nothing.

Run:  PYTHONPATH=. uv run python experiments/exp36_two_channel/dressing.py \
        {demo|fit|qa} [--dev] [--k 4]
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from hongshao.emulator import fit as emu_fit                          # noqa: E402
from hongshao.multi_epoch import _sample_eps, fit_rho                 # noqa: E402

OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
TABLE = ROOT / "data/processed/tng300_072_z0p4.fits"
FEATS = ("dmah_logmp", "dmah_logtc", "dmah_early", "dmah_late", "c_200c")
ZK = [0.4, 0.7, 1.0, 1.5, 2.0]
SPINE_KEY = "cogs_multi_2ch-fa"
KFOLD, SEED = 5, 0
N_DRAW = 3
K_DEFAULT = 4
MARKS = {"z04 2ch-prune": 16.4, "statistical wall": 15.6}


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def folds_of(n):
    order = np.random.default_rng(SEED).permutation(n)
    return np.array_split(order, KFOLD)


def load(dev=False):
    """Aligned (X, data, spine, logmh): exp36 gal order; the spine is the
    HELD-OUT multi 2ch-fa CoG from the exp36 CV (fold-refit theta per galaxy)."""
    from astropy.table import Table
    e36 = _load_by_path("exp36_run", HERE / "run.py")
    e36._w_init(None)
    gals = e36._W["gals"]
    d = np.load(HERE / "outputs/two_channel.npz")
    spine = d[SPINE_KEY]                                    # (n, 5, 24) linear
    data = np.stack([g["data"] for g in gals])
    logmh = np.array([g["logmh"] for g in gals])
    pop = np.load(POP_NPZ)
    t = Table.read(TABLE)
    trow = {int(g): i for i, g in enumerate(np.asarray(t["index"]))}
    rows = np.array([g["row"] for g in gals])
    X = np.array([[t[c][trow[int(pop["index"][r])]] for c in FEATS]
                  for r in rows], float)
    ok = (np.isfinite(X).all(1) & np.isfinite(spine).all(axis=(1, 2))
          & (spine > 0).all(axis=(1, 2)))
    if (~ok).sum():
        print(f"  masked {(~ok).sum()} galaxies (non-finite features/spine)")
    X, data, spine, logmh = X[ok], data[ok], spine[ok], logmh[ok]
    if dev:
        X, data, spine, logmh = X[:400], data[:400], spine[:400], logmh[:400]
    return X, data, spine, logmh, np.asarray(e36._W["e"].R, float)


def _pca_basis(resid, k):
    """Pooled residual basis over all epochs: (mean_r (24,), modes (k, 24))."""
    flat = resid.reshape(-1, resid.shape[-1])
    mean_r = flat.mean(0)
    _, _, vt = np.linalg.svd(flat - mean_r, full_matrices=False)
    return mean_r, vt[:k]


def _scores(resid, mean_r, modes):
    return (resid - mean_r) @ modes.T                       # (n, 5, k)


def _dress_oof(X, logdata, spine_log_fn, k, rng_base=100):
    """One full OOF pass of the dressing machinery.

    ``spine_log_fn(tr, te)`` -> (spine_log_tr (ntr,5,24), spine_log_te) so the
    flat spine can be train-defined per fold. Returns the dressed OOF mean
    (linear CoGs), OOF draws, rho, and the cross-epoch score-residual C."""
    n = len(X)
    mu24 = np.full(logdata.shape, np.nan)
    draws = np.full((N_DRAW, *logdata.shape), np.nan)
    zres = [[] for _ in range(5)]
    fold_state = []
    for fi, te in enumerate(folds_of(n)):
        tr = np.setdiff1d(np.arange(n), te)
        sl_tr, sl_te = spine_log_fn(tr, te)
        mean_r, modes = _pca_basis(logdata[tr] - sl_tr, k)
        sc_tr = _scores(logdata[tr] - sl_tr, mean_r, modes)
        sc_te = _scores(logdata[te] - sl_te, mean_r, modes)
        emus, mus, sigs = [], [], []
        for kk in range(5):
            emu = emu_fit(X[tr], sc_tr[:, kk], mean="poly2")
            m, s, _ = emu.predict(X[te])
            emus.append(emu)
            mus.append(m)
            sigs.append(s)
            zres[kk].append((sc_te[:, kk] - m)
                            / np.where(s > 0, s, np.inf))
        mus, sigs = np.stack(mus, 1), np.stack(sigs, 1)     # (nte, 5, k)
        mu24[te] = sl_te + mus @ modes + mean_r
        fold_state.append((te, sl_te, mean_r, modes, emus, mus, sigs, fi))
    C = np.eye(5)
    from scipy.stats import spearmanr
    zc = [np.concatenate(z) for z in zres]
    for a in range(5):
        for b in range(a + 1, 5):
            r_ab = [spearmanr(zc[a][:, j], zc[b][:, j])[0]
                    for j in range(zc[a].shape[1])]
            C[a, b] = C[b, a] = float(np.nan_to_num(np.mean(r_ab)))
    rho = fit_rho(C)                        # skips non-positive C entries
    if not np.isfinite(rho):
        rho = 0.0
    for te, sl_te, mean_r, modes, emus, mus, sigs, fi in fold_state:
        rng = np.random.default_rng(rng_base + fi)
        eps = _sample_eps(rho, [e.corr for e in emus], N_DRAW * len(te), rng)
        eps = eps.reshape(N_DRAW, len(te), 5, -1)
        draws[:, te] = sl_te + (mus + sigs * eps) @ modes + mean_r
    return 10.0 ** mu24, 10.0 ** draws, rho, C


def pinned_shape(model, data, R, rmin=5.0):
    """148-pinned shape max|rel| over R>rmin, per galaxy per epoch (n, 5) —
    the 16.4/15.6-mark convention."""
    cs = model * (data[..., -1:] / model[..., -1:])
    rel = np.abs((cs - data) / data)
    return np.nanmax(rel[..., R > rmin], axis=-1)


def _npz(tag=""):
    return OUTDIR / f"dressing{tag}.npz"


def cmd_fit(dev=False, k=K_DEFAULT):
    X, data, spine, logmh, R = load(dev)
    n = len(X)
    logdata = np.log10(data)
    spine_log = np.log10(spine)
    print(f"exp36 (C) dressing fit (n={n}{', DEV' if dev else ''}, K={k}, "
          f"spine={SPINE_KEY}, {KFOLD}-fold OOF)")

    mu_k, draws_k, rho_k, C_k = _dress_oof(
        X, logdata, lambda tr, te: (spine_log[tr], spine_log[te]), k)
    print(f"  kernel spine dressed: rho = {rho_k:.3f}; adjacent C: "
          + " ".join(f"{C_k[a, a+1]:+.2f}" for a in range(4)))

    def flat_fn(tr, te):
        med = np.median(logdata[tr], axis=0)                # (5, 24)
        return (np.broadcast_to(med, logdata[tr].shape),
                np.broadcast_to(med, logdata[te].shape))

    mu_f, draws_f, rho_f, C_f = _dress_oof(X, logdata, flat_fn, k,
                                           rng_base=500)
    print(f"  flat spine dressed:   rho = {rho_f:.3f}; adjacent C: "
          + " ".join(f"{C_f[a, a+1]:+.2f}" for a in range(4)))
    OUTDIR.mkdir(exist_ok=True)
    np.savez(_npz("_dev" if dev else ""), mu_kernel=mu_k, draws_kernel=draws_k,
             rho_kernel=rho_k, C_kernel=C_k, mu_flat=mu_f, draws_flat=draws_f,
             rho_flat=rho_f, C_flat=C_f, n=n, k=k)
    print(f"  wrote {_npz('_dev' if dev else '')}")


def cmd_qa(dev=False):
    from hongshao import qa
    X, data, spine, logmh, R = load(dev)
    tag = "_dev" if dev else ""
    d = np.load(_npz(tag))
    assert int(d["n"]) == len(X), "saved fit does not match the loaded sample"

    print(f"exp36 (C) dressing QA (n={len(X)}, K={int(d['k'])})")
    print("  held-out 148-pinned shape max|rel| R>5 [%], median per epoch "
          f"(marks: {MARKS}):")
    rows = (("kernel spine alone", spine), ("kernel + dressing", d["mu_kernel"]),
            ("flat + dressing (guard)", d["mu_flat"]))
    shapes = {}
    for label, model in rows:
        ps = 100 * np.nanmedian(pinned_shape(model, data, R), axis=0)
        shapes[label] = ps
        print(f"    {label:24s}: " + " ".join(
            f"z{z}: {v:5.1f}" for z, v in zip(ZK, ps)))
    guard = (shapes["kernel + dressing"] < shapes["flat + dressing (guard)"])
    print(f"    ablation guard (kernel beats flat at each epoch): "
          f"{guard.astype(int)} -> {'PASS' if guard.all() else 'FAIL'}")

    res = qa.evaluate(d["mu_kernel"], data, R, ZK, name=f"exp36_dressed{tag}",
                      figdir=FIGDIR, bin_by=logmh, bin_label="logMh",
                      draw_cogs=d["draws_kernel"])

    print("\n  generative planes (energy/floor full | centered), OOF draws "
          "vs truth (exp37 block record ~0.4-0.9x through z=1):")
    truth_m, _, _, _ = qa.measure_all(data, data, R)
    for src, dk in (("kernel", "draws_kernel"), ("flat", "draws_flat")):
        for di in range(N_DRAW):
            dm, _, _, _ = qa.measure_all(d[dk][di], d[dk][di], R)
            for (kx, ky) in qa.PLANES[:2]:
                cells = []
                for kz in range(5):
                    pe = qa.plane_energy(
                        np.column_stack(
                            [np.log10(np.clip(truth_m[kx][:, kz], 1.0, None)),
                             np.log10(np.clip(truth_m[ky][:, kz], 1.0, None))]),
                        np.column_stack(
                            [np.log10(np.clip(dm[kx][:, kz], 1.0, None)),
                             np.log10(np.clip(dm[ky][:, kz], 1.0, None))]))
                    cells.append(f"{pe['energy_ratio']:4.1f}|"
                                 f"{pe['energy_ratio_centered']:4.1f}")
                print(f"    [{src}] draw {di} {kx} vs {ky}: "
                      + "  ".join(cells))

    anchors = np.log10(data[..., -1])
    la = np.log10(d["draws_kernel"][..., -1])
    gt = np.column_stack([anchors[:, 4], anchors[:, 0]])
    print("\n  growth plane logMtot(z=2) vs logMtot(z=0.4), energy/floor "
          "(exp37 record: draws 1.0-1.3x):")
    for i in range(N_DRAW):
        g = qa.plane_energy(gt, np.column_stack([la[i, :, 4], la[i, :, 0]]))
        print(f"    draw {i}: {g['energy_ratio']:.1f} | "
              f"{g['energy_ratio_centered']:.1f}")
    print(f"  rho: kernel {float(d['rho_kernel']):.3f}, "
          f"flat {float(d['rho_flat']):.3f} (exp37 measured 0.62)")
    return res


def demo():
    rng = np.random.default_rng(3)
    # (1) the pooled basis round-trip is exact at full rank
    resid = rng.normal(0, 0.1, (40, 5, 24))
    mean_r, modes = _pca_basis(resid, 24)
    rec = _scores(resid, mean_r, modes) @ modes + mean_r
    assert np.allclose(rec, resid, atol=1e-10), "full-rank round-trip broken"

    # (2) a (near-)zero residual dresses back to the spine (mean and draws)
    n = 120
    X = rng.normal(0, 1, (n, 5))
    spine_log = np.cumsum(rng.uniform(0.01, 0.2, (n, 5, 24)), axis=-1) + 8.0
    logdata = spine_log + rng.normal(0, 1e-6, spine_log.shape)
    mu, draws, rho, _ = _dress_oof(
        X, logdata, lambda tr, te: (spine_log[tr], spine_log[te]), 3)
    assert np.nanmax(np.abs(np.log10(mu) - spine_log)) < 1e-4, \
        "near-zero residual must dress to the spine"
    assert np.nanmax(np.abs(np.log10(draws) - spine_log[None])) < 1e-3, \
        "near-zero-residual draws must collapse onto the spine"

    # (3) a feature-driven residual is recovered by the dressing OOF
    true_mode = np.sin(np.linspace(0, 3, 24))
    amp = 0.2 * X[:, 0][:, None] + 0.05 * rng.normal(size=(n, 5))
    logdata = spine_log + amp[..., None] * true_mode
    mu, draws, rho, _ = _dress_oof(
        X, logdata, lambda tr, te: (spine_log[tr], spine_log[te]), 2)
    raw = np.abs(logdata - spine_log).mean()
    dressed = np.abs(logdata - np.log10(mu)).mean()
    assert dressed < 0.35 * raw, \
        f"dressing must explain the feature-driven residual ({dressed:.4f} " \
        f"vs raw {raw:.4f})"

    # (4) pinned shape: a pure amplitude offset has ZERO pinned shape error
    R = np.linspace(2, 148, 24)
    data = 10.0 ** spine_log
    assert np.nanmax(pinned_shape(1.7 * data, data, R)) < 1e-12
    print("dressing demo OK: full-rank basis round-trip exact; zero residual "
          "dresses to the spine (mean and draws); a feature-driven residual "
          "mode is recovered OOF; pinned shape ignores pure amplitude")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    dev = "--dev" in sys.argv
    k = (int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv
         else K_DEFAULT)
    if cmd == "demo":
        demo()
    elif cmd == "fit":
        cmd_fit(dev, k)
    elif cmd == "qa":
        cmd_qa(dev)
    else:
        sys.exit(f"unknown subcommand {cmd!r}")
