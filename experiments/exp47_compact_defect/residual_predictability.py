"""exp47 stage 1b — can a HALO-CONDITIONED term fix the tilt, or only a
stochastic layer? The gate before any fitting campaign.

Stage 1 established that the plane failure is the central-mass error, and
that it is a signed TILT: compact and extended galaxies are displaced in
opposite directions on the observational plane. Proposals 2 and 4 propose
opposite fixes for that tilt —

  proposal 2  change the deterministic model (free the efficiency window)
  proposal 4  accept the residual as stochastic and give the layer the
              right compactness-dependent width

— and only one of them can work. The deciding question is whether the
across-ridge residual is PREDICTABLE from halo properties. The kernel is
already a function of those properties, so any part of the residual that
a halo feature can still predict is information the current model is
leaving on the table, and a conditioning or window change can capture it.
Any part that is not predictable cannot be reached by ANY deterministic
halo-conditioned model, however it is reparameterized.

Nested feature sets, cheapest to richest:
  A  the kernel's CURRENT conditioning vector  [logMh, c200c, fz2]
  B  A + DiffMAH shape                         [logtc, early, late]
  C  B + the exp46 EARLY HEAD START            [Mh(1 Gyr), Mh(2 Gyr)
                                                relative to final]
  D  C + the fitted concentration history      [c200c at 5 epochs]

Set D is included as a NEGATIVE control: exp46 measured concentration to
be useless for compactness (R^2 0.012-0.037), so if D beats C materially
something is wrong with the setup.

Every R^2 is reported against a SHUFFLE control (features permuted across
galaxies) at the same dimensionality, because R^2 rises with the number of
predictors even for pure noise.

No fitting of the kernel. Run: PYTHONPATH=. uv run python \
experiments/exp47_compact_defect/residual_predictability.py {demo|run}
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/exp46_highz_ridge"))

OUTDIR = HERE / "outputs"
E46 = ROOT / "experiments/exp46_highz_ridge/outputs"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
DMAH_CSV = (ROOT / "experiments/exp10_diffmah_fit/outputs/"
            "diffmah_params_tmin1gyr.csv")
Z_GRID = (0.4, 0.7, 1.0, 1.5, 2.0)
PLANE = ("kpc:M(<30)", "kpc:M(30-50)")


def ridge_r2(y, X, n_shuffle=8, seed=0):
    """Out-of-sample R^2 from 5-fold CV, with a shuffle control.

    CV rather than in-sample R^2: with 12 predictors and a target this
    noisy, in-sample R^2 would overstate what a conditioning term could
    actually deliver on unseen galaxies.
    """
    rng = np.random.default_rng(seed)
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    y, X = y[ok], X[ok]
    n = len(y)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    Xs = np.column_stack([np.ones(n), Xs])

    def cv(Xd, yd):
        idx = rng.permutation(n)
        pred = np.empty(n)
        for f in range(5):
            te = idx[f::5]
            tr = np.setdiff1d(idx, te)
            beta, *_ = np.linalg.lstsq(Xd[tr], yd[tr], rcond=None)
            pred[te] = Xd[te] @ beta
        return 1.0 - np.var(yd - pred) / np.var(yd)

    real = cv(Xs, y)
    sh = []
    for _ in range(n_shuffle):
        Xp = Xs.copy()
        Xp[:, 1:] = Xp[rng.permutation(n), 1:]
        sh.append(cv(Xp, y))
    return float(real), float(np.median(sh)), int(n)


def feature_sets():
    pop = np.load(POP_NPZ)
    n = len(pop["index"])
    f = {"logmh": pop["logmh"], "c200c": pop["c200c"], "fz2": pop["fz2"]}
    raw = np.genfromtxt(DMAH_CSV, delimiter=",", names=True)
    lut = {int(i): k for k, i in enumerate(raw["index"])}
    for col in ("dmah_logtc", "dmah_early", "dmah_late"):
        v = np.full(n, np.nan)
        for k, idx in enumerate(pop["index"]):
            j = lut.get(int(idx))
            if j is not None:
                v[k] = raw[col][j]
        f[col] = v
    # the exp46 head start, straight off the real MAH
    cm = np.load(E46 / "compact_matched.npz")
    t, S = cm["t_ref"], cm["mah_real"]
    fin = S[:, -1]
    for tt in (1.0, 2.0):
        j = int(np.argmin(np.abs(t - tt)))
        f[f"headstart_t{tt:g}"] = S[:, j] - fin
    ch = np.load(E46 / "conc_history.npz")
    c = np.where(ch["GroupFlag"] == 1, ch["c200c"], np.nan)
    for j, z in enumerate(Z_GRID):
        f[f"c200c_z{z:g}"] = c[:, j]
    A = ["logmh", "c200c", "fz2"]
    B = A + ["dmah_logtc", "dmah_early", "dmah_late"]
    C = B + ["headstart_t1", "headstart_t2"]
    D = C + [f"c200c_z{z:g}" for z in Z_GRID]
    return f, {"A current conditioning": A, "B + DiffMAH shape": B,
               "C + early head start": C, "D + concentration history": D}


def cmd_run():
    from hongshao import qa
    from unification import (_w_init, adopted_model_cogs, ridge_decompose,
                             _W, KS_ALL)

    _w_init(None)
    gals, R = _W["gals"], _W["e"].R
    data_cogs = np.stack([g["data"] for g in gals])
    n = len(gals)
    print(f"exp47 stage 1b — is the across-ridge residual predictable from "
          f"HALO properties? (n={n})\n", flush=True)

    model_cogs = adopted_model_cogs(data_cogs, None)
    truth, model, _, _ = qa.measure_all(model_cogs, data_cogs, R)
    m5_t = np.array([[np.interp(5.0, R, data_cogs[i, j]) for j in range(5)]
                     for i in range(n)])
    m5_m = np.array([[np.interp(5.0, R, model_cogs[i, j]) for j in range(5)]
                     for i in range(n)])
    bias5 = np.log10(np.clip(m5_m, 1.0, None)) - np.log10(
        np.clip(m5_t, 1.0, None))

    feats, sets = feature_sets()
    kx, ky = PLANE
    print("  5-fold cross-validated R^2 [shuffle control in brackets].")
    print("  The shuffle is the same design matrix with features permuted "
          "across\n  galaxies, so it shows what this many predictors buy "
          "from noise alone.\n")
    res = {}
    for target_name in ("across-ridge plane residual", "M*(<5 kpc) bias"):
        print(f"    TARGET: {target_name}")
        print(f"      {'z':>5}" + "".join(f"{k:>26s}" for k in sets))
        for j, z in enumerate(Z_GRID):
            if target_name.startswith("across"):
                tx = np.log10(np.clip(truth[kx][:, j], 1.0, None))
                ty = np.log10(np.clip(truth[ky][:, j], 1.0, None))
                mx = np.log10(np.clip(model[kx][:, j], 1.0, None))
                my = np.log10(np.clip(model[ky][:, j], 1.0, None))
                _, y, _ = ridge_decompose(tx, ty, mx, my)
            else:
                y = bias5[:, j]
            cells = []
            for sname, keys in sets.items():
                X = np.column_stack([feats[k] for k in keys])
                r, s, nn = ridge_r2(y, X, seed=j)
                res[(target_name, j, sname)] = (r, s, nn)
                cells.append(f"{r:.3f} [{s:+.3f}]".rjust(26))
            print(f"      {z:5.1f}" + "".join(cells))
        print()

    OUTDIR.mkdir(exist_ok=True)
    np.savez(OUTDIR / "residual_predictability.npz",
             keys=np.array([f"{a}|{b}|{c}" for (a, b, c) in res]),
             vals=np.array([res[k] for k in res]))
    print(f"  wrote {OUTDIR / 'residual_predictability.npz'}")


def demo():
    rng = np.random.default_rng(0)
    n = 1500
    X = rng.normal(size=(n, 4))
    # a KNOWN 40% of variance is linear in the features
    sig = X @ np.array([0.8, -0.5, 0.3, 0.0])
    y = sig + rng.normal(0, sig.std() * np.sqrt(1 / 0.4 - 1), n)
    r, s, _ = ridge_r2(y, X)
    assert 0.30 < r < 0.50, r
    assert abs(s) < 0.05, s
    # pure noise: real R^2 must collapse to the shuffle level, NOT to 0.5
    r2n, s2n, _ = ridge_r2(rng.normal(size=n), X)
    assert abs(r2n) < 0.05 and abs(s2n) < 0.05, (r2n, s2n)
    # and adding junk predictors must not inflate CV R^2
    Xbig = np.column_stack([X, rng.normal(size=(n, 12))])
    rb, sb, _ = ridge_r2(y, Xbig)
    assert rb < r + 0.03, (rb, r)
    print(f"demo OK — recovers a planted R^2=0.40 as {r:.3f} "
          f"(shuffle {s:+.3f}); pure noise gives {r2n:+.3f}; "
          f"12 junk predictors do not inflate it ({rb:.3f})")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "demo":
        demo()
    elif cmd == "run":
        cmd_run()
    else:
        raise SystemExit(__doc__)
