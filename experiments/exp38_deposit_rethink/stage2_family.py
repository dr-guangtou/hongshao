"""exp38 stage 2b — the profile family, fitted JOINTLY across epochs with
parameters forced smooth in z, then conditioned on the halo features.

Stage 1 found: the slope-sigmoid family fits every epoch to 0.6% but its
raw per-epoch parameters cannot be interpolated in z (62-112% held-epoch
closure) — the classic degenerate-parameter failure. The honest fix (the
exp30 lesson: fit through the data, not through per-epoch point
estimates): per galaxy, make each family parameter a QUADRATIC in z and
fit all 5 epochs' growth curves at once.

Families:
  sigmoid-z   the exp03 slope-sigmoid [logMstar0, beta_in, beta_out,
              log10 R_c, log10 Delta], each quadratic in z -> 15
              coefficients per galaxy, fit to 5 x (R>=5 kpc) points.
  template-z  the evolving-Re template (shape frozen at the z=0.4
              population median in R/R_half units); only [log Mtot,
              log R_half] evolve, each quadratic in z -> 6 coefficients.

Tests:
  fit       joint smooth-in-z fits; per-epoch max|rel| R>5 vs the stage-1
            per-epoch ceilings (0.6% sigmoid / 6-7% template) — does the
            capacity survive the smoothness constraint?
  closure   refit on 4 epochs, predict the held epoch (the real
            smoothness test).
  condition predict the per-galaxy coefficients from [DiffMAH(4), c200c]
            (hongshao.emulator heteroscedastic cores, poly2, 5-fold OOF)
            and score the reconstructed curves: held-out 148-pinned shape
            max|rel| R>5 per epoch vs the marks (16.4 z=0.4 single-epoch /
            15.6 statistical wall / 17.7-14.7 multi 2ch-fa).

Run: PYTHONPATH=. uv run python experiments/exp38_deposit_rethink/\
stage2_family.py {demo|fit|condition|report} [--dev]
"""
import importlib.util
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from hongshao.profiles import beta_of_R, fit_cog                       # noqa: E402
from hongshao.qa import half_mass_radius                               # noqa: E402

OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
E35_DIR = ROOT / "experiments/exp35_total_norm"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
TABLE = ROOT / "data/processed/tng300_072_z0p4.fits"
FEATS = ("dmah_logmp", "dmah_logtc", "dmah_early", "dmah_late", "c_200c")
ZK = np.array([0.4, 0.7, 1.0, 1.5, 2.0])
ZS = (ZK - ZK.mean()) / ZK.std()          # standardized z for the quadratics
RMIN = 5.0
KFOLD_COND, SEED = 5, 0
MARKS = "16.4 z04-only / 15.6 statistical wall / 17.7-14.7 multi 2ch-fa"
_W = {}


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _w_init(rows):
    e = _load_by_path("exp35_run", E35_DIR / "run.py")
    mh_scale = tuple(np.load(ROOT / "experiments/exp32_full_population/"
                             "outputs/um_slope_diffmah.npz")["mh_scale"])
    e._init("diffmah", mh_scale, None if rows is None else np.asarray(rows))
    _W["e"] = e
    _W["gals"] = e._G["gals"]
    _W["R"] = np.asarray(e.R, float)
    _W["mask"] = _W["R"] >= RMIN
    _W["Rm"] = _W["R"][_W["mask"]]      # the family's grid: the sigmoid
    # amplitude anchors at Rm[0] (the fit_cog r_min=5 convention)
    # the frozen template: z=0.4 population-median shape in R/R_half units
    xg = np.geomspace(0.15, 40.0, 30)
    shapes_ = []
    for g in _W["gals"]:
        d0 = g["data"][0]
        rh = half_mass_radius(d0, _W["R"])
        y = np.log10(d0) - np.log10(d0[-1])
        shapes_.append(np.interp(
            np.log10(np.clip(xg * rh, _W["R"][0], _W["R"][-1])),
            np.log10(_W["R"]), y))
    _W["tpl_x"] = xg
    _W["tpl_y"] = np.median(np.stack(shapes_), axis=0)


# --------------------------------------------------------------------------- #
# the two smooth-in-z families                                                 #
# --------------------------------------------------------------------------- #
_SIG_LO = np.array([8.0, 0.0, 0.0, np.log10(2.0), np.log10(0.05)])
_SIG_HI = np.array([13.5, 9.0, 3.0, np.log10(148.0), np.log10(3.0)])


def _params_at_z(coeffs, nz=5):
    """(npar*3,) quadratic coefficients -> (nz, npar) parameter values."""
    C = coeffs.reshape(-1, 3)
    return C[:, 0][None, :] + C[:, 1][None, :] * ZS[:nz, None] \
        + C[:, 2][None, :] * (ZS[:nz, None] ** 2)


def _sigmoid_cog_log(q5, R):
    """log10 CoG of the slope-sigmoid family (mirrors
    hongshao.profiles.cog_from_physical, box-clipped for the joint fit)."""
    q = np.clip(q5, _SIG_LO, _SIG_HI)
    beta = beta_of_R(R, q[1] + q[2], q[2], 10.0 ** q[3], 10.0 ** q[4])
    u = np.log(R)
    integ = np.concatenate([[0.0], np.cumsum(
        0.5 * (beta[1:] + beta[:-1]) * np.diff(u))])
    return q[0] + integ / np.log(10.0)


def _template_cog_log(q2, R):
    x = R / (10.0 ** q2[1])
    shape = np.interp(np.log10(np.clip(x, _W["tpl_x"][0], _W["tpl_x"][-1])),
                      np.log10(_W["tpl_x"]), _W["tpl_y"])
    return q2[0] + shape


_FAMS = {"sigmoid-z": (5, _sigmoid_cog_log), "template-z": (2, _template_cog_log)}


def _fit_joint(fam, logC, ks):
    """Per-galaxy joint fit over epochs ks: quadratic-in-z parameters,
    residuals = log-CoG mismatch over R >= RMIN at every kept epoch."""
    npar, cog_fn = _FAMS[fam]
    R = _W["R"]
    Rm = _W["Rm"]
    m = _W["mask"]

    def resid(c):
        P = _params_at_z(c)
        return np.concatenate([cog_fn(P[k], Rm) - logC[k][m] for k in ks])

    # warm start: independent per-epoch fits -> quadratic in z per param
    per = []
    for k in range(5):
        if fam == "sigmoid-z":
            d = fit_cog(R, logC[k], r_min=RMIN)
            per.append([d["logMstar0"], d["beta_in"] - d["beta_out"],
                        d["beta_out"], np.log10(d["R_c"]),
                        np.log10(d["Delta"])])
        else:
            rh = half_mass_radius(10.0 ** logC[k], R)
            per.append([logC[k][-1], np.log10(rh)])
    per = np.array(per)
    c0 = np.concatenate([np.polyfit(ZS, per[:, j], 2)[::-1]
                         for j in range(npar)])
    r = least_squares(resid, c0, method="lm", max_nfev=400 * len(c0))
    return r.x


def _gal_fit_all(args):
    """One galaxy: full joint fit + the 5 held-epoch closure refits."""
    i, do_closure = args
    g = _W["gals"][i]
    Rm = _W["Rm"]
    m = _W["mask"]
    logC = [np.log10(g["data"][k]) for k in range(5)]
    out = {}
    for fam, (npar, cog_fn) in _FAMS.items():
        c = _fit_joint(fam, logC, list(range(5)))
        P = _params_at_z(c)
        met = [float(np.abs(10.0 ** (cog_fn(P[k], Rm) - logC[k][m])
                            - 1.0).max()) for k in range(5)]
        clo = [np.nan] * 5
        if do_closure:
            for hold in range(5):
                ch = _fit_joint(fam, logC, [k for k in range(5) if k != hold])
                Ph = _params_at_z(ch)
                clo[hold] = float(np.abs(
                    10.0 ** (cog_fn(Ph[hold], Rm) - logC[hold][m])
                    - 1.0).max())
        out[fam] = (c, met, clo)
    return g["row"], out


def cmd_fit(dev=False):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    gals = _W["gals"]
    n = len(gals)
    tag = "_dev" if dev else ""
    workers = max(os.cpu_count() - 2, 2)
    print(f"exp38 stage 2b joint smooth-in-z family fits (n={n}"
          f"{', DEV' if dev else ''}; quadratic-in-z parameters, all 5 "
          "epochs at once, R>=5 kpc):")
    t0 = time.time()
    res = {fam: {} for fam in _FAMS}
    with Pool(workers, initializer=_w_init, initargs=(rows,)) as pool:
        for row, out in pool.imap_unordered(_gal_fit_all,
                                            [(i, True) for i in range(n)],
                                            chunksize=8):
            for fam in _FAMS:
                res[fam][row] = out[fam]
    print(f"  ({(time.time()-t0)/60:.1f} min)")
    save = {}
    for fam in _FAMS:
        rows_ = sorted(res[fam])
        C = np.stack([res[fam][r][0] for r in rows_])
        met = np.stack([res[fam][r][1] for r in rows_])
        clo = np.stack([res[fam][r][2] for r in rows_])
        save[f"coeffs_{fam}"] = C
        save[f"met_{fam}"] = met
        save[f"closure_{fam}"] = clo
        save["rows"] = np.array(rows_)
        line = " ".join(f"z{ZK[k]}: {100*np.nanmedian(met[:, k]):5.1f}"
                        for k in range(5))
        cline = " ".join(f"z{ZK[k]}: {100*np.nanmedian(clo[:, k]):5.1f}"
                         for k in range(5))
        print(f"  {fam:>10s} joint-fit max|rel| R>5 [%]: {line}")
        print(f"  {fam:>10s} held-epoch closure   [%]: {cline}")
    OUTDIR.mkdir(exist_ok=True)
    np.savez(OUTDIR / f"stage2_family{tag}.npz", **save)
    print(f"wrote {OUTDIR / f'stage2_family{tag}.npz'}")


# --------------------------------------------------------------------------- #
# conditioning on the halo features                                            #
# --------------------------------------------------------------------------- #
def cmd_condition(dev=False):
    from astropy.table import Table
    from hongshao.emulator import fit as emu_fit
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    gals = _W["gals"]
    m = _W["mask"]
    tag = "_dev" if dev else ""
    d = np.load(OUTDIR / f"stage2_family{tag}.npz")
    pop = np.load(POP_NPZ)
    t = Table.read(TABLE)
    trow = {int(g): i for i, g in enumerate(np.asarray(t["index"]))}
    row_to_i = {int(r): i for i, r in enumerate(d["rows"])}
    gals = [g for g in gals if g["row"] in row_to_i]
    X = np.array([[t[c][trow[int(pop["index"][g["row"]])]] for c in FEATS]
                  for g in gals], float)
    ok = np.isfinite(X).all(1)
    gals = [g for g, o in zip(gals, ok) if o]
    X = X[ok]
    n = len(gals)
    order = np.random.default_rng(SEED).permutation(n)
    folds = np.array_split(order, KFOLD_COND)
    print(f"exp38 stage 2b conditioning (n={n}{', DEV' if dev else ''}; "
          "[DiffMAH(4), c200c] -> family coefficients, poly2 cores, "
          f"{KFOLD_COND}-fold OOF; marks: {MARKS}):")
    for fam, (npar, cog_fn) in _FAMS.items():
        C = np.stack([d[f"coeffs_{fam}"][row_to_i[g["row"]]] for g in gals])
        mu = np.full_like(C, np.nan)
        for fold in folds:
            tr = np.setdiff1d(np.arange(n), fold)
            emu = emu_fit(X[tr], C[tr], mean="poly2")
            mu[fold] = emu.predict(X[fold])[0]
        met = np.full((n, 5), np.nan)
        for i, g in enumerate(gals):
            P = _params_at_z(mu[i])
            for k in range(5):
                mod = 10.0 ** cog_fn(P[k], _W["Rm"])
                dat = g["data"][k][m]
                mod = mod * (dat[-1] / mod[-1])          # 148-pinned shape
                met[i, k] = np.abs(mod / dat - 1.0).max()
        line = " ".join(f"z{ZK[k]}: {100*np.nanmedian(met[:, k]):5.1f}"
                        for k in range(5))
        print(f"  {fam:>10s} held-out pinned shape R>5 [%]: {line}")
        np.savez(OUTDIR / f"stage2_condition_{fam}{tag}.npz", met=met, mu=mu)
    print("wrote stage2_condition_*.npz")


def cmd_report(dev=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hongshao.plotting import set_style, save_fig, OKABE_ITO
    from hongshao.qa import _tex
    set_style()
    tag = "_dev" if dev else ""
    d = np.load(OUTDIR / f"stage2_family{tag}.npz")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for fam, col in zip(_FAMS, OKABE_ITO):
        met = d[f"met_{fam}"]
        clo = d[f"closure_{fam}"]
        axes[0].plot(ZK, [100 * np.nanmedian(met[:, k]) for k in range(5)],
                     "-o", ms=4, c=col, label=fam)
        axes[1].plot(ZK, [100 * np.nanmedian(clo[:, k]) for k in range(5)],
                     "-o", ms=4, c=col, label=fam)
        cnd = OUTDIR / f"stage2_condition_{fam}{tag}.npz"
        if cnd.exists():
            mc = np.load(cnd)["met"]
            axes[1].plot(ZK, [100 * np.nanmedian(mc[:, k]) for k in range(5)],
                         "--s", ms=4, c=col, label=f"{fam} conditioned")
    axes[0].set(xlabel="epoch z",
                ylabel=_tex("joint-fit max|rel| R>5") + " [percent]",
                title="smooth-in-z joint fit (capacity kept?)")
    axes[1].axhline(16.4, color="0.3", ls=":", lw=1.2)
    axes[1].axhline(15.6, color="0.6", ls=":", lw=1.2)
    axes[1].set(xlabel="epoch z",
                ylabel=_tex("held-out / closure max|rel| R>5") + " [percent]",
                title="closure + halo-conditioned (marks dotted)")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.suptitle("exp38 stage 2b — the smooth-in-z profile family",
                 fontsize=12)
    fig.tight_layout()
    FIGDIR.mkdir(exist_ok=True)
    print("wrote", save_fig(fig, FIGDIR / f"stage2_family{tag}")[0])


def demo():
    rows = np.load(POP_NPZ)["dev100"][:10]
    _w_init(rows)
    R = _W["R"]
    Rm = _W["Rm"]
    m = _W["mask"]
    # (1) quadratic layer: constant coefficients give constant parameters
    c = np.concatenate([[11.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    P = _params_at_z(c)
    assert np.allclose(P, [[11.0, 2.0]] * 5)
    # (2) the sigmoid family reproduces hongshao.profiles on a real curve
    # (both anchored at Rm[0], the fit_cog r_min convention)
    g = _W["gals"][0]
    logC = np.log10(g["data"][0])
    dfit = fit_cog(R, logC, r_min=RMIN)
    q5 = np.array([dfit["logMstar0"], dfit["beta_in"] - dfit["beta_out"],
                   dfit["beta_out"], np.log10(dfit["R_c"]),
                   np.log10(dfit["Delta"])])
    got = _sigmoid_cog_log(q5, Rm)
    from hongshao.profiles import cog_from_physical
    ref = cog_from_physical(Rm, dfit["logMstar0"], dfit["beta_in"],
                            dfit["beta_out"], dfit["R_c"], dfit["Delta"])
    assert np.abs(got - ref).max() < 1e-9, "sigmoid wrapper must match library"
    # (3) a joint fit on synthetic smooth-in-z truth recovers the CURVES
    truth_c = np.concatenate([[11.2, 0.15, 0.02], [4.0, -0.3, 0.0],
                              [0.6, 0.05, 0.0], [1.1, 0.1, 0.0],
                              [-0.3, 0.0, 0.0]])
    Pt = _params_at_z(truth_c)
    logC_syn = []
    for k in range(5):
        full = np.full(len(R), np.nan)
        full[m] = _sigmoid_cog_log(Pt[k], Rm)
        full[~m] = full[m][0] - 0.5 * (1.0 - R[~m] / Rm[0])   # inner filler
        logC_syn.append(full)
    cfit = _fit_joint("sigmoid-z", logC_syn, list(range(5)))
    Pf = _params_at_z(cfit)
    err = max(np.abs(_sigmoid_cog_log(Pf[k], Rm) - logC_syn[k][m]).max()
              for k in range(5))
    assert err < 1e-3, f"joint fit must recover synthetic truth ({err:.1e})"
    print("stage2b demo OK: quadratic layer exact; the sigmoid wrapper "
          "matches hongshao.profiles to 1e-9 on the anchored grid; the "
          "joint smooth-in-z fit recovers a synthetic smooth truth")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    dev = "--dev" in sys.argv
    if cmd == "demo":
        demo()
    elif cmd == "fit":
        cmd_fit(dev)
    elif cmd == "condition":
        cmd_condition(dev)
    elif cmd == "report":
        cmd_report(dev)
    else:
        sys.exit(f"unknown subcommand {cmd!r}")
