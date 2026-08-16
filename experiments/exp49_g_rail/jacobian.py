"""exp49 stage 1 — local geometry of the 12-parameter kernel at the bounded solution.

**This measures LOCAL GEOMETRY, not parameter uncertainty.** `g` sits on an
active bound here, so nothing in this file may be read as a confidence
interval, and a Hessian at this point would not be unconstrained curvature.
The interior Hessian belongs at the widened solution (stage 3).

What it does: finite-difference the FULL fractional-CoG residual array
(n_galaxies x n_epochs x n_radii) with respect to the 12 population
parameters, then

  - report RAW column norms, which say how much each parameter moves the
    residuals in absolute terms;
  - COLUMN-NORMALIZE before the SVD, because without it the singular vectors
    report unit choices (log_rc in dex against gamma dimensionless) rather
    than degeneracies;
  - read off the weak directions, and how much of each lies in the
    (log_rc, g) plane -- the pair exp49 showed to be degenerate along
    g = 1.4822*log_rc - 0.0843.

Three finite-difference steps (1e-6, 1e-5, 1e-4) are run and compared. Measured
in this project: gradients on this loss are stable from 1e-4 down to 1e-7,
while 1e-2 misreads the deposit-scale growth exponent g by a factor of five.
Where singular values are nearly equal the individual singular vectors rotate
freely, so subspaces are compared by PRINCIPAL ANGLES rather than vector by
vector.

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python \
experiments/exp49_g_rail/jacobian.py [--dev]
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/exp38_deposit_rethink"))

import stage2_multiepoch as s2                      # noqa: E402
from hongshao.objective import Objective            # noqa: E402
from hongshao.plotting import save_fig, set_style   # noqa: E402

OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
KS = [0, 1, 2, 3]                                   # the adopted z<=1.5 scope
STEPS = (1e-6, 1e-5, 1e-4)
NAMES = ["log_rc", "g", "q", "mu", "sig", "gamma",
         "rc:logMh", "rc:c200c", "rc:fz2",
         "sig:logMh", "sig:c200c", "sig:fz2"]
PAIR = (0, 1)                                       # the (log_rc, g) plane

# The production objective's OWN residual, so this diagnostic cannot drift from
# the thing being diagnosed. Private on purpose: there is no public accessor for
# the per-radius residual array, and duplicating the formula here is exactly the
# kind of copy the objective-module refactor removed.
_OBJ = Objective()


def residuals(theta, gals, R):
    """Flat fractional-CoG residual array at ``theta``. NaN-free by assertion."""
    cogs = np.empty((len(gals), len(KS), len(R)))
    for i, g in enumerate(gals):
        c = s2.model_cogs(theta, g, KS, "1ch-mof")
        if c is None:
            raise RuntimeError(f"model failed on galaxy {i} at this theta; the "
                               f"Jacobian is undefined here")
        cogs[i] = np.asarray(c)
    data = np.stack([np.array([g["data"][k] for k in KS], float) for g in gals])
    r, _ = _OBJ._residual(cogs, data, np.asarray(R, float))
    return r.ravel()


def jacobian(theta, gals, R, h):
    """Central-difference d(residual)/d(theta), shape (n_residuals, 12)."""
    J = np.empty((len(gals) * len(KS) * len(R), len(theta)))
    for j in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h
        tm[j] -= h
        J[:, j] = (residuals(tp, gals, R) - residuals(tm, gals, R)) / (2.0 * h)
        print(f"    column {NAMES[j]:<10} done", flush=True)
    return J


def principal_angles(A, B):
    """Principal angles (degrees) between the subspaces spanned by the COLUMNS
    of A and B. Both are orthonormalized first. Small angles = same subspace."""
    qa, _ = np.linalg.qr(A)
    qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def analyse(J, tag):
    """Raw sensitivity, then the column-normalized SVD."""
    col = np.linalg.norm(J, axis=0)
    Jn = J / col[None, :]
    U, sv, Vt = np.linalg.svd(Jn, full_matrices=False)
    corr = Jn.T @ Jn                       # unit columns -> this IS the correlation
    print(f"\n  [{tag}] RAW column norms (absolute sensitivity of the "
          f"residual array):")
    order = np.argsort(-col)
    for j in order:
        print(f"    {NAMES[j]:<11}{col[j]:12.4f}")
    print(f"\n  [{tag}] singular values of the COLUMN-NORMALIZED Jacobian "
          f"(condition number {sv[0] / sv[-1]:.3g}):")
    for i, s in enumerate(sv):
        print(f"    sv{i:<2} {s:10.5f}   ratio to largest {s / sv[0]:9.2e}")
    return dict(col=col, sv=sv, V=Vt.T, corr=corr)


def report_weak(res, n_weak=3):
    """What the weakest directions are made of, and how much of each lies in
    the (log_rc, g) plane that exp49 measured as degenerate."""
    V, sv = res["V"], res["sv"]
    print(f"\n  weakest {n_weak} directions (largest entries first). "
          f"'in (log_rc,g)' = squared overlap with that plane:")
    for k in range(1, n_weak + 1):
        v = V[:, -k]
        frac = v[PAIR[0]] ** 2 + v[PAIR[1]] ** 2
        big = np.argsort(-np.abs(v))[:5]
        terms = "  ".join(f"{NAMES[j]} {v[j]:+.3f}" for j in big)
        print(f"    sv{len(sv)-k} = {sv[-k]:.5f} | in (log_rc,g) = "
              f"{frac:5.3f} | {terms}")
    # Along the weakest direction the residuals barely change, so its slope in
    # the (log_rc, g) plane IS the valley direction: d(g)/d(log_rc) = v_g/v_lrc.
    v = V[:, -1]
    if abs(v[PAIR[0]]) > 1e-12:
        print(f"\n    weakest direction implies d(g)/d(log_rc) = "
              f"{v[PAIR[1]] / v[PAIR[0]]:+.4f}")
        print("    (exp49's 2-D loss-valley fit gave +1.4822, with the other "
              "ten parameters FROZEN; this one lets them move, so the two "
              "need not agree)")


def figures(res, tag):
    set_style()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    im = axes[0].imshow(res["corr"], vmin=-1, vmax=1, cmap="RdBu_r")
    axes[0].set_xticks(range(12), NAMES, rotation=90)
    axes[0].set_yticks(range(12), NAMES)
    axes[0].set_title("parameter correlation\n(column-normalized Jacobian)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    V = res["V"]
    im2 = axes[1].imshow(V, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    axes[1].set_yticks(range(12), NAMES)
    axes[1].set_xticks(range(12), [f"sv{i}" for i in range(12)], rotation=90)
    axes[1].set_title("right singular vectors\n(sv0 strongest, sv11 weakest)")
    fig.colorbar(im2, ax=axes[1], fraction=0.046)
    fig.suptitle("exp49 stage 1 — local geometry at the BOUNDED solution "
                 "(g on an active bound; not parameter uncertainty)",
                 fontsize=10)
    fig.tight_layout()
    paths = save_fig(fig, FIGDIR / f"exp49_jacobian_{tag}")
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.semilogy(range(12), res["sv"] / res["sv"][0], "o-")
    ax.set_xlabel("singular-value index")
    ax.set_ylabel("singular value / largest")
    ax.set_title("sensitivity spectrum of the kernel")
    ax.grid(alpha=0.3)
    fig2.tight_layout()
    paths += save_fig(fig2, FIGDIR / f"exp49_spectrum_{tag}")
    plt.close(fig2)
    return paths


def main(dev=False):
    rows = (np.load(ROOT / "experiments/exp32_full_population/outputs/"
                    "population.npz")["dev100"] if dev else None)
    s2._w_init(rows)
    gals, R = s2._W["gals"], s2._W["e"].R
    store = np.load(OUTDIR / "railtest.npz", allow_pickle=True)
    theta = np.asarray(store["theta::bounded::adopted"], float)
    tag = "dev" if dev else "full"
    print(f"exp49 stage 1 — residual-Jacobian SVD at the BOUNDED solution\n"
          f"  n = {len(gals)} galaxies, {len(KS)} epochs, {len(R)} radii "
          f"-> {len(gals)*len(KS)*len(R)} residuals\n"
          f"  theta: g = {theta[1]:.5f} (ON its bound 4.0), "
          f"log_rc = {theta[0]:.5f}", flush=True)
    nfail = sum(s2.model_cogs(theta, g, KS, "1ch-mof") is None for g in gals)
    assert nfail == 0, f"{nfail} galaxies fail here; the Jacobian is undefined"

    out, results = {}, {}
    for h in STEPS:
        print(f"\n  finite-difference step h = {h:g}", flush=True)
        J = jacobian(theta, gals, R, h)
        res = analyse(J, f"h={h:g}")
        report_weak(res)
        results[h] = res
        for key, val in res.items():
            out[f"{key}::{h:g}"] = val

    # stability across steps: compare the weak SUBSPACES, not single vectors
    print("\n" + "=" * 74)
    print("STABILITY ACROSS FINITE-DIFFERENCE STEPS (principal angles, degrees)")
    print("=" * 74)
    ref = results[STEPS[1]]["V"]
    for h in STEPS:
        if h == STEPS[1]:
            continue
        for k in (1, 2, 3):
            ang = principal_angles(ref[:, -k:], results[h]["V"][:, -k:])
            print(f"  h={h:g} vs {STEPS[1]:g}, weakest-{k} subspace: "
                  f"max angle {ang.max():7.3f} deg")

    OUTDIR.mkdir(exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, cwd=ROOT).stdout.strip()
    out["meta"] = np.array([json.dumps(dict(
        git_sha=sha, n_galaxies=len(gals), epochs=KS, steps=list(STEPS),
        names=NAMES, objective=_OBJ.name, theta=theta.tolist(),
        note="LOCAL GEOMETRY at the bounded solution; g on an active bound; "
             "NOT parameter uncertainty"))])
    np.savez(OUTDIR / f"jacobian_{tag}.npz", **out)
    paths = figures(results[STEPS[1]], tag)
    print(f"\n  wrote {OUTDIR / f'jacobian_{tag}.npz'}")
    for p in paths:
        print(f"  wrote {p}")


if __name__ == "__main__":
    main("--dev" in sys.argv)
