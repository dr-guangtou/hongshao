"""exp29 — standard aperture + outer-envelope stellar-mass QA.

A reusable evaluation step to run AFTER any model fit, IN PARALLEL with the profile
(max|rel|) metric. The profile metric asks "is the shape right at every radius?"; this
asks "are the integrated masses right?" -- aperture masses M*(<R) and, more sensitively,
outer-envelope masses M*(>R).

Two radial bin sets, same measurements:
  - kpc (physical): natural at low z.  Apertures at KPC_APER, envelopes at KPC_ENV.
  - R_half (relative): comparable across redshift.  Apertures at k*R_half (RE_APER),
    envelopes at k*R_half (RE_ENV). R_half = half-mass radius of the DATA CoG, used for
    BOTH model and truth, so a fixed k probes the same physical radius (mass error is
    isolated from size error) while adapting to each galaxy's size across epochs.

QA figure per bin set: rows = quantities, cols = [truth-vs-model value (log-log, 1:1),
truth-vs-relative-error (Y-X)/X]; points colored by epoch, per-epoch medians overplotted.

Use from a fitting script:
    import mass_qa
    mass_qa.evaluate(model_cogs, data_cogs, R, ANCHOR_Z, name="loose-quad")

Standalone demo (uses cached CoGs from integrated_check.npz):
    PYTHONPATH=. uv run python experiments/exp29_outer_deposit/mass_qa.py [name]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from hongshao.plotting import set_style, save_fig                                    # noqa: E402

set_style()
FIGDIR = HERE / "figures"

# --- standard radial bin definitions (edit here to change the standard) ---------
KPC_APER = [10.0, 30.0, 50.0, 100.0]        # M*(<R) cumulative apertures [kpc]
KPC_ENV = [50.0, 100.0]                      # M*(>R) outer envelopes     [kpc]
RE_APER = [1.0, 2.0, 4.0]                    # M*(<k*R_half) apertures
RE_ENV = [2.0, 4.0]                          # M*(>k*R_half) envelopes


def half_mass_radius(cog, R):
    """Half-mass radius from a (monotonic) CoG; log-log extrapolation below the grid
    for compact high-z galaxies whose R_half < R[0]."""
    target = 0.5 * cog[-1]
    if target <= cog[0]:
        sl = (np.log(cog[1]) - np.log(cog[0])) / (np.log(R[1]) - np.log(R[0]))
        return float(max(R[0] * np.exp((np.log(target) - np.log(cog[0])) / max(sl, 1e-3)), 0.3))
    return float(np.interp(target, cog, R))


def measure(cog, R, rhalf):
    """Standard aperture + envelope masses for one CoG. Keys are '<set>:label'."""
    mtot = cog[-1]
    m = {}
    for a in KPC_APER:
        m[f"kpc:M(<{int(a)})"] = float(np.interp(a, R, cog))
    for a in KPC_ENV:
        m[f"kpc:M(>{int(a)})"] = float(mtot - np.interp(a, R, cog))
    for k in RE_APER:
        m[f"Re:M(<{k:g}Re)"] = float(np.interp(k * rhalf, R, cog))
    for k in RE_ENV:
        m[f"Re:M(>{k:g}Re)"] = float(mtot - np.interp(k * rhalf, R, cog))
    return m


def measure_all(model_cogs, data_cogs, R):
    """Truth/model mass arrays (n, nz) per quantity; R_half from DATA (shared)."""
    n, nz = data_cogs.shape[:2]
    keys = list(measure(data_cogs[0, 0], R, 1.0).keys())
    truth = {k: np.zeros((n, nz)) for k in keys}
    model = {k: np.zeros((n, nz)) for k in keys}
    rhalf = np.zeros((n, nz))
    for i in range(n):
        for j in range(nz):
            rh = half_mass_radius(data_cogs[i, j], R); rhalf[i, j] = rh
            md = measure(data_cogs[i, j], R, rh); mm = measure(model_cogs[i, j], R, rh)
            for k in keys:
                truth[k][i, j] = md[k]; model[k][i, j] = mm[k]
    return truth, model, rhalf, keys


def _relerr(model, truth):
    return (model - truth) / np.clip(truth, 1.0, None)


def evaluate(model_cogs, data_cogs, R, anchor_z, name="model", figdir=FIGDIR, verbose=True):
    """Standard mass QA for one model. Prints a summary table and writes two QA figures
    (kpc and R_half bin sets). Returns (truth, model, rhalf) mass dicts/array."""
    truth, model, rhalf, keys = measure_all(model_cogs, data_cogs, R)
    nz = len(anchor_z)
    if verbose:
        print(f"\nmass QA [{name}]  (n={data_cogs.shape[0]}), median (model-data)/data per epoch:")
        print(f"    {'quantity':>14s} | " + " | ".join(f"z={z}".rjust(7) for z in anchor_z) +
              " |  median |truth|")
        for k in keys:
            re = _relerr(model[k], truth[k])
            row = " | ".join(f"{100*np.median(re[:, j]):+6.1f}%" for j in range(nz))
            print(f"    {k:>14s} | {row} |  {np.median(truth[k]):.2e}")
        rh_by_z = "  ".join(f"z={anchor_z[j]}:{np.median(rhalf[:,j]):.1f}" for j in range(nz))
        print(f"    R_half [kpc] median per epoch:  {rh_by_z}")

    _figure("kpc", [k for k in keys if k.startswith("kpc:")], truth, model, anchor_z, name, figdir)
    _figure("Re", [k for k in keys if k.startswith("Re:")], truth, model, anchor_z, name, figdir)
    return truth, model, rhalf


def _figure(tag, keys, truth, model, anchor_z, name, figdir):
    nz = len(anchor_z); cmap = matplotlib.colormaps["viridis"]
    cols = [cmap(j / max(nz - 1, 1)) for j in range(nz)]
    nrow = len(keys)
    fig, axes = plt.subplots(nrow, 2, figsize=(9.5, 2.7 * nrow), squeeze=False)
    for ri, key in enumerate(keys):
        X, Y = truth[key], model[key]
        av, ar = axes[ri, 0], axes[ri, 1]
        for j in range(nz):
            x, y = X[:, j], Y[:, j]
            good = (x > 0) & (y > 0)
            av.scatter(x[good], y[good], s=9, c=[cols[j]], alpha=0.45, edgecolors="none",
                       label=f"z={anchor_z[j]}" if ri == 0 else None)
            re = _relerr(y, x)
            ar.scatter(x[good], re[good], s=9, c=[cols[j]], alpha=0.35, edgecolors="none")
            mx = np.median(x[good]) if good.any() else np.nan
            ar.plot(mx, np.median(re[good]) if good.any() else np.nan, "o", c=cols[j], ms=9,
                    mec="k", mew=0.6)
        lo = np.nanmin([X[X > 0].min(), Y[Y > 0].min()]); hi = np.nanmax([X.max(), Y.max()])
        av.plot([lo, hi], [lo, hi], "k--", lw=1, zorder=0)
        av.set(xscale="log", yscale="log", ylabel=f"model\n{key}")
        ar.axhline(0, c="0.5", lw=0.8)
        for gpm in (0.1, -0.1):
            ar.axhline(gpm, c="0.7", ls=":", lw=0.8)
        ar.set(xscale="log", ylim=(-0.6, 0.6), ylabel="(model$-$truth)/truth")
        if ri == 0:
            av.legend(fontsize=7, loc="upper left", markerscale=1.6)
            av.set_title("truth vs model  (values, 1:1)"); ar.set_title("relative error vs truth")
        if ri == nrow - 1:
            av.set_xlabel(r"truth $M_*$ [$M_\odot$]"); ar.set_xlabel(r"truth $M_*$ [$M_\odot$]")
    unit = "physical kpc" if tag == "kpc" else r"$R_{\rm half}$ units"
    fig.suptitle(f"exp29 mass QA [{name}] — {unit}: aperture M*(<R) & envelope M*(>R)", fontsize=12)
    fig.tight_layout()
    print("wrote", save_fig(fig, figdir / f"exp29_massqa_{tag}_{name}")[0])


def _demo():
    """Run the standard QA on the cached loose-quad (or named) model CoGs."""
    from run import ANCHOR_Z
    d = np.load(HERE / "outputs" / "integrated_check.npz")
    name = sys.argv[1] if len(sys.argv) > 1 else "loose"
    key = {"loose": "loose", "independent": "independent"}.get(name, "loose")
    evaluate(d[key], d["data"], d["R"], ANCHOR_Z, name={"loose": "loose-quad"}.get(name, name))


if __name__ == "__main__":
    sys.exit(_demo())
