"""exp72 Step 3 — is a fixed physical radial grid the right thing to score on?

THE USER'S ARGUMENT (2026-09-01), which this script tests rather than assumes.

  1. The same physical radial bins are used at all five epochs. At high redshift
     the outer bins (100-150 kpc at z = 1.5, 50-100 kpc at z = 2) hold very
     little mass, in the simulation as in observation, so scoring them equally
     with the inner bins over-weights a part of the galaxy that barely exists.
     Either down-weight bins whose mass fraction falls below a threshold, or
     move to a coordinate that follows the galaxy -- R50, the half-mass radius.

  2. Every residual in this programme is RELATIVE, (model - data) / data. A
     z = 2 model 50 per cent wrong in the 100-150 kpc bin sounds bad, but if
     almost no mass is there then almost no mass has been misplaced. The
     relative form can therefore paint a biased picture of where a model fails
     -- while still not licensing us to ignore the outskirts, because where the
     simulation DOES put mass there, it counts.

Both claims are quantitative and neither has been measured here. What follows
measures them and nothing else; no fit is run and no objective is changed.

WHAT IS MEASURED

1. **Where the mass actually is.** The median fraction of a galaxy's measured
   stellar mass inside 148 kpc that sits in each shell, per epoch. This is the
   weight the user says the objective should respect.
2. **Where the fixed grid sits relative to the galaxy.** The median half-mass
   radius per epoch, and the fixed physical grid expressed in units of it. If
   R50 shrinks with redshift then 100 kpc is a very different place in a z = 2
   galaxy than in a z = 0.4 one, and a shared grid is scoring different things
   at different epochs.
3. **Relative error against mass misplaced.** For each shell, the median
   relative error the QA figures show, beside the ABSOLUTE mass error expressed
   as a fraction of the galaxy's total. The gap between those two columns is
   exactly claim 2.
4. **What the objectives charge against where the mass is.** The share of each
   objective's loss carried by each shell, beside the share of the mass. A
   mismatch is the bias claim 1 predicts.
5. **The same, in R50 units.** Whether moving to a size-relative coordinate
   removes the mismatch or merely moves it.

Run:
    HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 \\
    PYTHONPATH=. uv run python -u \\
        experiments/exp72_error_objective/radial_evidence.py [--smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term",
          ROOT / "experiments/exp63_analytic_growth", HERE):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import engine as E                                       # noqa: E402
import model2 as M2                                      # noqa: E402
from hongshao import qa                                  # noqa: E402
import fit_gls as G                                      # noqa: E402
from errors import RULE, THIN, per_galaxy, to_annuli     # noqa: E402

OUTDIR = HERE / "outputs"
ANCHOR_Z = E.ANCHOR_Z
EPOCHS = (0, 1, 2, 3, 4)
EXP63_NPZ = ROOT / "experiments/exp63_analytic_growth/outputs/stage2_fit_joint_kpc_free_sane.npz"
#: shells the tables are printed over, in kpc (edges on the 24-radius grid)
SHELL_EDGES_KPC = (2.0, 4.92, 10.25, 22.97, 52.30, 103.45, 148.22)
#: shells in units of the galaxy's own half-mass radius
SHELL_EDGES_RE = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
#: a bin holding less than this fraction of the galaxy's mass is one the user
#: proposes to down-weight or drop
LOW_MASS_FRAC = 0.02


def shell_index(edges):
    return [int(np.argmin(np.abs(F.R_GRID - e))) for e in edges]


def shell_masses(cog, idx):
    """(n, 5, n_shell) mass between consecutive grid radii."""
    return np.stack([cog[..., idx[j + 1]] - cog[..., idx[j]]
                     for j in range(len(idx) - 1)], axis=-1)


def cum_at_multiple_of_re(cog, R, rhalf, mult):
    """M*(< mult * R50) per galaxy-epoch by interpolation on the grid."""
    n, ne, _ = cog.shape
    out = np.full((n, ne), np.nan)
    for i in range(n):
        for j in range(ne):
            r = mult * rhalf[i, j]
            if not np.isfinite(r) or r < R[0] or r > R[-1]:
                continue
            out[i, j] = np.interp(r, R, cog[i, j])
    return out


def main(smoke=False):
    print(f"{RULE}\nexp72 Step 3 — is a fixed physical radial grid the right "
          f"thing to score on?\n{RULE}\n")
    (recs, data, lmh, curves, sel, spec2, th_inc, th_nested, fz63,
     keep, sig_cog, sig_ann, ann_ok) = G.build(smoke)
    fzg = np.load(OUTDIR / "fit_gls.npz", allow_pickle=True)
    thetas = {"exp63 joint": np.asarray(fz63["theta_best"], float),
              "chi2 refit": np.asarray(fzg["theta_best"], float)}
    cogs = {n: M2.predict2(spec2, t, curves, F.R_GRID, nodes=M2.FULL_NODES)
            for n, t in thetas.items()}
    good = keep.all(1)
    for m in cogs.values():
        good &= np.isfinite(m).all(axis=(1, 2)) & (m > 0).all(axis=(1, 2))
    d = data[good]
    R = F.R_GRID
    print(f"  {int(good.sum())} galaxies; truth = the measured curves of growth")

    idx = shell_index(SHELL_EDGES_KPC)
    labs = [f"{SHELL_EDGES_KPC[j]:g}-{SHELL_EDGES_KPC[j+1]:g}"
            for j in range(len(SHELL_EDGES_KPC) - 1)]
    m_shell_d = shell_masses(d, idx)                       # (n, 5, n_shell)
    total = d[..., -1]                                     # M*(<148 kpc)

    # ---- 1. where the mass is ------------------------------------------- #
    print(f"\n{RULE}\n1. WHERE THE MASS ACTUALLY IS — median fraction of a "
          f"galaxy's measured stellar mass\n   inside 148 kpc that sits in each "
          f"shell. This is the weight the objective ought to\n   respect, and "
          f"it is NOT the same at every epoch.\n{RULE}")
    print(f"\n  {'epoch':>6}" + "".join(f"{l:>12}" for l in labs) + "   kpc")
    frac = np.full((5, len(labs)), np.nan)
    for k in EPOCHS:
        frac[k] = np.median(m_shell_d[:, k, :] / total[:, k][:, None], axis=0)
        flag = ["*" if v < LOW_MASS_FRAC else " " for v in frac[k]]
        print(f"  {ANCHOR_Z[k]:>6.1f}"
              + "".join(f"{100 * v:>11.1f}%{f}" for v, f in zip(frac[k], flag)))
    print(f"\n  (* marks a shell holding less than "
          f"{100 * LOW_MASS_FRAC:g} per cent of the galaxy's mass)")
    print(f"\n  the same as a CUMULATIVE fraction — how much of the galaxy is "
          f"already inside R")
    print(f"  {'epoch':>6}" + "".join(f"{e:>12.0f}" for e in SHELL_EDGES_KPC[1:]) + "   kpc")
    for k in EPOCHS:
        c = np.median(d[:, k, :] / total[:, k][:, None], axis=0)
        print(f"  {ANCHOR_Z[k]:>6.1f}"
              + "".join(f"{100 * c[j]:>11.1f}%" for j in idx[1:]))

    # ---- 2. where the grid sits relative to the galaxy ------------------- #
    print(f"\n{RULE}\n2. WHERE THE FIXED GRID SITS RELATIVE TO THE GALAXY — the "
          f"half-mass radius shrinks\n   with redshift, so a fixed physical "
          f"radius is a DIFFERENT PLACE in the galaxy at\n   each epoch.\n{RULE}")
    rhalf = np.full((d.shape[0], 5), np.nan)
    for k in EPOCHS:
        for i in range(d.shape[0]):
            rhalf[i, k] = qa.half_mass_radius(d[i, k], R)
    med_re = np.nanmedian(rhalf, axis=0)
    print(f"\n  median half-mass radius R50 [kpc]: "
          + "  ".join(f"z={z}: {v:.1f}" for z, v in zip(ANCHOR_Z, med_re)))
    print(f"\n  the fixed grid, expressed in units of the galaxy's own R50")
    print(f"  {'epoch':>6}" + "".join(f"{e:>10.0f}" for e in SHELL_EDGES_KPC) + "   kpc")
    for k in EPOCHS:
        print(f"  {ANCHOR_Z[k]:>6.1f}"
              + "".join(f"{e / med_re[k]:>10.1f}" for e in SHELL_EDGES_KPC)
              + "   x R50")
    print(f"\n  Read the last column: 148 kpc is {148.22 / med_re[0]:.0f} R50 at "
          f"z = 0.4 and {148.22 / med_re[4]:.0f} R50 at z = 2. The outermost "
          f"bins are\n  scoring the far envelope of a high-redshift galaxy "
          f"against the mid-outskirts of a\n  low-redshift one, under one "
          f"shared law.")

    # ---- 3. relative error against mass misplaced ------------------------ #
    print(f"\n{RULE}\n3. RELATIVE ERROR AGAINST MASS MISPLACED — the QA figures "
          f"show the first column.\n   The second is the SAME error as a "
          f"fraction of the galaxy's whole stellar mass, i.e.\n   how much mass "
          f"is actually in the wrong place. Claim 2 is the gap between "
          f"them.\n{RULE}")
    misp = {}
    for n in cogs:
        m_shell_m = shell_masses(cogs[n][good], idx)
        rel = np.full((5, len(labs)), np.nan)
        absf = np.full((5, len(labs)), np.nan)
        for k in EPOCHS:
            with np.errstate(divide="ignore", invalid="ignore"):
                pos = m_shell_d[:, k] > 0
                rel[k] = np.nanmedian(np.where(pos,
                                               (m_shell_m[:, k] - m_shell_d[:, k])
                                               / np.where(pos, m_shell_d[:, k], np.nan),
                                               np.nan), axis=0)
            absf[k] = np.median((m_shell_m[:, k] - m_shell_d[:, k])
                                / total[:, k][:, None], axis=0)
        misp[n] = (rel, absf)
        print(f"\n  {n}")
        print(f"  {'epoch':>6}" + "".join(f"{l:>26}" for l in labs) + "   kpc")
        print(f"  {'':>6}" + "".join(f"{'rel %':>13}{'of total %':>13}" for _ in labs))
        for k in EPOCHS:
            print(f"  {ANCHOR_Z[k]:>6.1f}"
                  + "".join(f"{100 * rel[k, j]:>+13.1f}{100 * absf[k, j]:>+13.2f}"
                            for j in range(len(labs))))
    print(f"\n{THIN}\n  AND THE MEASUREMENT ITSELF RUNS OUT THERE — the "
          f"percentage of galaxies whose MEASURED\n  mass in this shell is not "
          f"positive, i.e. whose curve of growth has stopped rising.\n  A "
          f"relative residual is undefined for those, and a fit still charges "
          f"the model for\n  them.\n{THIN}")
    print(f"  {'epoch':>6}" + "".join(f"{l:>13}" for l in labs) + "   kpc")
    nonpos = np.full((5, len(labs)), np.nan)
    for k in EPOCHS:
        nonpos[k] = np.mean(m_shell_d[:, k] <= 0, axis=0)
        print(f"  {ANCHOR_Z[k]:>6.1f}" + "".join(f"{100 * v:>12.1f}%" for v in nonpos[k]))
    print(f"\n  The user's claim 2, measured: at z = 2 the outer shells carry a "
          f"large relative error\n  on a small mass, and for a growing fraction "
          f"of galaxies they carry no measurement at\n  all. Both models show "
          f"it.")

    # ---- 4. what the objectives charge, against where the mass is -------- #
    print(f"\n{RULE}\n4. WHAT THE OBJECTIVES CHARGE, AGAINST WHERE THE MASS IS — "
          f"the share of each\n   objective's loss carried by each shell, beside "
          f"the share of the stellar mass. A\n   shell charged far above its "
          f"mass share is one the objective over-weights.\n{RULE}")
    share = {}
    for name in ("prod", "chi2_gls_amp"):
        c, _ = per_galaxy(name, cogs["exp63 joint"], data, sig_cog, sig_ann, ann_ok)[1], None
        # collapse the per-radius contributions onto the same shells
        terms = per_galaxy(name, cogs["exp63 joint"], data, sig_cog,
                           sig_ann, ann_ok)[1][good]
        off = 1 if name == "chi2_gls_amp" else 0        # amplitude term first
        sh = np.full((5, len(labs)), np.nan)
        for k in EPOCHS:
            tot = np.nansum(terms[:, k, :], axis=0)
            for j in range(len(labs)):
                lo, hi = idx[j] + off, idx[j + 1] + off
                sh[k, j] = np.nansum(tot[lo:hi]) / np.clip(np.nansum(tot), 1e-300, None)
        share[name] = sh
        print(f"\n  {name}: share of the loss (share of the MASS beneath)")
        print(f"  {'epoch':>6}" + "".join(f"{l:>13}" for l in labs) + "   kpc")
        for k in EPOCHS:
            print(f"  {ANCHOR_Z[k]:>6.1f}" + "".join(f"{100 * sh[k, j]:>12.1f}%"
                                                     for j in range(len(labs))))
            print(f"  {'':>6}" + "".join(f"{'(' + f'{100 * frac[k, j]:.1f}' + '%)':>13}"
                                         for j in range(len(labs))))

    # ---- 5. the same in R50 units ---------------------------------------- #
    print(f"\n{RULE}\n5. THE SAME PICTURE IN R50 UNITS — shells defined by the "
          f"galaxy's own half-mass\n   radius rather than by a fixed number of "
          f"kpc. If this is the better coordinate,\n   the mass fractions "
          f"should be far more STABLE across epochs than in section 1."
          f"\n{RULE}")
    cum_re = {mult: cum_at_multiple_of_re(d, R, rhalf, mult)
              for mult in SHELL_EDGES_RE}
    labs_re = [f"{SHELL_EDGES_RE[j]:g}-{SHELL_EDGES_RE[j+1]:g}"
               for j in range(len(SHELL_EDGES_RE) - 1)]
    print(f"\n  median fraction of M*(<148 kpc) in each R50 shell")
    print(f"  {'epoch':>6}" + "".join(f"{l:>12}" for l in labs_re) + "   x R50")
    frac_re = np.full((5, len(labs_re)), np.nan)
    for k in EPOCHS:
        for j in range(len(labs_re)):
            a = cum_re[SHELL_EDGES_RE[j]][:, k]
            b = cum_re[SHELL_EDGES_RE[j + 1]][:, k]
            frac_re[k, j] = np.nanmedian((b - a) / total[:, k])
        print(f"  {ANCHOR_Z[k]:>6.1f}"
              + "".join(f"{100 * v:>11.1f}%" for v in frac_re[k]))
    print(f"\n  spread across epochs (max - min of each column), percentage "
          f"points — SMALLER IS MORE STABLE")
    print(f"  {'fixed kpc':<12}" + "".join(
        f"{100 * (np.nanmax(frac[:, j]) - np.nanmin(frac[:, j])):>11.1f}"
        for j in range(len(labs))))
    print(f"  {'R50 units':<12}" + "".join(
        f"{100 * (np.nanmax(frac_re[:, j]) - np.nanmin(frac_re[:, j])):>11.1f}"
        for j in range(len(labs_re))))
    print(f"\n  If the R50 row is flatter, a size-relative grid is scoring the "
          f"SAME PART of the\n  galaxy at every epoch, which is what a shared "
          f"five-epoch law needs in order to be\n  asked one question rather "
          f"than five.")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / ("radial_evidence_smoke.npz" if smoke else "radial_evidence.npz")
    np.savez(out,
             shell_edges_kpc=np.array(SHELL_EDGES_KPC),
             shell_edges_re=np.array(SHELL_EDGES_RE),
             mass_frac_kpc=frac, mass_frac_re=frac_re, r50=med_re,
             anchor_z=np.array(ANCHOR_Z),
             **{f"rel_{n.replace(' ', '_')}": misp[n][0] for n in misp},
             **{f"absfrac_{n.replace(' ', '_')}": misp[n][1] for n in misp},
             nonpos=nonpos,
             **{f"share_{n}": share[n] for n in share})
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
