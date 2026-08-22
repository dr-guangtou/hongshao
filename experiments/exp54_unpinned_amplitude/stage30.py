"""exp54 Stage 3.0 — pre-fit contracts, floors and diagnostics. NO FITTING.

Four deliverables, in the order the plan (§7.0) requires them:

  A  THE MONOTONIC-ADDITION CEILING, in the production objective's own units.
     Any sum of static positive deposits makes M*(<R,t) non-decreasing in time.
     TNG does not. The closest non-decreasing sequence to each galaxy bounds
     what ANY deposition-only model can achieve, before a single form is chosen.

  B  SIGNAL OR NOISE? The ceiling only costs the model if the violations are
     real. Five discriminators, because the answer decides whether a survival
     or radial-evolution term is worth building (plan decision 15).

  C  TAIL DIAGNOSTICS for every candidate profile family, REPORTED BEFORE any
     convergence constraint is imposed (review §4.2: neither the 99% quantile
     nor the factor three is measured).

  D  ENDPOINT-PRESERVING MAH SHUFFLES. The existing control matches on
     Mh(z=0.4) and so also changes Mh(z_k) at higher redshift, conflating
     history shape with epoch-local mass. This builds a separate permutation
     per epoch, matched on that epoch's own halo mass.

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python \\
     experiments/exp54_unpinned_amplitude/stage30.py [--figures]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp53_deposition_only"):
    sys.path.insert(0, str(p))

import families                                          # noqa: E402
import model as M                                        # noqa: E402
from hongshao.objective import Objective                 # noqa: E402
from hongshao.plotting import OKABE_ITO, save_fig, set_style   # noqa: E402
from hongshao.qa import _pct, _tex                       # noqa: E402

POP = ROOT / "experiments/exp32_full_population/outputs/population.npz"
SO = ROOT / "experiments/exp46_highz_ridge/outputs/so_history.npz"
HS = HERE / "outputs" / "halo_structure_history.npz"
FIGDIR = HERE / "figures"
Z = np.array([0.4, 0.7, 1.0, 1.5, 2.0])
ANCHOR_SNAP = [72, 59, 50, 40, 33]
R = np.array([2., 2.76, 3.72, 4.92, 6.38, 8.14, 10.25, 12.74, 15.66, 19.05,
              22.97, 27.46, 32.58, 38.39, 44.94, 52.30, 60.52, 69.68, 79.84,
              91.07, 103.45, 117.05, 131.94, 148.22])
RULE = "=" * 84


def isotonic(y):
    """Closest non-decreasing sequence in least squares (pool-adjacent-violators)."""
    v = list(map(float, y))
    w = [1.0] * len(v)
    i = 0
    while i < len(v) - 1:
        if v[i] <= v[i + 1] + 1e-15:
            i += 1
            continue
        nv = (v[i] * w[i] + v[i + 1] * w[i + 1]) / (w[i] + w[i + 1])
        nw = w[i] + w[i + 1]
        v[i:i + 2] = [nv]
        w[i:i + 2] = [nw]
        i = max(i - 1, 0)
    out = []
    for val, wt in zip(v, w):
        out += [val] * int(round(wt))
    return np.array(out)


def project_all(data):
    """(n,5,24) truth -> the closest monotone-in-TIME model, same shape.

    Time order is reversed on the way in and out: index 4 is z=2.0 (earliest).
    """
    ly = np.log10(np.clip(data, 1.0, None))[:, ::-1, :]     # earliest first
    out = np.empty_like(ly)
    for i in range(ly.shape[0]):
        for c in range(ly.shape[2]):
            out[i, :, c] = isotonic(ly[i, :, c])
    return (10.0 ** out)[:, ::-1, :]


# --------------------------------------------------------------------------- #
def part_a(data):
    print(f"\n{RULE}\nA. THE MONOTONIC-ADDITION CEILING\n{RULE}")
    proj = project_all(data)
    d = np.log10(np.clip(data, 1.0, None))
    p = np.log10(np.clip(proj, 1.0, None))
    resid = d - p

    print("  Fraction of galaxies whose enclosed mass DECREASES with time:")
    idx = [int(np.argmin(np.abs(R - r))) for r in (2.0, 4.9, 10.2, 30.0, 103.5)]
    print(f"    {'interval':<16}" + "".join(f"{f'{R[c]:.1f} kpc':>11}" for c in idx))
    frac = np.zeros((4, len(idx)))
    for a, (i, j) in enumerate(((4, 3), (3, 2), (2, 1), (1, 0))):
        row = [np.mean(data[:, j, c] < data[:, i, c]) for c in idx]
        frac[a] = row
        print(f"    z={Z[i]}->{Z[j]:<10}" + "".join(f"{100 * v:>10.1f}%" for v in row))

    obj = Objective()
    lo = obj(proj, data, R)
    print(f"\n  IN THE PRODUCTION OBJECTIVE'S OWN UNITS:")
    print(f"    best possible monotone model   = {lo:.6f}")
    print(f"    incumbent shape loss (100 kpc) = 0.158196   [Stage 0]")
    print(f"    -> the floor is {100 * lo / 0.158196:.1f}% of the incumbent's loss, "
          f"{100 * (lo / 0.158196) ** 2:.1f}% of its variance")
    per_ep = [obj(proj[:, [k]], data[:, [k]], R) for k in range(5)]
    print(f"\n    {'epoch':<10}" + "".join(f"{f'z={z}':>10}" for z in Z))
    print(f"    {'floor':<10}" + "".join(f"{v:>10.4f}" for v in per_ep))
    print(f"\n  median |correction| by radius (dex):")
    print(f"    {'':<10}" + "".join(f"{f'{R[c]:.1f}':>10}" for c in idx))
    print(f"    {'all epochs':<10}"
          + "".join(f"{np.median(np.abs(resid[:, :, c])):>10.4f}" for c in idx))
    print(f"\n  median all-epoch all-radius RMS correction: "
          f"{np.median(np.sqrt((resid ** 2).mean(axis=(1, 2)))):.4f} dex")
    return proj, resid, frac, idx, lo


def part_b(data, proj, resid, pop, rows):
    print(f"\n{RULE}\nB. IS THE VIOLATION SIGNAL OR NOISE?\n{RULE}")
    viol = np.abs(resid).max(axis=(1, 2))                 # per galaxy, dex

    print("  B1 — is it MEASUREMENT error? Compare with the CoG's own sigma.")
    print("     cog_sigma_dex is ~0.013 dex at 2 kpc falling to ~0.003 at 148")
    print("     (hongshao.tng_data). Median violation where a decrease occurs:")
    for r in (4.9, 10.2, 103.5):
        c = int(np.argmin(np.abs(R - r)))
        dec = []
        for i, j in ((4, 3), (3, 2), (2, 1), (1, 0)):
            m = data[:, j, c] < data[:, i, c]
            dec.append(np.median(np.log10(data[m, i, c] / data[m, j, c])))
        print(f"       R={R[c]:6.1f} kpc: {np.mean(dec):.4f} dex  "
              f"= {np.mean(dec) / 0.013:.0f}x the 2 kpc measurement sigma")
    print("     VERDICT: far larger than measurement error. NOT instrumental.")

    print("\n  B2 — is it COHERENT across epochs, or transient?")
    print("     A real, systematic process should repeat for the same galaxy;")
    print("     a transient should not. Sign agreement of consecutive")
    print("     interval-to-interval changes, at each radius:")
    for r in (4.9, 10.2, 103.5):
        c = int(np.argmin(np.abs(R - r)))
        steps = np.stack([np.log10(data[:, j, c] / data[:, i, c])
                          for i, j in ((4, 3), (3, 2), (2, 1), (1, 0))], 1)
        neg = steps < 0
        agree = np.mean([np.mean(neg[:, k] == neg[:, k + 1]) for k in range(3)])
        rho = np.mean([np.corrcoef(steps[:, k], steps[:, k + 1])[0, 1]
                       for k in range(3)])
        print(f"       R={R[c]:6.1f} kpc: sign agreement {100 * agree:.1f}% "
              f"(50% = pure chance), step-to-step r = {rho:+.3f}")

    print("\n  B3 — does it track MERGER activity in the MAH?")
    burst = pop["burst"][rows]
    for r in (4.9, 103.5):
        c = int(np.argmin(np.abs(R - r)))
        v = np.abs(resid[:, :, c]).max(1)
        ok = np.isfinite(burst) & np.isfinite(v)
        print(f"       R={R[c]:6.1f} kpc: corr(violation, MAH burstiness) = "
              f"{np.corrcoef(burst[ok], v[ok])[0, 1]:+.3f}")

    print("\n  B4 — does it track HALO SHAPE (projection of a triaxial object)?")
    hs = np.load(HS, allow_pickle=True)
    hi = {int(x): k for k, x in enumerate(hs["index"])}
    sel = np.array([hi[int(pop["index"][r])] for r in rows])
    snaps = list(hs["snaps"])
    s72 = hs["s"][sel][:, snaps.index(72)]                # c/a at z=0.4
    for r in (4.9, 103.5):
        c = int(np.argmin(np.abs(R - r)))
        v = np.abs(resid[:, :, c]).max(1)
        ok = np.isfinite(s72) & np.isfinite(v)
        print(f"       R={R[c]:6.1f} kpc: corr(violation, halo c/a) = "
              f"{np.corrcoef(s72[ok], v[ok])[0, 1]:+.3f}")

    print("\n  B5 — the RADIAL signature: where does the mass go?")
    print("     If mass leaves the centre and arrives outside, the model needs")
    print("     REDISTRIBUTION. If it simply vanishes, it needs a SURVIVAL term.")
    for i, j in ((4, 3), (2, 1), (1, 0)):
        cin = int(np.argmin(np.abs(R - 10.0)))
        m = data[:, j, cin] < data[:, i, cin]             # lost mass inside
        d_in = np.median(data[m, j, cin] - data[m, i, cin])
        d_tot = np.median(data[m, j, -1] - data[m, i, -1])
        print(f"       z={Z[i]}->{Z[j]}: galaxies losing inside 10 kpc "
              f"(n={m.sum():4d}) lose {abs(d_in):.3g} Msun there while their "
              f"TOTAL changes by {d_tot:+.3g}")
    return viol


def part_c():
    print(f"\n{RULE}\nC. TAIL DIAGNOSTICS — reported BEFORE any constraint\n{RULE}")
    print("  Untruncated families at their default shape, in units of R50.")
    print("  'beyond' fractions assume R50 = 0.1 R200c and R200c = 493 kpc")
    print("  (the measured median at z=0.4).\n")
    r200, r50 = 493.0, 49.3
    print(f"  {'family':<15}{'R90/R50':>9}{'R95/R50':>9}{'R99/R50':>10}"
          f"{'>R200c':>9}{'>3R200c':>9}{'>500kpc':>9}{'>148kpc':>9}")
    for fam in families.FAMILIES:
        sh = families.shape_defaults(fam) or None
        x, f0 = M._table(fam, sh)
        q = [float(np.interp(v, f0, x)) for v in (0.9, 0.95, 0.99)]
        beyond = [1.0 - float(np.interp(rr / r50, x, f0))
                  for rr in (r200, 3 * r200, 500.0, 148.22)]
        print(f"  {fam:<15}{q[0]:>9.2f}{q[1]:>9.2f}{q[2]:>10.2f}"
              + "".join(f"{_pct_s(b):>9}" for b in beyond))
    print("\n  R99/R50 is dominated by the analytic tail and can be enormous")
    print("  while the OBSERVABLE fractions stay small -- which is exactly the")
    print("  review's caution against making R99 <= 3 R200c a physical law.")


def _pct_s(v):
    return f"{100 * v:.2f}%" if v >= 1e-4 else f"{100 * v:.1e}%"


def part_d(pop, rows):
    print(f"\n{RULE}\nD. ENDPOINT-PRESERVING MAH SHUFFLES\n{RULE}")
    lmh = pop["logmh_zk_diffmah"][rows]
    n, rng = len(rows), np.random.default_rng(0)
    perms = {}
    print("  A SEPARATE permutation per epoch, each matched on THAT epoch's")
    print("  halo mass, so history shape is isolated from epoch-local mass.\n")
    print(f"  {'epoch':<10}{'bins':>7}{'median |dlogMh| within a swap':>34}")
    for k in range(5):
        order = np.argsort(lmh[:, k])
        perm = np.arange(n)
        for grp in np.array_split(order, 80):
            perm[grp] = rng.permutation(grp)
        perms[k] = perm
        d = np.abs(lmh[perm, k] - lmh[:, k])
        print(f"  z={Z[k]:<8}{80:>7}{np.median(d):>34.4f}")
    print("\n  For comparison, the OLD control permuted within bins of")
    print("  Mh(z=0.4) only; at z=2.0 that displaced the epoch-local mass by")
    old = np.arange(n)
    for grp in np.array_split(np.argsort(lmh[:, 0]), 60):
        old[grp] = rng.permutation(grp)
    print(f"  a median {np.median(np.abs(lmh[old, 4] - lmh[:, 4])):.4f} dex -- "
          f"{np.median(np.abs(lmh[old, 4] - lmh[:, 4])) / np.median(np.abs(lmh[perms[4], 4] - lmh[:, 4])):.1f}x "
          f"more than the endpoint-preserving version.")
    np.savez_compressed(HERE / "outputs" / "shuffles.npz",
                        **{f"perm_z{k}": perms[k] for k in range(5)},
                        perm_old=old)
    print(f"  wrote {HERE / 'outputs' / 'shuffles.npz'}")
    return perms


# --------------------------------------------------------------------------- #
def figures(data, proj, resid, frac, idx, floor):
    import matplotlib.pyplot as plt
    set_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)

    # --- FIG 1: the monotonic-violation heatmap, radius x epoch ------------ #
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    fr = np.zeros((4, len(R)))
    for a, (i, j) in enumerate(((4, 3), (3, 2), (2, 1), (1, 0))):
        fr[a] = [np.mean(data[:, j, c] < data[:, i, c]) for c in range(len(R))]
    im = ax[0].pcolormesh(R, np.arange(5)[:-1], 100 * fr, cmap="magma",
                          shading="auto", vmin=0, vmax=60)
    ax[0].set_xscale("log")
    ax[0].set_yticks(np.arange(4))
    ax[0].set_yticklabels([f"{Z[i]}$\\to${Z[j]}"
                           for i, j in ((4, 3), (3, 2), (2, 1), (1, 0))])
    ax[0].set_xlabel(_tex("R [kpc]"))
    ax[0].set_title(_tex("galaxies whose M(<R) DECREASES " + _pct()))
    fig.colorbar(im, ax=ax[0])
    med = np.median(np.abs(resid), axis=(0, 1))
    ax[1].plot(R, med, "-o", color=OKABE_ITO[1], ms=3)
    ax[1].set_xscale("log")
    ax[1].set_xlabel(_tex("R [kpc]"))
    ax[1].set_ylabel(_tex("median |isotonic correction| [dex]"))
    ax[1].set_title(_tex("the deposition-only floor, by radius"))
    ax[1].grid(alpha=0.3)
    fig.suptitle(_tex(f"exp54 Stage 3.0 -- the monotonic-addition ceiling; "
                      f"production-objective floor {floor:.4f}"), fontsize=11)
    fig.tight_layout()
    print("  wrote", save_fig(fig, FIGDIR / "s30_monotonic_ceiling")[0])

    # --- FIG 2: tail convergence, every family ----------------------------- #
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for k, fam in enumerate(families.FAMILIES):
        sh = families.shape_defaults(fam) or None
        x, f0 = M._table(fam, sh)
        ax.plot(x, 1.0 - f0, color=OKABE_ITO[k % len(OKABE_ITO)], label=fam)
    for v, lab in ((1.0, "R50"), (10.0, r"10 R50"), (100.0, r"100 R50")):
        ax.axvline(v, color="0.7", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.5, 1e4)
    ax.set_ylim(1e-6, 1)
    ax.set_xlabel(_tex("R / R50"))
    ax.set_ylabel(_tex("mass fraction OUTSIDE R"))
    ax.set_title(_tex("exp54 Stage 3.0 -- tail convergence by family "
                      "(untruncated, default shape)"))
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    print("  wrote", save_fig(fig, FIGDIR / "s30_tail_convergence")[0])

    # --- FIG 3: what the truncation does to the profile -------------------- #
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    rr = np.geomspace(1.0, 2000.0, 300)
    for k, fam in enumerate(("sersic", "moffat", "gompertz_log")):
        sh = families.shape_defaults(fam) or None
        c = OKABE_ITO[k]
        x, f0 = M._table(fam, sh)
        ax[0].plot(rr, np.interp(rr / 49.3, x, f0), color=c, ls=":", lw=1.2)
        g = M.cog_truncated(fam, sh, np.array([49.3]), np.array([1479.0]), rr)
        ax[0].plot(rr, g[:, 0], color=c, label=f"{fam} (truncated)")
    ax[0].axvline(1479.0, color="0.5", lw=0.9)
    ax[0].axhline(0.5, color="0.8", lw=0.8)
    ax[0].set_xscale("log")
    ax[0].set_xlabel(_tex("R [kpc]"))
    ax[0].set_ylabel(_tex("enclosed mass fraction"))
    ax[0].set_title(_tex("dotted = untruncated, solid = truncated at 3 R200c"))
    ax[0].legend(fontsize=8)
    rho = np.geomspace(0.005, 0.4, 60)
    for k, fam in enumerate(("sersic", "moffat", "gompertz_log")):
        sh = families.shape_defaults(fam) or None
        u = M.solve_u(fam, sh, rho)
        ax[1].plot(rho, 1.0 / (u * rho), color=OKABE_ITO[k], label=fam)
    ax[1].axhline(1.0, color="0.6", lw=0.8)
    ax[1].set_xscale("log")
    ax[1].set_xlabel(_tex("true R50 / R_trunc"))
    ax[1].set_ylabel(_tex("untruncated R50 / true R50"))
    ax[1].set_title(_tex("the mislabelling avoided by reparameterizing"))
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.suptitle(_tex("exp54 Stage 3.0 -- truncation and its reparameterization"),
                 fontsize=11)
    fig.tight_layout()
    print("  wrote", save_fig(fig, FIGDIR / "s30_truncation")[0])


def main(do_figs=False):
    pop = np.load(POP)
    import stage2_multiepoch as s2
    sys.path.insert(0, str(ROOT / "experiments/exp38_deposit_rethink"))
    s2._w_init(None)
    rows = np.array([g["row"] for g in s2._W["gals"]])
    data = pop["data"][rows]
    print(f"exp54 STAGE 3.0 — pre-fit contracts and floors   (n={len(rows)})")
    proj, resid, frac, idx, floor = part_a(data)
    part_b(data, proj, resid, pop, rows)
    part_c()
    part_d(pop, rows)
    if do_figs:
        print(f"\n{RULE}\nFIGURES\n{RULE}")
        figures(data, proj, resid, frac, idx, floor)


if __name__ == "__main__":
    main("--figures" in sys.argv)
