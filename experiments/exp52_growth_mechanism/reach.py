"""exp52 — how FAR does the transport term move stellar mass?

Adiabatic expansion is physically well grounded for massive galaxies: strong,
continued AGN/SMBH feedback offsets baryons outward, the inner potential
re-configures, and the stellar profile responds. But the established result --
theoretical and from hydro simulations -- is that this FLATTENS the inner
stellar profile. It does not build the stellar halo at 50-150 kpc.

So the question is not "does the kernel transport mass" (some transformation
that lowers the central density with time is legitimately needed) but "how far
does it carry it". This measures the reach: for the material already assembled
at the earlier epoch, where does transport take mass FROM and deliver it TO.

Mass leaving a shell is negative, arriving is positive; summed over all shells
including beyond 500 kpc the terms cancel, since transport conserves mass.

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python \
experiments/exp52_growth_mechanism/reach.py [--dev]
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/exp38_deposit_rethink"))

import stage2_multiepoch as s2                      # noqa: E402

ANCHOR_Z = [0.4, 0.7, 1.0, 1.5, 2.0]
EDGES = [0.0, 5.0, 10.0, 30.0, 50.0, 100.0, 148.0, 300.0, 500.0, 1e5]
PAIRS = [(2, 0), (4, 2)]


def enclosed(Rl, rc, gam):
    if Rl <= 0.0:
        return np.zeros_like(rc)
    if Rl >= 1e4:
        return np.ones_like(rc)
    return 1.0 - (1.0 + (Rl / rc) ** 2) ** (1.0 - gam)


def main(dev=False):
    rows = (np.load(ROOT / "experiments/exp32_full_population/outputs/"
                    "population.npz")["dev100"] if dev else None)
    s2._w_init(rows)
    gals, e = s2._W["gals"], s2._W["e"]
    theta = np.asarray(np.load(ROOT / "experiments/exp40_epoch_objective/"
                               "outputs/latestart.npz")["theta_z15"], float)[:12]
    print(f"exp52 transport REACH — adopted theta, n={len(gals)}")
    print("  Where transport takes mass FROM (negative) and delivers it TO "
          "(positive),\n  for material already assembled at the earlier "
          "epoch. Un-normalized.\n")

    for k1, k0 in PAIRS:
        shells = np.zeros(len(EDGES) - 1)
        r50 = np.zeros(2)
        n = 0
        for g in gals:
            mah = g["mah"]
            b6, _, _ = s2.theta_of(theta, g, "1ch-mof")
            gam = float(np.clip(b6[5], 1.05 + 1e-6, 6.0))
            ti, t_obs = mah["t"], mah["t_obs"]
            rc0 = np.clip(10.0 ** b6[0] * (ti / t_obs) ** b6[1], 1e-4, 1e5)
            w = e.weights(mah["z"], b6[3], b6[4]) * mah["dMh"]
            if not np.isfinite(w).all() or w.sum() <= 0:
                continue
            dM = w / w.sum()
            old = (mah["snap"] <= e.ANCHOR_SNAP[k1]).astype(float)

            def cum(t, Rl):
                fc = np.exp(-np.clip(t - ti, 0.0, None) / ti)
                rcw = np.clip(rc0 * (t / ti) ** max(b6[2], 0.0), 1e-4, 1e5)
                E = (fc * enclosed(Rl, rc0, gam)
                     + (1.0 - fc) * enclosed(Rl, rcw, gam))
                return float(np.sum(dM * old * E))

            c1 = np.array([cum(e.pe.AT[k1], R) for R in EDGES])
            c0 = np.array([cum(e.pe.AT[k0], R) for R in EDGES])
            shells += np.diff(c0) - np.diff(c1)
            tot1, tot0 = c1[-1], c0[-1]
            r50 += [np.interp(0.5 * tot1, c1, EDGES),
                    np.interp(0.5 * tot0, c0, EDGES)]
            n += 1
        r50 /= n
        drained = -shells[shells < 0].sum()
        print(f"  === z={ANCHOR_Z[k1]} -> z={ANCHOR_Z[k0]} ===")
        print(f"    {'shell (kpc)':<16}{'mass moved':>13}{'share of the '
                                                          'drained mass':>28}")
        for i in range(len(shells)):
            lo, hi = EDGES[i], EDGES[i + 1]
            lab = f"{lo:g}-{hi:g}" if hi < 1e4 else f"beyond {lo:g}"
            share = 100 * shells[i] / drained
            tag = "  <- drained" if shells[i] < 0 else ""
            print(f"    {lab:<16}{shells[i]:13.3e}{share:20.1f}%{tag}")
        beyond50 = shells[4:][shells[4:] > 0].sum()
        inner = shells[:4][shells[:4] > 0].sum()
        print(f"\n    drained mass = {100*drained/n:.2f}{'%'} of ALL deposited "
              f"stellar mass (dM sums to 1 per galaxy)")
        print(f"    half-mass radius of the assembled stars: "
              f"{r50[0]:.2f} -> {r50[1]:.2f} kpc "
              f"({100*(r50[1]/r50[0]-1):+.0f}{'%'})")
        print(f"    of the drained mass, delivered INSIDE 50 kpc:  "
              f"{100*inner/drained:5.1f}%")
        print(f"    of the drained mass, delivered BEYOND 50 kpc:  "
              f"{100*beyond50/drained:5.1f}%\n")


if __name__ == "__main__":
    main("--dev" in sys.argv)
