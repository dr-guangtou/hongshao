"""exp50 — how does the model grow a massive galaxy's outskirts?

The kernel is a TRANSPORT model: mass deposited at time t_i with scale rc0
later has a fraction (1-fc) expanded to rc0*(t_k/t_i)^q. So between two epochs
the model can grow an annulus two ways -- by depositing NEW material, or by
moving material it already had.

The established physical picture for massive galaxies is that late-time
outskirt growth is dominated by EX-SITU accretion: continued dry, mostly minor
mergers depositing new stars at large radii. It is NOT the redistribution of
stars already assembled in the centre. If the model's growth is transport-
dominated, the model is telling a story incompatible with that understanding,
and it would do so invisibly, because the objective only ever sees the summed
profile.

This measures the split, exactly. For an epoch pair k1 (earlier) -> k0 (later)
split the deposits into

  old  those already present at k1 (snap <= ANCHOR_SNAP[k1])
  new  those arriving between k1 and k0

and write the change in a normalized aperture/annulus mass as three exact terms

  dM_hat = N_k0*(A_old_k0 - A_old_k1)   TRANSPORT      same material, moved
         + (N_k0 - N_k1)*A_old_k1       RENORMALIZATION  the M(<500) pinning
         + N_k0*A_new_k0                NEW DEPOSITION

where N_k = m500_k / M_k(500) is the kernel's own normalization. The
UN-NORMALIZED split (transport vs new deposition alone) is reported too, since
that is the physics question with the bookkeeping removed.

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python \
experiments/exp50_growth_mechanism/decompose.py [--dev]
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
PAIRS = [(2, 0), (4, 2), (2, 1), (1, 0), (3, 2), (4, 3)]
APERTURES = [("M(<10)", 0.0, 10.0), ("M(<30)", 0.0, 30.0),
             ("M[50,100]", 50.0, 100.0), ("M[100,148]", 100.0, 148.0),
             ("M(<148)", 0.0, 148.0), ("M(<500)", 0.0, 500.0)]


def enclosed(Rl, rc, gam):
    """Moffat enclosed-mass fraction inside Rl for scale rc."""
    if Rl <= 0.0:
        return np.zeros_like(rc)
    return 1.0 - (1.0 + (Rl / rc) ** 2) ** (1.0 - gam)


def galaxy_terms(g, theta, e):
    """Per-galaxy: for each epoch, the un-normalized enclosed mass at every
    aperture edge, split into 'present at k1' and 'arrived later'."""
    mah = g["mah"]
    b6, _, _ = s2.theta_of(theta, g, "1ch-mof")
    gam = float(np.clip(b6[5], 1.05 + 1e-6, 6.0))
    ti, t_obs = mah["t"], mah["t_obs"]
    rc0 = np.clip(10.0 ** b6[0] * (ti / t_obs) ** b6[1], 1e-4, 1e5)
    w = e.weights(mah["z"], b6[3], b6[4]) * mah["dMh"]
    if not np.isfinite(w).all() or w.sum() <= 0:
        return None
    dM = w / w.sum()
    return dict(dM=dM, rc0=rc0, gam=gam, ti=ti, snap=mah["snap"])


def enclosed_at(t, terms, sel, Rl):
    """Un-normalized enclosed mass inside Rl at cosmic time t, summing only the
    deposits selected by boolean `sel`. Transport is evaluated AT t."""
    ti, rc0, gam = terms["ti"], terms["rc0"], terms["gam"]
    fc = np.exp(-np.clip(t - ti, 0.0, None) / ti)
    rcw = np.clip(rc0 * (t / ti) ** max(terms["q"], 0.0), 1e-4, 1e5)
    E = fc * enclosed(Rl, rc0, gam) + (1.0 - fc) * enclosed(Rl, rcw, gam)
    return float(np.sum(terms["dM"] * sel * E))


def main(dev=False):
    rows = (np.load(ROOT / "experiments/exp32_full_population/outputs/"
                    "population.npz")["dev100"] if dev else None)
    s2._w_init(rows)
    gals, e = s2._W["gals"], s2._W["e"]
    R = e.R
    theta = np.asarray(np.load(ROOT / "experiments/exp40_epoch_objective/"
                               "outputs/latestart.npz")["theta_z15"], float)[:12]
    print(f"exp50 growth decomposition — adopted 1ch-mof theta, n={len(gals)}")
    print(f"  transport exponent q = {theta[2]:.4f}; efficiency window "
          f"mu={theta[3]:.4f} sig={theta[4]:.4f} "
          f"(peak z={np.exp(theta[3])-1:.2f})\n")

    for k1, k0 in PAIRS:
        acc = {a[0]: np.zeros(5) for a in APERTURES}   # tr, renorm, new, model, truth
        acc["_m500"] = np.zeros(4)   # truth m500 at k1,k0; model un-normalized at k1,k0
        for g in gals:
            t = galaxy_terms(g, theta, e)
            if t is None:
                continue
            b6, _, _ = s2.theta_of(theta, g, "1ch-mof")
            t["q"] = b6[2]
            old = (t["snap"] <= e.ANCHOR_SNAP[k1]).astype(float)
            new = ((t["snap"] <= e.ANCHOR_SNAP[k0]) &
                   (t["snap"] > e.ANCHOR_SNAP[k1])).astype(float)
            tk1, tk0 = e.pe.AT[k1], e.pe.AT[k0]
            N1 = g["m500"][k1] / enclosed_at(tk1, t, old, 500.0)
            all0 = (t["snap"] <= e.ANCHOR_SNAP[k0]).astype(float)
            M500_k1 = enclosed_at(tk1, t, old, 500.0)
            M500_k0 = enclosed_at(tk0, t, all0, 500.0)
            N0 = g["m500"][k0] / M500_k0
            acc["_m500"] += np.array([g["m500"][k1], g["m500"][k0],
                                      M500_k1, M500_k0])
            for name, r_in, r_out in APERTURES:
                def ann(tt, sel):
                    return (enclosed_at(tt, t, sel, r_out)
                            - enclosed_at(tt, t, sel, r_in))
                A_old_k1 = ann(tk1, old)
                A_old_k0 = ann(tk0, old)
                A_new_k0 = ann(tk0, new)
                tr = N0 * (A_old_k0 - A_old_k1)
                rn = (N0 - N1) * A_old_k1
                nw = N0 * A_new_k0
                d1 = g["data"][k1]
                d0 = g["data"][k0]
                tru = ((np.interp(r_out, R, d0) - np.interp(r_in, R, d0))
                       - (np.interp(r_out, R, d1) - np.interp(r_in, R, d1)))
                acc[name] += np.array([tr, rn, nw, tr + rn + nw, tru])
        # How much does the M(<500) pinning have to inflate? If the model's
        # own deposits kept pace with the measured mass growth this is 1.0.
        mg = acc["_m500"]
        print(f"  === z={ANCHOR_Z[k1]} -> z={ANCHOR_Z[k0]}  (population-summed) ===")
        print(f"    measured M(<500) growth  x{mg[1]/mg[0]:.3f}   "
              f"model's OWN deposit growth  x{mg[3]/mg[2]:.3f}   "
              f"-> pinning inflates by x{(mg[1]/mg[0])/(mg[3]/mg[2]):.3f}")
        print(f"    {'aperture':<12}{'model dM':>13}{'truth dM':>13}"
              f"{'TRANSPORT':>12}{'NEW DEP':>11}{'transport share':>17}")
        for name, _, _ in APERTURES:
            tr, rn, nw, mod, tru = acc[name]
            # A share is only meaningful when both terms push the same way;
            # in the centre transport is NEGATIVE (material leaves) and a
            # ratio would be nonsense.
            if tr > 0 and nw > 0:
                lab = f"{100 * tr / (tr + nw):15.1f}%"
            else:
                lab = "  opposite signs" if tr * nw < 0 else "            n/a"
            print(f"    {name:<12}{mod:13.3e}{tru:13.3e}"
                  f"{tr:12.3e}{nw:11.3e}{lab:>16}")
        print("    transport share = transport / (transport + new deposition),")
        print("    i.e. the model's OWN growth with the M(<500) pinning removed.\n")


if __name__ == "__main__":
    main("--dev" in sys.argv)
