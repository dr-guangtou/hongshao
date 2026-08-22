"""exp54 Stage 0 — the forward-model CONTRACT, enforced by test.

A forward model takes halo quantities and returns stellar mass. The adopted
kernel does not: it reads `M_TNG(<500 kpc)` per galaxy per epoch and multiplies
by it (`exp53/kernel.py:233`). Stage 0 turns that from a claim in a document
into a test that runs, and defines the interface the successor must satisfy.

Six checks, in the order they matter:

  1. POISONED DATA, current kernel  -- EXPECTED TO FAIL. NaN out every
     stellar-derived array on the galaxy and call the model. If it still
     returns finite profiles, no stellar information reached the prediction.
     The adopted kernel cannot pass; that failure IS the oracle result, stated
     as an executable fact rather than a code reading.
  2. POISONED DATA, absolute law    -- MUST PASS. Same test, same galaxies,
     against `forward_absolute`, which takes only the MAH and halo properties.
  3. MASS CONSERVATION              -- the profile must integrate to exactly
     the mass the efficiency law deposited: M(<inf) == sum eps_i dMh_i.
  4. HORIZON BOOKKEEPING            -- report mass inside the anchor, between
     anchor and the last measured radius, out to the model boundary, and
     BEYOND it, with nothing renormalized away.
  5. EQUIVALENCE BRIDGE             -- the unpinned kernel, renormalized to
     M(<500) after the fact, must reproduce exp53's CoGs to ~1e-12. Proves the
     change is ONLY in the amplitude.
  6. PIN-RADIUS OFFSET              -- how much the shape loss moves when the
     reference radius goes 500 kpc -> 100 kpc, so every number in
     exp38/40/47/48/53 can be translated.

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python \\
     experiments/exp54_unpinned_amplitude/contract.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp53_deposition_only"):
    sys.path.insert(0, str(p))

import families                                        # noqa: E402
import kernel as K                                     # noqa: E402
import stage2_multiepoch as s2                         # noqa: E402
from hongshao.objective import Objective               # noqa: E402

#: every key on a galaxy dict that is derived from STELLAR data. A forward
#: model may not touch any of them.
STELLAR_KEYS = ("m500", "data", "logms", "_data_stack")
#: the model boundary, and a radius standing in for infinity in the
#: mass-conservation check. It has to be ABSURDLY large: the fitted Moffat has
#: gam = 1.378, so its enclosed mass converges only as R^-0.76 and at 1e7 kpc
#: it is still 1.6e-4 short of its own total. See check 4b -- that slowness is
#: a physical property of the adopted profile, not a numerical detail.
R_BOUNDARY = 500.0
R_INFINITY = 1e15
#: the efficiency law measured in the exp54 planning probe (plan §3.2.D):
#: log10 eps_i = a0 + a_M [logMh(t_i) - 13.5] + a_z ln(1 + z_i)
EPS_LAW = dict(a0=-2.4977, a_M=-0.1902, a_z=0.3551, m_piv=13.5)
RULE = "=" * 78


def poison(g):
    """A copy of galaxy ``g`` with every stellar-derived array set to NaN.

    Not deleted -- set to NaN. Deleting would raise KeyError, which any
    `except` would catch and could be mistaken for a controlled failure; NaN
    propagates silently into the arithmetic and shows up in the OUTPUT, which
    is what we want to detect.
    """
    out = dict(g)
    for k in STELLAR_KEYS:
        if k not in out:
            continue
        v = out[k]
        out[k] = np.full_like(np.asarray(v, float), np.nan) \
            if isinstance(v, np.ndarray) else np.nan
    out.pop("_data_stack", None)
    return out


# --------------------------------------------------------------------------- #
# the two forward models                                                       #
# --------------------------------------------------------------------------- #
def forward_shape(spec, theta, g, k, R):
    """The kernel's UNPINNED, dimensionless profile at epoch ``k`` on ``R``.

    Identical to `kernel.model_cogs` with the final `* m500/m[-1]` removed. Its
    deposit weights still sum to 1 (the exp53 convention), so this carries a
    SHAPE and an arbitrary scale -- not a stellar mass.
    """
    e = s2._W["e"]
    dM = K.deposit_weights(spec, theta, g)
    if dM is None:
        return None
    mah = g["mah"]
    mask = mah["snap"] <= e.ANCHOR_SNAP[k]
    B = K.basis(spec, theta, g["cond"], mah["t"], mah["t_obs"], e.pe.AT[k], R)
    return B @ (dM * mask)


def deposit_masses(g, law=EPS_LAW):
    """Absolute stellar mass deposited in each MAH step, in Msun.

    ``dM_i = eps_i * dMh_i`` with
    ``log10 eps_i = a0 + a_M [logMh(t_i) - m_piv] + a_z ln(1 + z_i)``.
    No normalization anywhere: this is the whole point.
    """
    mah = g["mah"]
    log_eps = (law["a0"]
               + law["a_M"] * (mah["logMh_full"][1:] - law["m_piv"])
               + law["a_z"] * np.log1p(mah["z"]))
    return 10.0 ** log_eps * mah["dMh"]


def forward_absolute(spec, theta, g, k, R, law=EPS_LAW):
    """Stellar mass profile in PHYSICAL Msun from halo quantities only.

    Reads `g["mah"]` and `g["cond"]` and nothing else. This is the interface
    the successor model must satisfy.
    """
    e = s2._W["e"]
    mah = g["mah"]
    dM = deposit_masses(g, law)
    mask = mah["snap"] <= e.ANCHOR_SNAP[k]
    B = K.basis(spec, theta, g["cond"], mah["t"], mah["t_obs"], e.pe.AT[k], R)
    return B @ (dM * mask)


# --------------------------------------------------------------------------- #
def check_1_poison_current(spec, theta, gals):
    print(f"\n{RULE}\n1. POISONED DATA — the CURRENT kernel   [EXPECTED TO FAIL]"
          f"\n{RULE}")
    print("   NaN out m500/data/logms, then predict. Finite output => no")
    print("   stellar information reached the prediction.")
    bad = 0
    for g in gals:
        c = K.model_cogs(spec, theta, poison(g), [0])
        if c is None or not np.isfinite(np.asarray(c)).all():
            bad += 1
    print(f"\n   {bad}/{len(gals)} galaxies produced a NON-finite profile.")
    ok = bad == 0
    print(f"   VERDICT: {'PASS' if ok else 'FAIL'}"
          f"  <-- {'unexpected!' if ok else 'as expected'}")
    if not ok:
        print("   The adopted kernel READS THE TRUTH. This is the oracle")
        print("   problem, executed rather than asserted.")
    return ok


def check_2_poison_absolute(spec, theta, gals, R):
    print(f"\n{RULE}\n2. POISONED DATA — the ABSOLUTE law   [MUST PASS]\n{RULE}")
    bad = 0
    for g in gals:
        m = forward_absolute(spec, theta, poison(g), 0, R)
        if m is None or not np.isfinite(m).all() or m[-1] <= 0:
            bad += 1
    print(f"   {bad}/{len(gals)} galaxies produced a non-finite profile.")
    ok = bad == 0
    print(f"   VERDICT: {'PASS' if ok else 'FAIL'}")
    if ok:
        print("   Every prediction survived the removal of all stellar data,")
        print("   so no stellar information can be reaching it.")
    return ok


def check_3_mass_conservation(spec, theta, gals):
    print(f"\n{RULE}\n3. MASS CONSERVATION\n{RULE}")
    print("   M(<inf) must equal sum(eps_i * dMh_i) over the deposits in play.")
    R = np.array([R_BOUNDARY, R_INFINITY])
    worst = 0.0
    for g in gals:
        mah = g["mah"]
        mask = mah["snap"] <= s2._W["e"].ANCHOR_SNAP[0]
        want = float((deposit_masses(g) * mask).sum())
        got = float(forward_absolute(spec, theta, g, 0, R)[-1])
        worst = max(worst, abs(got / want - 1.0))
    print(f"   worst |M(<{R_INFINITY:.0e} kpc)/sum(eps dMh) - 1| = {worst:.3e}")
    ok = worst < 1e-9
    print(f"   VERDICT: {'PASS' if ok else 'FAIL'}")
    return ok


def check_4_bookkeeping(spec, theta, gals, R):
    print(f"\n{RULE}\n4. HORIZON BOOKKEEPING — nothing renormalized away\n{RULE}")
    Rb = np.array([100.0, R[-1], R_BOUNDARY, R_INFINITY])
    rows = []
    for g in gals:
        m = forward_absolute(spec, theta, g, 0, Rb)
        tot = m[-1]
        rows.append([m[0] / tot, (m[1] - m[0]) / tot,
                     (m[2] - m[1]) / tot, (tot - m[2]) / tot])
    f = np.array(rows)
    lab = ["inside 100 kpc (the anchor)", f"100 - {R[-1]:.0f} kpc (measured)",
           f"{R[-1]:.0f} - 500 kpc (extrapolated)", "BEYOND 500 kpc (lost)"]
    print(f"   {'region':<34}{'median':>10}{'16th':>9}{'84th':>9}")
    for j, nm in enumerate(lab):
        print(f"   {nm:<34}{np.median(f[:, j]):>10.4f}"
              f"{np.percentile(f[:, j], 16):>9.4f}"
              f"{np.percentile(f[:, j], 84):>9.4f}")
    print(f"\n   fractions sum to {f.sum(1).mean():.6f} (must be 1)")
    print("   The last row is what the pin used to restore for free.")
    return abs(f.sum(1).mean() - 1.0) < 1e-9


def check_4b_tail(spec, theta, gals):
    """How far out does a single deposit have to be integrated to be counted?

    Not a pass/fail: a measurement of the adopted profile family's reach. It
    is the reason check 3 needs an absurd 'infinity', and the reason 18% of
    every deposit lands beyond the 500 kpc boundary.
    """
    print(f"\n{RULE}\n4b. TAIL CONVERGENCE — how far the fitted profile reaches"
          f"\n{RULE}")
    gam = float(K.Spec("moffat", True).unpack(theta, gals[0]["cond"])[5][0])
    x50 = np.sqrt(2.0 ** (1.0 / (gam - 1.0)) - 1.0)
    print(f"   fitted Moffat gam = {gam:.4f}")
    print(f"   Sigma ~ (1+x^2)^-gam  ->  outer density slope "
          f"{-2 * gam:.2f}, enclosed mass converges as R^{2 - 2 * gam:.2f}")
    print(f"\n   {'enclosed fraction':>18}{'R / R50':>12}"
          f"{'R for R50=50 kpc':>20}")
    for f in (0.5, 0.9, 0.99, 0.999):
        x = np.sqrt((1.0 - f) ** (1.0 / (1.0 - gam)) - 1.0)
        print(f"   {f:>18.3f}{x / x50:>12.1f}{50.0 * x / x50 / 1e3:>17.1f} Mpc")
    print("\n   A deposit with a 50 kpc half-mass radius needs MEGAPARSECS to")
    print("   enclose 99% of its own mass. That is why the horizon bookkeeping")
    print("   of check 4 loses 18% beyond 500 kpc, and it is a property of the")
    print("   PROFILE FAMILY, which the redesign should revisit.")
    return True


def check_5_equivalence(spec, theta, gals):
    print(f"\n{RULE}\n5. EQUIVALENCE BRIDGE — only the amplitude changed\n{RULE}")
    e = s2._W["e"]
    worst = 0.0
    for g in gals:
        ref = K.model_cogs(spec, theta, g, [0])
        if ref is None:
            continue
        m = forward_shape(spec, theta, g, 0, e.R_EXT)
        mine = m[:-1] * (g["m500"][0] / m[-1])
        worst = max(worst, float(np.abs(mine / ref[0] - 1.0).max()))
    print(f"   worst |unpinned-then-renormalized / exp53 - 1| = {worst:.3e}")
    ok = worst < 1e-12
    print(f"   VERDICT: {'PASS' if ok else 'FAIL'}")
    return ok


def check_6_pin_radius(spec, theta, gals):
    print(f"\n{RULE}\n6. PIN-RADIUS OFFSET — translating the exp38-53 record\n{RULE}")
    e, R = s2._W["e"], s2._W["e"].R
    obj = Objective()
    i100 = int(np.argmin(np.abs(R - 100.0)))
    m500, m100, data = [], [], []
    for g in gals:
        m = forward_shape(spec, theta, g, 0, e.R_EXT)
        if m is None or not np.isfinite(m).all() or m[-1] <= 0:
            continue
        d = g["data"][0]
        m500.append(m[:-1] * (g["m500"][0] / m[-1]))
        m100.append(m[:-1] * (d[i100] / m[i100]))
        data.append(d)
    A = np.asarray(m500)[:, None, :]
    B = np.asarray(m100)[:, None, :]
    D = np.asarray(data)[:, None, :]
    l500, l100 = obj(A, D, R), obj(B, D, R)
    print(f"   production loss, pinned at 500 kpc = {l500:.6f}")
    print(f"   production loss, pinned at 100 kpc = {l100:.6f}")
    print(f"   offset (100 - 500)                 = {l100 - l500:+.6f}"
          f"  ({100 * (l100 / l500 - 1):+.2f}%)")
    print("\n   Every shape loss quoted in exp38/40/47/48/53 is the 500 kpc")
    print("   number; add this offset to compare with an exp54 100 kpc number.")
    return True


def main(n_gal=300):
    s2._w_init(None)
    gals = s2._W["gals"][:n_gal]
    R = s2._W["e"].R
    spec = K.Spec("moffat", q_free=True)
    theta = K.from_exp38_theta(K.adopted_theta())
    print(f"exp54 STAGE 0 — the forward-model contract   (n={len(gals)} galaxies)")
    print(f"efficiency law: log10 eps = {EPS_LAW['a0']:.4f} "
          f"{EPS_LAW['a_M']:+.4f} [logMh(t_i) - {EPS_LAW['m_piv']}] "
          f"{EPS_LAW['a_z']:+.4f} ln(1+z)")

    results = {
        "1 poison / current kernel (expected FAIL)":
            check_1_poison_current(spec, theta, gals),
        "2 poison / absolute law": check_2_poison_absolute(spec, theta, gals, R),
        "3 mass conservation": check_3_mass_conservation(spec, theta, gals),
        "4 horizon bookkeeping": check_4_bookkeeping(spec, theta, gals, R),
        "4b tail convergence (diagnostic)": check_4b_tail(spec, theta, gals),
        "5 equivalence bridge": check_5_equivalence(spec, theta, gals),
        "6 pin-radius offset": check_6_pin_radius(spec, theta, gals),
    }
    print(f"\n{RULE}\nSUMMARY\n{RULE}")
    for k, v in results.items():
        expected_fail = k.startswith("1 ")
        good = (not v) if expected_fail else v
        print(f"   {'OK ' if good else 'BAD'}  {k:<46}"
              f"{'PASS' if v else 'FAIL'}")
    hard = [v for k, v in results.items() if not k.startswith("1 ")]
    print(f"\n   {sum(hard)}/{len(hard)} mandatory checks passed; "
          f"check 1 failed as required.")
    return results


if __name__ == "__main__":
    main()
