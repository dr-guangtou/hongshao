"""exp53 — how much work is the `M(<500 kpc)` normalization doing?

Raised by the user (2026-08-22): is pinning every galaxy's `M(<500)` to the
truth at every epoch too strong an assumption, and does it cost more than it
buys?

The normalization was introduced in exp35 to close the aperture-horizon trap
(a model could otherwise buy loss by pushing mass past the outermost measured
radius). It does that. But it also has consequences that no judged metric in
this program can see, because EVERY metric operates on the POST-normalization
profiles. Three of them are measured here.

1. **The pinned quantity is partly EXTRAPOLATED.** The measured CoG ends at
   148 kpc; `M(<500)` comes from an exp34 power-law tail fitted to it and
   evaluated at 500 kpc.

2. **The normalization silently restores mass thrown out of the aperture.** The
   kernel deposits unit mass; whatever lands beyond 500 kpc is discarded and
   the rest is scaled back up, so a model is not charged for over-reaching.

3. **The kernel carries NO absolute stellar mass at all** — `dM` is normalized
   to sum to 1 by construction, so 100% of the stellar mass comes from the
   pinned value, and so does the mass GROWTH between epochs. What the model
   does predict, and what nothing scores, is the FRACTION of its lifetime
   deposits in place inside 500 kpc by each epoch. Comparing that accumulation
   against the truth's growth is a free out-of-model test: the fit never sees
   it.

Run: `HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python \
experiments/exp53_deposition_only/pinning.py [--cells a,b]`
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/exp38_deposit_rethink"))

import fit as F                                      # noqa: E402
import kernel as K                                   # noqa: E402
import stage2_multiepoch as s2                       # noqa: E402

Z = [0.4, 0.7, 1.0, 1.5, 2.0]
DEFAULT = ("moffat-qfree", "moffat-q0", "gompertz_log-q0", "richards-q0",
           "sersic-q0", "gauss-q0")


def main(cells=None):
    s2._w_init(None)
    gals, e = s2._W["gals"], s2._W["e"]
    done = F._load(F._store())
    lookup = dict(F.all_cells())
    m500 = np.stack([g["m500"] for g in gals])
    m148 = np.stack([[g["data"][k][-1] for k in range(5)] for g in gals])

    print(f"exp53 — the M(<500) pinning, audited (n={len(gals)})\n")
    print("(1) The pinned quantity is not measured: the CoG ends at 148 kpc and")
    print("    M(<500) is an exp34 power-law tail evaluated at 500 kpc.")
    print("    Fraction of M(<500) beyond the last MEASURED radius:")
    for k in range(5):
        f = 1.0 - m148[:, k] / m500[:, k]
        print(f"      z={Z[k]}: median {100*np.median(f):5.1f}%  "
              f"(16-84 {100*np.percentile(f, 16):4.1f}-"
              f"{100*np.percentile(f, 84):4.1f}%)")

    rows = {}
    for nm in (cells or DEFAULT):
        b = F.best_cell(done, nm)
        if b is None:
            continue
        sp, th = lookup[nm], b[0]
        acc = np.zeros(5)
        fin = np.full((len(gals), 5), np.nan)
        for i, g in enumerate(gals):
            dM = K.deposit_weights(sp, th, g)
            if dM is None:
                continue
            for k in range(5):
                mask = g["mah"]["snap"] <= e.ANCHOR_SNAP[k]
                B = K.basis(sp, th, g["cond"], g["mah"]["t"], g["mah"]["t_obs"],
                            e.pe.AT[k], e.R_EXT)
                inside = (B @ (dM * mask))[-1]
                acc[k] += inside * g["m500"][0]        # arbitrary common scale
                w = (dM * mask).sum()
                fin[i, k] = inside / w if w > 0 else np.nan
        rows[nm] = (acc, fin)

    print("\n(2) Mass thrown BEYOND 500 kpc and silently restored by the")
    print("    normalization — the inflation factor 1/f_in (1.000 = nothing lost):")
    print(f"      {'model':<18}" + "".join(f"{f'z={z}':>10}" for z in Z))
    for nm, (_, fin) in rows.items():
        print(f"      {nm:<18}" + "".join(
            f"{1.0/np.nanmedian(fin[:, k]):10.3f}" for k in range(5)))

    print("\n(3) MASS ASSEMBLY — the model's own accumulation vs the truth's")
    print("    growth, both relative to z=2. The fit never sees this.")
    tr = m500.sum(0)
    tru = tr / tr[4]
    print(f"      {'quantity':<18}" + "".join(f"{f'z={z}':>10}" for z in Z[::-1]))
    print(f"      {'TRUTH M(<500)':<18}"
          + "".join(f"{tru[k]:10.3f}" for k in range(4, -1, -1)))
    print()
    for nm, (acc, _) in rows.items():
        rel = acc / acc[4]
        corr = np.log10(tru[0] / rel[0])
        print(f"      {nm:<18}"
              + "".join(f"{rel[k]:10.3f}" for k in range(4, -1, -1))
              + f"   pinning must correct {corr:+.3f} dex over the baseline")
    print("\n    A model whose own accumulation matched the truth would need a")
    print("    0.000 dex correction. This is a FREE out-of-model test of the")
    print("    mass-assembly history, and no judged metric in exp53 sees it.")


if __name__ == "__main__":
    cl = None
    if "--cells" in sys.argv:
        cl = sys.argv[sys.argv.index("--cells") + 1].split(",")
    main(cl)
