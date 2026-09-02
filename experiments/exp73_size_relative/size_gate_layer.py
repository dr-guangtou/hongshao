"""exp73 Block D, second use — THE STOCHASTIC LAYER scored on the size gate
(the user's decision 3, 2026-09-02).

Block D measured that every conditional-mean model's size scatter at fixed mass
is 0.2-0.6 of the truth's and collapses with redshift (exp63 R50 0.63 at z=0.4
-> 0.30 at z=2; R80 0.16). That was C16 made quantitative: a mean model is
EXPECTED to fail the width sub-gate, and the failure only becomes a verdict
when the layer's draws are scored. This script scores them.

The layer is hongshao v1's adopted Option A (`exp60 stage3d_adopt.py`,
`outputs/hongshao_v1_layer.npz`): the exp60 cube around the INCUMBENT mean
(exp57's), a formation-tilted core mixture, a size-axis draw and an additive
amplitude noise calibrated per epoch. It was built around the incumbent, not
exp63, so the mean it is scored with is the incumbent's; exp63's mean has no
layer yet. Draws are regenerated with stage3d's seeds and the frozen sig_add
(no recalibration), so this is the shipped artifact.

For the mean and for each of N_DRAW drawn populations the standard tier 2d
gate is run (`qa.evaluate` with figures off); offsets and width ratios are then
averaged over draws and their scatter across draws reported.

Run:
    HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 \\
    PYTHONPATH=. uv run python -u experiments/exp73_size_relative/size_gate_layer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E60 = ROOT / "experiments/exp60_stochastic_layer"
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term", E60, HERE):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import stage0_cost as S0                                 # noqa: E402
from stage3c_judge import Sampler                        # noqa: E402
from hongshao import qa                                  # noqa: E402

RULE, THIN = "=" * 100, "-" * 100
ANCHOR_Z = [0.4, 0.7, 1.0, 1.5, 2.0]
N_DRAW, SEED = 8, 42                                     # stage3d's
KEYS = ("R20", "R50", "R80")
OUTDIR = HERE / "outputs"


def gate_tables(cogs, data, lmh):
    out = qa.evaluate(cogs, data, F.R_GRID, ANCHOR_Z, figdir=None, figures=False,
                      verbose=False, bin_by=lmh[:, 0], halo_mass_epochs=lmh)
    return out["size_gate_ms"][0], out["size_gate_mh"][0]


def main():
    print(f"{RULE}\nexp73 Block D, second use — hongshao v1's STOCHASTIC LAYER on the "
          f"size gate\n{RULE}\n")
    recs, data, mask, lmh, base, theta, slow = S0.build(False)
    n = len(recs)
    cz = np.load(E60 / "outputs/stage3a_cube.npz")
    a1 = np.load(E60 / "outputs/stage1_core_anatomy.npz")
    a2 = {"delta_c": np.load(E60 / "outputs/stage2_size_anatomy.npz")["delta_c"]}
    t0 = dict(np.load(E60 / "outputs/stage0_targets.npz"))
    layer = np.load(E60 / "outputs/hongshao_v1_layer.npz", allow_pickle=True)
    good = np.isfinite(data).all(axis=(1, 2)) & (data > 0).all(axis=(1, 2))
    gal = np.where(good)[0]
    print(f"  {int(good.sum())} of {n} galaxies with a finite CoG at every epoch "
          f"(the population the layer was calibrated on; not a fit, so THE FITTING "
          f"SAMPLE rule does not apply)")

    smp = Sampler(cz, a1, a2, t0, np.arange(n), a1["fform"], tilt=True,
                  shift=float(layer["shift"]))
    smp.sig_add = np.asarray(layer["sig_add"], float)
    print(f"  layer: {str(layer['variant'])}; pool shift {float(layer['shift']):+.4f}; "
          f"sig_add " + " ".join(f"{v:.4f}" for v in smp.sig_add) + " (frozen, no recalibration)")

    mean = slow.predict(theta)
    lmh_g = lmh[good]
    # sanity: the cube's (dc=0, A=0) node IS the incumbent mean
    i0, j0 = np.where(cz["dc_grid"] == 0)[0][0], np.where(cz["a_grid"] == 0)[0][0]
    dev = np.nanmax(np.abs(cz["cube"][i0, j0, gal] - np.log10(np.clip(mean[gal], 1.0, None))))
    print(f"  GATE  the cube's (0, 0) node reproduces the incumbent mean to {dev:.1e} dex  "
          f"{'OK' if dev < 1e-3 else 'FAILED'}")
    assert dev < 1e-3

    print(f"\n  scoring the MEAN ...")
    g_mean = gate_tables(mean[good], data[good], lmh_g)
    draws_g = []
    for r in range(N_DRAW):
        rng = np.random.default_rng(SEED + 10 + r)
        d = 10.0 ** smp.profiles(gal, rng).astype(np.float64)
        ok = np.isfinite(d).all(axis=(1, 2)) & (d > 0).all(axis=(1, 2))
        print(f"  scoring draw {r + 1}/{N_DRAW} ({int(ok.sum())} finite) ...")
        draws_g.append(gate_tables(d[ok], data[good][ok], lmh_g[ok]))

    summary = {}
    for which, label in ((0, "STELLAR mass"), (1, "HALO mass")):
        print(f"\n{RULE}\n  tier 2d GATE at fixed {label} — the incumbent MEAN, then the LAYER's "
              f"draws (mean over {N_DRAW} draws, +- scatter across draws)\n"
              f"  offset = median dlog R (pass |.| <= {qa.SIZE_GATE_OFFSET}); width = model "
              f"scatter at fixed mass / truth's (pass within {int(100 * qa.SIZE_GATE_WIDTH)}%)\n{RULE}")
        print(f"  {'size':<5}{'':<7}" + "".join(f"{f'z={z}':>19}" for z in ANCHOR_Z)
              + f"{'OFFSET':>9}{'WIDTH':>7}")
        for key in KEYS:
            for src in ("mean", "layer"):
                offs, wids, po, pw = [], [], 0, 0
                cells = []
                for j in range(5):
                    if src == "mean":
                        v = g_mean[which][(key, j)]
                        o, w, so, sw = v["offset"], v["width_ratio"], 0.0, 0.0
                    else:
                        oo = np.array([g[which][(key, j)]["offset"] for g in draws_g])
                        ww = np.array([g[which][(key, j)]["width_ratio"] for g in draws_g])
                        o, w, so, sw = oo.mean(), ww.mean(), oo.std(), ww.std()
                    a = abs(o) <= qa.SIZE_GATE_OFFSET
                    b = abs(w - 1) <= qa.SIZE_GATE_WIDTH
                    po += a; pw += b
                    offs.append(o); wids.append(w)
                    tag = "ok" if (a and b) else ("OFF" if not a else "WID")
                    cells.append(f"{o:+.3f} {w:.2f} {tag:<3}".rjust(19) if src == "mean"
                                 else f"{o:+.3f} {w:.2f}±{sw:.2f} {tag:<3}".rjust(19))
                    summary[(label, key, j, src)] = (o, w, so, sw)
                print(f"  {key:<5}{src:<7}" + "".join(cells) + f"{po:>7}/5{pw:>5}/5")
        print(f"    (R20 on the 2 kpc grid is extrapolated for most high-z galaxies — a weaker row)")

    # the headline: how much of the truth's size diversity the layer supplies
    print(f"\n{RULE}\n  READ — width ratio (model scatter at fixed STELLAR mass / truth's), "
          f"MEAN -> LAYER\n{RULE}")
    print(f"  {'size':<6}" + "".join(f"{f'z={z}':>16}" for z in ANCHOR_Z))
    for key in KEYS:
        print(f"  {key:<6}" + "".join(
            f"{summary[('STELLAR mass', key, j, 'mean')][1]:>7.2f} -> {summary[('STELLAR mass', key, j, 'layer')][1]:.2f}"
            for j in range(5)))
    np.savez(OUTDIR / "size_gate_layer.npz",
             keys=np.array([f"{a}|{b}|{c}|{d}" for (a, b, c, d) in summary]),
             vals=np.array(list(summary.values())), n_draw=N_DRAW)
    print(f"\nwrote {OUTDIR / 'size_gate_layer.npz'}")


if __name__ == "__main__":
    main()
