"""exp60 Stage 3c — compose the layer and judge it, held out, both ways.

THE SAMPLER. For each galaxy in the SCORING half (pools always from the
OTHER half; both half-assignments run and averaged — exp41's protocol):
  * core strength `A`: resampled from the Stage-1 empirical pool of the
    galaxy's formation-time quartile (the TILT; the untilted control pools
    all quartiles), then shifted by the variant's location;
  * size deviation `dc`: resampled from the Stage-2 empirical pool;
  * the profile: bilinear interpolation of the Stage-3a cube at (dc, A);
  * amplitude: a cross-epoch-correlated Gaussian whose per-epoch width is
    calibrated (on the calibration half) so the TOTAL drawn amplitude
    deviation matches the frozen Stage-0 target — the width the (dc, A)
    draws already induce is subtracted in quadrature, so nothing is
    double-counted.

THE TWO VARIANTS, because stage3b_tension measured that S1 and S3 cannot
both hold around this mean model:
  * `S3-strict`: the A pool is recentred to zero median, so the drawn
    population's median profile stays the incumbent's (S3 holds everywhere);
    S1's declining fraction is then structurally short.
  * `S1-first`: the A pool is used as measured (median +0.59); S3 is scored
    in two pieces (inside/outside 10 kpc) so the price — a mean central
    shift the user declined as a standalone mean-model change — is printed,
    not hidden.

Verdicts against the frozen Stage-0 targets and plan gates; the incumbent
("no-layer") row is the null; tilted-vs-untilted is the conditioning
control; tier-2b plane energies (gate S2) come from `qa.evaluate`, whose
energy ratios are already normalised by the truth's split-half floor, with
exp41's old-kernel layer numbers quoted alongside as the user-set context.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp60_stochastic_layer/stage3c_judge.py [--fast]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term"):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import stage0_cost as S0                                 # noqa: E402
from hongshao import qa                                  # noqa: E402

I5, I52, I148 = 3, 15, 23
SEED, N_REAL = 17, 8
ANCHOR_Z = (0.4, 0.7, 1.0, 1.5, 2.0)
RULE = "=" * 96


def bilinear(cube, dc_grid, a_grid, gal, dc, a):
    i = np.clip(np.searchsorted(dc_grid, dc) - 1, 0, len(dc_grid) - 2)
    j = np.clip(np.searchsorted(a_grid, a) - 1, 0, len(a_grid) - 2)
    wi = (dc - dc_grid[i]) / (dc_grid[i + 1] - dc_grid[i])
    wj = (a - a_grid[j]) / (a_grid[j + 1] - a_grid[j])
    wi, wj = wi[:, None, None], wj[:, None, None]
    return ((1 - wi) * (1 - wj) * cube[i, j, gal]
            + wi * (1 - wj) * cube[i + 1, j, gal]
            + (1 - wi) * wj * cube[i, j + 1, gal]
            + wi * wj * cube[i + 1, j + 1, gal])


class Sampler:
    def __init__(self, cube_z, a1, a2, t0, calib_idx, fform, tilt, shift):
        self.cube = cube_z["cube"]
        self.dcg, self.ag = cube_z["dc_grid"], cube_z["a_grid"]
        self.t0, self.tilt, self.shift = t0, tilt, shift
        edges = [-np.inf, *t0["q_edges"], np.inf]
        self.qof = lambda f: np.clip(np.searchsorted(t0["q_edges"], f), 0, 3)
        a_i, dc_i = a1["a_i"], a2["delta_c"]
        self.a_pools = [a_i[calib_idx][np.isfinite(a_i[calib_idx])
                                       & (self.qof(fform[calib_idx]) == q)]
                        for q in range(4)]
        self.a_pool_all = a_i[calib_idx][np.isfinite(a_i[calib_idx])]
        self.dc_pool = dc_i[calib_idx][np.isfinite(dc_i[calib_idx])]
        self.fform = fform
        self.sig_add = None                      # set by calibrate()

    #: pool-scale factor: A' = pool_median + scale * (A - pool_median) +
    #: shift. Scale < 1 narrows the drawn distribution around its centre —
    #: the lever that controls HOW DEEP the drawn declines run, calibrated
    #: (with the shift) against the frozen fraction and decliner-median
    #: targets on the calibration half.
    scale = 1.0

    def draw_da(self, gal, rng):
        if self.tilt:
            qs = self.qof(self.fform[gal])
            A = np.array([rng.choice(self.a_pools[q]) for q in qs])
        else:
            A = rng.choice(self.a_pool_all, size=len(gal))
        med = np.median(self.a_pool_all)
        A = med + self.scale * (A - med)
        A = np.clip(A + self.shift, self.ag[0], self.ag[-1])
        dc = rng.choice(self.dc_pool, size=len(gal))
        return dc, A

    def profiles(self, gal, rng):
        dc, A = self.draw_da(gal, rng)
        prof = bilinear(self.cube, self.dcg, self.ag, gal, dc, A)
        if self.sig_add is not None:
            L = np.linalg.cholesky(self.t0["amp_corr"]
                                   + 1e-9 * np.eye(5))
            eps = (rng.standard_normal((len(gal), 5)) @ L.T) * self.sig_add
            prof = prof + eps[:, :, None]
        return prof

    def calibrate(self, gal, rng):
        """sig_add per epoch so TOTAL amplitude deviation hits the target."""
        base = self.cube[np.where(self.dcg == 0)[0][0],
                         np.where(self.ag == 0)[0][0], gal]
        devs = []
        for _ in range(N_REAL):
            devs.append(self.profiles(gal, rng)[:, :, F.I100]
                        - base[:, :, F.I100])
        d = np.concatenate(devs)
        ind = np.nanstd(d, axis=0)
        tgt = self.t0["amp_sigma"]
        self.sig_add = np.sqrt(np.clip(tgt ** 2 - ind ** 2, 0.0, None))
        return ind

    def calibrate_shift(self, gal, rng, target_frac):
        """S1-first only: the pool shift that makes the CALIBRATION half's
        drawn declining fraction hit the target — a calibration against a
        frozen target, done held-out like every other calibration here."""
        lo_s, hi_s = -0.5, 2.0
        for _ in range(18):
            self.shift = 0.5 * (lo_s + hi_s)
            fr = []
            for _ in range(3):
                p = self.profiles(gal, rng)
                dch = p[:, 0, I5] - p[:, 4, I5]
                fr.append(np.mean(dch[np.isfinite(dch)] < 0))
            if np.mean(fr) < target_frac:
                lo_s = self.shift
            else:
                hi_s = self.shift
        self.shift = 0.5 * (lo_s + hi_s)
        return self.shift

    def calibrate_scale(self, gal, rng, target_frac, target_dmed):
        """(scale, shift) hitting BOTH the declining fraction and the
        decliner median, on the calibration half."""
        best = None
        for s in (1.0, 0.8, 0.6, 0.45, 0.3, 0.2):
            self.scale = s
            self.calibrate_shift(gal, rng, target_frac)
            dm = []
            for _ in range(3):
                p = self.profiles(gal, rng)
                dch = p[:, 0, I5] - p[:, 4, I5]
                dch = dch[np.isfinite(dch)]
                dm.append(np.median(dch[dch < 0]) if (dch < 0).any()
                          else np.nan)
            err = abs(np.nanmean(dm) - target_dmed)
            if best is None or err < best[0]:
                best = (err, s, self.shift)
        _, self.scale, self.shift = best
        return self.scale, self.shift


def judge(prof, base, data_h, t0, good):
    """All non-plane gate quantities for one drawn realization."""
    p, b = prof[good], base[good]
    dch = p[:, 0, I5] - p[:, 4, I5]
    fin = np.isfinite(dch)
    s1 = dict(frac=float(np.mean(dch[fin] < 0)),
              med=float(np.median(dch[fin])),
              med_dec=float(np.median(dch[fin & (dch < 0)]))
              if (fin & (dch < 0)).any() else np.nan,
              width=float(np.diff(np.percentile(dch[fin], [10, 90]))[0]))
    dmed = np.nanmedian(p - b, axis=0)                     # (5, 24)
    R = F.R_GRID
    s3_all = float(np.nanmax(np.abs(dmed[:3])))
    s3_out = float(np.nanmax(np.abs(dmed[:3][:, R > 10.5])))
    dev = p[:, :, F.I100] - b[:, :, F.I100]
    sig = np.nanstd(dev, axis=0)
    fin2 = np.isfinite(dev).all(axis=1)
    corr = np.corrcoef(dev[fin2].T)
    near = float(np.mean(np.diag(corr, 1)))
    ann_m = 10.0 ** p[:, 0, I148] - 10.0 ** p[:, 0, I52]
    ann_t = data_h[good][:, 0, I148] - data_h[good][:, 0, I52]
    okann = np.isfinite(ann_m) & (ann_m > 0) & (ann_t > 0)
    s4 = float(np.median(np.log10(ann_m[okann] / ann_t[okann])))
    return s1, s3_all, s3_out, sig, near, s4


def main(fast=False):
    print(f"{RULE}\nexp60 STAGE 3c — the layer, composed and judged both "
          f"ways\n{RULE}\n")
    recs, data, mask, lmh, base_sp, theta, slow = S0.build(False)
    n = len(recs)
    cz = np.load(HERE / "outputs" / "stage3a_cube.npz")
    a1 = np.load(HERE / "outputs" / "stage1_core_anatomy.npz")
    a2 = {"delta_c": np.load(HERE / "outputs" / "stage2_size_anatomy.npz"
                             )["delta_c"]}
    t0 = dict(np.load(HERE / "outputs" / "stage0_targets.npz"))
    fform = a1["fform"]
    cube0 = cz["cube"][np.where(cz["dc_grid"] == 0)[0][0],
                       np.where(cz["a_grid"] == 0)[0][0]]

    shift_strict = -float(np.median(a1["a_i"][np.isfinite(a1["a_i"])]))
    rows = (("no-layer", None, None),
            ("S3-strict tilted", True, shift_strict),
            ("S3-strict untilted", False, shift_strict),
            ("S1-first tilted", True, 0.0),
            ("S1-first untilted", False, 0.0),
            ("S1-scaled tilted", True, "scale"))
    print(f"  targets (frozen): declining {100 * float(t0['frac_decline']):.1f}%, "
          f"median {float(t0['med_all']):+.4f}, decliner median "
          f"{float(t0['med_dec']):+.4f}, 10-90 width "
          f"{float(t0['p90'] - t0['p10']):.4f}")
    print(f"  S3-strict pool shift {shift_strict:+.3f}; N_REAL {N_REAL}; "
          f"both half-assignments averaged\n")

    rng0 = np.random.default_rng(SEED)
    perm = rng0.permutation(n)
    halves = (perm[: n // 2], perm[n // 2:])
    good_all = np.isfinite(data).all(axis=(1, 2)) & (data > 0).all(axis=(1, 2))

    qof = lambda f: np.clip(np.searchsorted(t0["q_edges"], f), 0, 3)

    hdr = (f"  {'row':<20}{'declin.':>8}{'med chg':>9}{'dec med':>9}"
           f"{'width':>7}{'S3 all':>8}{'S3>10k':>8}{'sig rat':>8}"
           f"{'corr1':>7}{'S4 ann':>8}")
    print(hdr)
    plane_rows, qfracs = {}, {}
    for name, tilt, shift in rows:
        acc, qacc = [], []
        for hswap in (0, 1):
            cal, sco = halves[hswap], halves[1 - hswap]
            gsco = sco[good_all[sco]]
            if name == "no-layer":
                prof = cube0
                r = judge(prof, cube0, data, t0, gsco)
                acc.append(r)
                if hswap == 0:
                    plane_rows[name] = 10.0 ** cube0[gsco]
                    plane_rows[name + "#gal"] = gsco
                continue
            smp = Sampler(cz, a1, a2, t0, cal, fform, tilt,
                          0.0 if shift == "scale" else shift)
            gcal = cal[good_all[cal]]
            smp.sig_add = None
            smp.calibrate(gcal, np.random.default_rng(SEED + 100 + hswap))
            if shift == "scale":
                sc, sh = smp.calibrate_scale(
                    gcal, np.random.default_rng(SEED + 7),
                    float(t0["frac_decline"]), float(t0["med_dec"]))
                smp.sig_add = None
                smp.calibrate(gcal,
                              np.random.default_rng(SEED + 100 + hswap))
                if hswap == 0:
                    print(f"    (S1-scaled calibrated scale {sc:.2f}, "
                          f"shift {sh:+.3f})")
            elif "S1-first" in name:
                # shift is calibrated WITH the amplitude noise active (it
                # widens the change distribution and moves the fraction),
                # then the noise width is re-calibrated at the final shift
                s = smp.calibrate_shift(gcal, np.random.default_rng(SEED + 7),
                                        float(t0["frac_decline"]))
                smp.sig_add = None
                smp.calibrate(gcal,
                              np.random.default_rng(SEED + 100 + hswap))
                if hswap == 0 and tilt:
                    print(f"    (S1-first calibrated shift {s:+.3f})")
            for rr in range(2 if fast else N_REAL):
                rng = np.random.default_rng(SEED + 1000 * hswap + rr)
                prof_s = smp.profiles(gsco, rng)
                full = np.full((n, 5, 24), np.nan, np.float32)
                full[gsco] = prof_s
                r = judge(full, cube0, data, t0, gsco)
                acc.append(r)
                dch = prof_s[:, 0, I5] - prof_s[:, 4, I5]
                qs = qof(fform[gsco])
                qacc.append([np.mean((dch < 0)[np.isfinite(dch) & (qs == q)])
                             for q in range(4)])
                if hswap == 0 and rr == 0 and "untilted" not in name:
                    plane_rows[name] = 10.0 ** prof_s
                    plane_rows[name + "#gal"] = gsco
        s1s = [a[0] for a in acc]
        m = lambda k: np.mean([s[k] for s in s1s])
        sig_r = np.mean([np.mean(a[3] / t0["amp_sigma"]) for a in acc])
        near = np.mean([a[4] for a in acc])
        near_t = float(np.mean(np.diag(t0["amp_corr"], 1)))
        print(f"  {name:<20}{100 * m('frac'):>7.1f}%{m('med'):>+9.4f}"
              f"{m('med_dec'):>+9.4f}{m('width'):>7.3f}"
              f"{np.mean([a[1] for a in acc]):>8.4f}"
              f"{np.mean([a[2] for a in acc]):>8.4f}"
              f"{sig_r:>8.2f}{near - near_t:>+7.2f}"
              f"{np.mean([a[5] for a in acc]):>+8.4f}")
        if qacc:
            qfracs[name] = np.mean(qacc, axis=0)

    print(f"\n  S1-conditional — declining fraction per f_form quartile "
          f"(earliest first;\n  the tilt exists to reproduce this gradient):")
    print(f"  {'row':<22}" + "".join(f"{f'Q{q + 1}':>9}" for q in range(4)))
    print(f"  {'MEASURED target':<22}"
          + "".join(f"{100 * v:>8.1f}%" for v in t0["q_frac"]))
    for name, v in qfracs.items():
        print(f"  {name:<22}" + "".join(f"{100 * x:>8.1f}%" for x in v))

    print(f"\n  read: declining -> {100 * float(t0['frac_decline']):.0f}%; "
          f"S3 columns are the drawn-median shift vs the incumbent "
          f"(strict gate 0.010,\n  S1-first judged on the >10 kpc column); "
          f"sig rat -> 1.00 (S5, within 10%); corr1 -> 0.00 (within 0.05); "
          f"S4 vs the\n  incumbent's own annulus value on the no-layer row, "
          f"band 0.02.")

    # ---- S2: tier-2b plane energies vs the split-half floor -------------- #
    print(f"\n  S2 — kpc-plane energy over the truth's split-half floor "
          f"(target <= 1.3 at z <= 1.0; exp41's\n  old-kernel layer reached "
          f"1.0-1.4 at z <= 0.7):")
    for name in ("no-layer", "S3-strict tilted", "S1-first tilted",
                 "S1-scaled tilted"):
        gsco = plane_rows[name + "#gal"]
        ev = qa.evaluate(plane_rows[name], data[gsco], F.R_GRID,
                         list(ANCHOR_Z), name=f"exp60_{name}", figdir=None,
                         figures=False, verbose=False,
                         bin_by=lmh[gsco][:, 0], bin_label="logMh")
        for pl in ev["planes"]:
            lab = f"{pl[0]} vs {pl[1]}"
            if "Re" in lab:
                continue
            row = [ev["planes"][pl][j][1]["energy_ratio"] for j in range(3)]
            print(f"    {name:<20}{lab:<28}"
                  + "".join(f"{v:>8.2f}" for v in row))


if __name__ == "__main__":
    main(fast="--fast" in sys.argv[1:])
