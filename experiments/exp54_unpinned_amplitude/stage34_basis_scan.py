"""exp54 Stage 3.4 — WOULD A DIFFERENT BASIS RAISE THE CEILING?

`stage34_ceiling.py` bounds the model with the basis it already has. This asks
the next question, and it is the one that decides whether the two proposed
extensions are worth building:

  * **the size law.** `R50 = f0 R200c(t_j) (1+z_j)^b` was fitted jointly with
    the efficiency law, so its value is a compromise between "where should this
    deposit go" and "how much mass should it carry". At the ceiling the
    efficiency is perfect by construction, so re-scanning `(log_f0, b)` there
    asks what the size law would be if it had only one job. If the ceiling-
    optimal `b` is far from the fitted one, the size law is mis-set.

  * **a redshift-dependent deposit shape**, `c = c0 + c_z ln(1+z)`, so that
    high-redshift deposits are intrinsically more concentrated. This is the one
    nested parameter agreed as the next thing to try against "the model's z=2
    galaxies are too diffuse". Scanning it HERE, at the ceiling, tests it
    before it is built: if a redshift-dependent shape cannot lower the bound,
    it cannot lower the fitted loss either, and the extension is dead without a
    single fit. That is the move Stage 3.0 made against the survival term.

WHY THIS IS NOT JUST "MORE FREEDOM". Every point on the grid gets exactly the
same per-galaxy freedom -- the same 71 non-negative weights. Only the GLOBAL
basis parameters move. So a drop in the ceiling is a real statement about the
basis and not about degrees of freedom, which is what would happen if the
deposits were instead handed a free choice of shape one by one.

THE CONTROL. Each grid point is also scored on the PERMUTED target -- this
galaxy's basis fitted to another galaxy's measured profile. A basis that
lowers the true ceiling and the permuted ceiling equally has bought generic
flexibility, not physics. The reported quantity is therefore always the pair.

HOW IT IS COMPUTED. Every galaxy shares the same 72-snapshot deposit grid (the
run asserts it), so `R50`, `R_trunc` and the shape parameter can be evaluated
one SNAPSHOT at a time across all galaxies at once -- 72 vectorised profile
builds per grid point instead of 2397 x 72 scalar ones.

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python -u \\
     experiments/exp54_unpinned_amplitude/stage34_basis_scan.py [--smoke]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink", HERE):
    sys.path.insert(0, str(p))

# `fit` is imported FIRST and before `halo`/`model` on purpose: those two
# insert other experiments' directories at the head of `sys.path`, and more
# than one experiment ships a module called `fit`. Importing it here binds the
# exp54 one. (Unused directly; see `stage34_ceiling.F`.)
import fit as _fit_import_order_guard                    # noqa: E402,F401
import halo as H                                         # noqa: E402
import model as M                                        # noqa: E402
import stage33 as S33                                    # noqa: E402
import stage34_ceiling as C                              # noqa: E402
import families                                          # noqa: E402

R = C.R
Z = C.Z
RULE = C.RULE
OUT = HERE / "outputs" / "stage34_basis_scan.npz"
#: the adopted model's basis, from the per-epoch mh-complete fit
LABEL = "gompertz_log-E2-S2"
#: galaxies used for the grid; the winner is re-measured on all of them
SCAN_STRIDE = 3


class Stack:
    """The sample as flat arrays on the shared snapshot grid.

    Holds only halo quantities and the measured curves of growth, so a basis
    can be rebuilt for any candidate parameters without touching `Halo`
    objects again.
    """

    def __init__(self, recs, data, mask):
        z0 = recs[0].z
        for h in recs:
            if len(h.z) != len(z0) or not np.allclose(h.z, z0):
                raise SystemExit(
                    "REFUSING TO RUN: the galaxies do not share one snapshot "
                    "grid, so the snapshot-vectorised basis build is invalid.")
        self.z = np.asarray(z0, float)                    # (nd,)
        self.r200c = np.array([h.r200c for h in recs])    # (n, nd)
        self.em = np.asarray(recs[0].epoch_mask, bool)    # (ne, nd) shared
        self.good = np.isfinite(self.r200c) & (self.r200c > 0)
        self.data = data
        self.mask = mask
        self.n, self.nd = self.r200c.shape
        self.ne = self.em.shape[0]

    def sub(self, idx):
        s = object.__new__(Stack)
        s.z, s.em, s.nd, s.ne = self.z, self.em, self.nd, self.ne
        s.r200c, s.good = self.r200c[idx], self.good[idx]
        s.data, s.mask = self.data[idx], self.mask[idx]
        s.n = len(idx)
        return s


def build_columns(stack, family, log_f0, b, shape_of_z):
    """(nd, nR, n) unit-mass truncated deposit profiles, one snapshot at a time.

    `shape_of_z` maps redshift to the family's shape tuple, so a redshift-
    dependent deposit shape costs one extra profile table per snapshot and
    nothing per galaxy.
    """
    cols = np.zeros((stack.nd, len(R), stack.n))
    r50_scale = 10.0 ** (log_f0 + b * np.log10(1.0 + stack.z))
    for j in range(stack.nd):
        g = stack.good[:, j]
        if not g.any():
            continue
        r200 = stack.r200c[g, j]
        cols[j][:, g] = M.cog_truncated(family, shape_of_z(stack.z[j]),
                                        r50_scale[j] * r200, 3.0 * r200, R)
    return cols


def ceiling_of(stack, cols, target=None, wmod=None):
    """Joint causal NNLS with this basis. `target` defaults to each galaxy's
    own measured curve of growth; pass a permutation of it for the control.

    With `wmod` (the fitted law's own deposit masses) the solve is restricted
    to FIVE weights per galaxy, one per epoch interval, with the within-interval
    distribution held at the model's -- the `interval5` rung. That is the
    freedom level a global efficiency law can plausibly reach, so a basis change
    that helps only the 71-weight solve and not this one has not earned a fit.
    """
    tgt = stack.data if target is None else target
    pred = np.empty((stack.n, stack.ne, len(R)))
    for i in range(stack.n):
        g = stack.good[i]
        B = cols[:, :, i][g].T                            # (nR, ngood)
        A = np.concatenate([B * stack.em[k][g][None, :]
                            for k in range(stack.ne)], axis=0)
        if wmod is not None:
            grp = C.interval_groups(stack.em[:, g])
            w = wmod[i][g]
            G = np.zeros((A.shape[0], stack.ne))
            for gg in range(stack.ne):
                s = grp == gg
                if s.any() and w[s].sum() > 0:
                    G[:, gg] = A[:, s] @ w[s] / w[s].sum()
            A = G
        pred[i] = (A @ C._solve(A, tgt[i].ravel())).reshape(stack.ne, -1)
    return pred


def mean_score_F(stack, pred, truth=None):
    _, sf = C._score_masked(pred, stack.data if truth is None else truth,
                            stack.mask)
    return float(np.mean(sf)), sf


def scan(stack, family, points, clip, permuted=True, wmod=None, verbose=True):
    """Evaluate the ceiling at every point of a grid. Returns a record array."""
    perm = (np.arange(stack.n) + max(stack.n // 3, 1)) % stack.n
    rows = []
    t0 = time.time()
    for p in points:
        log_f0, b, c0, c_z = p
        sh = _shape_of_z(c0, c_z, clip)
        cols = build_columns(stack, family, log_f0, b, sh)
        m, sf = mean_score_F(stack, ceiling_of(stack, cols, wmod=wmod))
        mp = np.nan
        if permuted:
            mp, _ = mean_score_F(stack, ceiling_of(stack, cols, wmod=wmod,
                                                   target=stack.data[perm]),
                                 truth=stack.data[perm])
        rows.append((log_f0, b, c0, c_z, m, mp, *sf))
        if verbose:
            print(f"      log_f0={log_f0:+.3f} b={b:+.3f} c0={c0:+.3f} "
                  f"c_z={c_z:+.3f}   ceiling {m:.4f}   permuted {mp:.4f}",
                  flush=True)
    print(f"    {len(points)} grid points in {(time.time() - t0) / 60:.1f} min")
    return np.array(rows)


def _shape_of_z(c0, c_z, clip):
    """`c = c0 + c_z ln(1+z)`, clipped to the family's own admissible range."""
    return lambda zz: (float(np.clip(c0 + c_z * np.log1p(zz), *clip)),)


def _report_cz(rec, where):
    """Does `c_z` earn its place at this freedom level? Net of the control."""
    best = rec[np.argmin(rec[:, 4])]
    at0 = rec[np.abs(rec[:, 3]) < 1e-12]
    b0 = at0[np.argmin(at0[:, 4])]
    gain, gain_p = b0[4] - best[4], b0[5] - best[5]
    print(f"\n    at {where}:")
    print(f"      best overall           c0={best[2]:+.4f} c_z={best[3]:+.4f}"
          f"   ceiling {best[4]:.4f}   permuted {best[5]:.4f}")
    print(f"      best with c_z = 0      c0={b0[2]:+.4f} c_z=+0.0000"
          f"   ceiling {b0[4]:.4f}   permuted {b0[5]:.4f}")
    print(f"      a redshift-dependent shape buys {gain:+.4f} in mean score_F; "
          f"the permuted control\n      buys {gain_p:+.4f} of that, so the part "
          f"attributable to the galaxies is {gain - gain_p:+.4f}.")
    return gain, gain_p


def _grid_edge_warning(rec, names, lo, hi):
    """Say so out loud when the optimum sits on the boundary of the scan."""
    best = rec[np.argmin(rec[:, 4])]
    on_edge = [n for n, j, a, b in zip(names, range(4), lo, hi)
               if abs(best[j] - a) < 1e-9 or abs(best[j] - b) < 1e-9]
    if on_edge:
        print(f"    NOTE: the optimum sits on the grid EDGE in "
              f"{', '.join(on_edge)}; the surface is degenerate there and the "
              f"reported gain is a lower bound on what a wider grid would find.")


def main(smoke=False):
    import selection as S
    import stage2_multiepoch as s2
    s2._w_init(None)
    pop = np.load(C.POP)
    all_rows = np.array([g["row"] for g in s2._W["gals"]])
    cuts = np.load(C.SEL, allow_pickle=True)["cuts"]
    m200 = S.sample_masses(np.load(S.HS_NPZ, allow_pickle=True))
    complete_all = np.isfinite(m200) & (m200 >= cuts[None, :])

    rows = all_rows[::40] if smoke else all_rows
    recs = H.build_records(rows=rows, verbose=not smoke)
    sel = np.array([h.row for h in recs])
    stack = Stack(recs, pop["data"][sel], complete_all[sel])

    spec = S33.spec_from_label(LABEL)
    theta = np.asarray(np.load(C.PEREPOCH, allow_pickle=True)
                       [f"{LABEL}_theta"], float)
    _, size, shape = spec.unpack(theta)
    log_f0_0, b_0 = float(size[0]), float(size[1])
    c_0 = float(shape[0])
    clip = families.shape_bounds(spec.family)[0]

    print(f"exp54 STAGE 3.4 — WOULD A DIFFERENT BASIS RAISE THE CEILING?")
    print(f"  basis   : {LABEL}, fitted log_f0={log_f0_0:+.4f} b={b_0:+.4f} "
          f"c={c_0:+.4f}   (shape bounds {clip})")
    print(f"  galaxies: {stack.n}   mh-complete per epoch: "
          + " ".join(str(int(v)) for v in stack.mask.sum(0)))

    # the gate: at the fitted parameters this module must reproduce
    # stage34_ceiling's ceiling exactly, or the two are not the same object
    cols0 = build_columns(stack, spec.family, log_f0_0, b_0,
                          _shape_of_z(c_0, 0.0, clip))
    ref, ref_sf = mean_score_F(stack, ceiling_of(stack, cols0))
    prev = np.load(C.OUT, allow_pickle=True) if C.OUT.exists() else None
    print(f"\n  CEILING GATE (this module vs stage34_ceiling at the fitted basis)")
    if prev is not None and not smoke:
        # this module solves on all five epochs with no mask, so the
        # matching product in stage34_ceiling is the `all5` sample definition
        want = np.asarray(prev[f"{LABEL}|all5|ceiling_sF"], float)
        dev = float(np.max(np.abs(ref_sf - want)))
        print(f"    per-epoch score_F  " + " ".join(f"{v:.5f}" for v in ref_sf))
        print(f"    stage34_ceiling    " + " ".join(f"{v:.5f}" for v in want))
        print(f"    max |difference| = {dev:.2e}")
        if dev > 1e-9:
            raise SystemExit(
                "REFUSING TO RUN: the snapshot-vectorised basis is not the "
                "same basis stage34_ceiling bounded, so a scan around it means "
                "nothing.")
        print(f"    OK — same basis, same ceiling\n")
    else:
        print(f"    skipped (no stage34_ceiling output, or smoke run)\n")

    idx = np.arange(stack.n)[::(1 if smoke else SCAN_STRIDE)]
    sub = stack.sub(idx)
    print(f"  scanning on {sub.n} galaxies; every point quoted as a CONCLUSION "
          f"is then re-measured\n  on all {stack.n} (`confirm` below). The "
          f"first draft of this file printed that promise\n  and did not keep "
          f"it.\n")

    n_g = 3 if smoke else 7
    print(f"{RULE}\n  SCAN A — the SIZE LAW at the ceiling: R50 = f0 R200c(t_j) "
          f"(1+z_j)^b\n{RULE}")
    lf_lo, lf_hi = log_f0_0 - 0.6, log_f0_0 + 0.6
    b_lo, b_hi = b_0 - 1.2, b_0 + 1.2
    pts = [(lf, b, c_0, 0.0)
           for lf in np.round(np.linspace(lf_lo, lf_hi, n_g), 4)
           for b in np.round(np.linspace(b_lo, b_hi, n_g), 4)]
    A = scan(sub, spec.family, pts, clip)
    best_a = A[np.argmin(A[:, 4])]
    at_fit = A[np.argmin(np.abs(A[:, 0] - log_f0_0) + np.abs(A[:, 1] - b_0)), 4]
    print(f"\n    ceiling-optimal size law: log_f0={best_a[0]:+.4f} "
          f"b={best_a[1]:+.4f}   (fitted {log_f0_0:+.4f} {b_0:+.4f})")
    print(f"    mean score_F {best_a[4]:.4f} against {at_fit:.4f} at the fitted "
          f"values — a gain of {at_fit - best_a[4]:+.4f} "
          f"({100 * (at_fit - best_a[4]) / at_fit:+.1f} per cent)")
    _grid_edge_warning(A, ("log_f0", "b", "c0", "c_z"),
                       (lf_lo, b_lo, -np.inf, -np.inf),
                       (lf_hi, b_hi, np.inf, np.inf))

    print(f"\n{RULE}\n  SCAN B — a REDSHIFT-DEPENDENT DEPOSIT SHAPE at the "
          f"ceiling: c = c0 + c_z ln(1+z)\n{RULE}")
    lf_b, b_b = float(best_a[0]), float(best_a[1])
    c_lo, c_hi = c_0 - 0.45, c_0 + 0.45
    cz_lo, cz_hi = -0.6, 0.6
    pts = [(lf_b, b_b, c0, cz)
           for c0 in np.round(np.linspace(c_lo, c_hi, n_g), 4)
           for cz in np.round(np.linspace(cz_lo, cz_hi, n_g), 4)]
    B = scan(sub, spec.family, pts, clip)
    _report_cz(B, "the 71-weight ceiling")
    _grid_edge_warning(B, ("log_f0", "b", "c0", "c_z"),
                       (-np.inf, -np.inf, c_lo, cz_lo),
                       (np.inf, np.inf, c_hi, cz_hi))

    print(f"\n{RULE}\n  SCAN C — the same, at the freedom a GLOBAL LAW can "
          f"actually reach\n{RULE}")
    print(f"     The 71-weight ceiling could in principle be insensitive to the "
          f"deposit shape and\n     the fitted model still not be, so the scan "
          f"is repeated at the `interval5` rung:\n     five weights per galaxy, "
          f"the within-interval distribution held at the fitted law's.\n")
    wmod = np.array([np.where(np.isfinite(h.r200c) & (h.r200c > 0),
                              10.0 ** np.clip(M.log_eps(spec, spec.unpack(theta)[0], h),
                                              -30, 10) * h.dmh, 0.0)
                     for h in recs])[idx]
    Cc = scan(sub, spec.family, pts, clip, wmod=wmod)
    _report_cz(Cc, "the 5-weight `interval5` rung")

    print(f"\n{RULE}\n  CONFIRMATION ON ALL {stack.n} GALAXIES\n{RULE}")
    print(f"     The scan grid runs on every {SCAN_STRIDE}rd galaxy. These are "
          f"the four points the\n     conclusions rest on, re-measured on the "
          f"whole sample.\n")
    b_cz0 = B[np.abs(B[:, 3]) < 1e-12][
        np.argmin(B[np.abs(B[:, 3]) < 1e-12][:, 4])]
    best_cz = B[np.abs(B[:, 3]) > 1e-12][
        np.argmin(B[np.abs(B[:, 3]) > 1e-12][:, 4])]
    confirm_pts = [
        ("the fitted basis", (log_f0_0, b_0, c_0, 0.0)),
        ("the ceiling-optimal size law", (lf_b, b_b, c_0, 0.0)),
        ("best with c_z = 0", tuple(b_cz0[:4])),
        ("best with c_z != 0", tuple(best_cz[:4])),
    ]
    conf = scan(stack, spec.family, [p for _, p in confirm_pts], clip,
                verbose=False)
    coh = stack.sub(np.where(stack.mask[:, 4])[0])
    conf_c = scan(coh, spec.family, [p for _, p in confirm_pts], clip,
                  permuted=False, verbose=False)
    print(f"     {'point':<30}{'log_f0':>8}{'b':>8}{'c0':>8}{'c_z':>8}"
          f"{'all5':>9}{'permuted':>10}{'cohort':>9}")
    for (nm, _), row, rc in zip(confirm_pts, conf, conf_c):
        print(f"     {nm:<30}{row[0]:>8.3f}{row[1]:>8.3f}{row[2]:>8.3f}"
              f"{row[3]:>8.3f}{row[4]:>9.4f}{row[5]:>10.4f}{rc[4]:>9.4f}")
    print(f"\n     On all {stack.n} galaxies a non-zero c_z changes the "
          f"ceiling by {conf[3][4] - conf[2][4]:+.4f} in mean score_F, and on "
          f"the\n     {coh.n}-galaxy z=2 cohort by "
          f"{conf_c[3][4] - conf_c[2][4]:+.4f}. Positive means WORSE.")

    if smoke:
        print("\n  smoke run: writing nothing")
        return
    np.savez_compressed(OUT, scan_size=A, scan_shape=B, scan_shape_i5=Cc,
                        confirm=conf, confirm_cohort=conf_c,
                        confirm_names=np.array([n for n, _ in confirm_pts]),
                        columns=np.array(["log_f0", "b", "c0", "c_z",
                                          "ceiling", "permuted",
                                          *[f"sF_z{z}" for z in Z]]),
                        fitted=np.array([log_f0_0, b_0, c_0, 0.0]),
                        rows=sel, stride=SCAN_STRIDE)
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
