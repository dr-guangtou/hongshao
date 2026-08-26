"""exp59 — the efficiency law conditioned on the halo's assembly state.

THE ONE-LINE PHYSICS. exp54's efficiency law converts each halo-mass deposit
into stars at a rate that depends only on the halo's MASS and the TIME —
two haloes of the same mass at the same epoch always get identical stellar
output. exp58 measured that this is the one assumption the z=2 residuals
reject: at fixed halo mass, early-assembled haloes (the ones whose centres
later decline) hold about 0.1 dex MORE stellar mass at z=2 than the shared law
grants, and the correlation survives the completeness mask (r = -0.45 with the
decline flag, -0.38 with concentration excess, +0.29 with formation time; see
`exp58_reassessment/README.md` P-B). This module lets the law see it:

    F1 : log_eps += e_f * (f_form_j - 0.5)     f_form_j = t_half(t_j)/t_j,
                                               per DEPOSIT, epoch-local
    C1 : log_eps += a_c * dlogc_j              concentration excess per
                                               deposit (0 where unmeasured,
                                               so absent = no effect)

Both quantities already live on `halo.Halo` per deposit; NO stellar
information enters. `F1+C1` exists but is fitted only if both are identified
alone. Pre-registered signs (from exp58 P-B, before any fit): `e_f < 0`
(early-assembled = small f_form = boosted) and `a_c > 0`.

WHY THIS IS A DOCUMENTED COPY AND HOW THE COPY IS GUARDED. `predict` below is
`stage35_time_law.StackedProblem.predict` with ONE inserted line (the two
conditioning terms added to `log_eps` before the clip). exp57's P6 recorded
that an assertion does not make duplication safe — so the guard here is the
strongest available: `check_nesting` demands that with every conditioning
coefficient at 0.0 this class reproduces the parent's `predict` **to exact
array equality, NaN pattern included** (adding a 0.0 term is exact in floating
point, so unlike exp57's per-epoch reassociation there is no ulp excuse), and
the module self-check refuses to pass without it. If exp54's parent ever
changes, the equality check fails loudly rather than drifting.

Self-check: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run \\
    python experiments/exp59_conditioned_efficiency/conditioned.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP54 = ROOT / "experiments/exp54_unpinned_amplitude"
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink", EXP54):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import model as M                                        # noqa: E402
import stage35_time_law as S35                           # noqa: E402

#: law -> (parameter names). "" is the exp54 incumbent, unconditioned.
LAWS = {"": (), "F1": ("e_f",), "C1": ("a_c",), "F1+C1": ("e_f", "a_c")}
#: two-sided on purpose: "the model does not want conditioning" must be an
#: interior point, not a rail (exp57's design rule). The spans are set from
#: the feature ranges — f_form-0.5 spans about [-0.2, +0.4] and dlogc about
#: [-0.4, +0.4] dex — so ±5 allows ±1-2 dex of efficiency modulation, far
#: beyond anything physical, leaving room to fail honestly.
BOUNDS = {"e_f": (-5.0, 5.0), "a_c": (-5.0, 5.0)}
#: the conditioning pivot. 0.5 matches the S3 convention in `model.r50_of`,
#: so a coefficient here is comparable with the S3 slopes exp54 rejected.
F_PIV = 0.5


@dataclass(frozen=True)
class CSpec:
    """An exp54 `model.Spec` plus a conditioning law; same interface the
    fitting harness uses (`theta_names`, `bounds()`, `n_theta`, `unpack`)."""

    base: M.Spec
    law: str = ""

    def __post_init__(self):
        if self.law not in LAWS:
            raise ValueError(f"law must be one of {sorted(LAWS)}, got "
                             f"{self.law!r}")

    @property
    def label(self):
        return self.base.label + (f"+{self.law}" if self.law else "")

    @property
    def x_names(self):
        return list(LAWS[self.law])

    @property
    def theta_names(self):
        return list(self.base.theta_names) + self.x_names

    @property
    def n_theta(self):
        return self.base.n_theta + len(LAWS[self.law])

    def __getattr__(self, name):
        """Everything not defined here (`family`, `eff`, `e3`, `size`,
        `conditioning`, `trunc_C`, ...) is the base spec's business, so
        `model.log_eps` / `r50_of` / `cog_truncated` see the base unchanged."""
        return getattr(self.base, name)

    def bounds(self):
        return list(self.base.bounds()) + [BOUNDS[n] for n in self.x_names]

    def unpack(self, theta):
        """(eff, size, shape, cond) — the base's three plus the conditioning."""
        theta = np.asarray(theta, float)
        if theta.shape != (self.n_theta,):
            raise ValueError(f"{self.label} wants {self.n_theta} params, got "
                             f"{theta.shape}")
        nx = len(LAWS[self.law])
        base_t = theta[: self.n_theta - nx] if nx else theta
        eff, size, shape = self.base.unpack(base_t)
        return eff, size, shape, (theta[self.n_theta - nx:] if nx
                                  else np.array([]))

    def nest(self, theta_base):
        """The incumbent's theta lifted into this spec, conditioning OFF.

        Every coefficient is 0.0 exactly, so `log_eps` gains a term that is
        identically zero and the model reproduces the incumbent to exact
        array equality (`check_nesting` is the warrant)."""
        return np.concatenate([np.asarray(theta_base, float),
                               np.zeros(len(LAWS[self.law]))])


class ConditionedProblem(S35.StackedProblem):
    """`StackedProblem` whose per-deposit efficiency sees the assembly state.

    Constructed with a `CSpec`; the parent is built on the BASE spec so every
    inherited method (`loss`, `scores`, `per_epoch`, `residuals`) works
    unchanged — they all go through `predict`, which is overridden here.
    """

    def __init__(self, cspec, halos, data, mask, **kw):
        super().__init__(cspec.base, halos, data, mask, **kw)
        self.cspec = cspec

    def cond_le(self, le, cond, d):
        """The conditioning term, per deposit, added to `log10 eps`."""
        names = LAWS[self.cspec.law]
        if "e_f" in names:
            le = le + cond[names.index("e_f")] * (d.f_form - F_PIV)
        if "a_c" in names:
            le = le + cond[names.index("a_c")] * d.dlogc
        return le

    def predict(self, theta):
        sp = self.cspec
        eff, size, shape, cond = sp.unpack(theta)
        d = self.dep
        n, nd = d.z.shape
        nE, nR = len(self.epochs), len(self.r_all)

        ok = np.isfinite(d.r200c) & (d.r200c > 0)
        # THE ONE CHANGE from the parent: the conditioning term joins log_eps
        # inside the same clip, through `cond_le` so a diagnostic subclass can
        # swap the term WITHOUT copying `predict` (exp57 P6: an assertion does
        # not make duplication safe — so there is no duplication to guard).
        # Everything below is the parent's code verbatim; `check_nesting`
        # enforces that claim.
        le = self.cond_le(M.log_eps(sp, eff, d), cond, d)
        dm = 10.0 ** np.clip(le, -30, 10) * d.dmh
        r50 = M.r50_of(sp, size, d)
        r_tr = sp.trunc_C * d.r200c
        good = (ok & np.isfinite(dm) & (dm >= 0)
                & np.isfinite(r50) & (r50 > 0))

        r50s = np.where(good, r50, 1.0)
        r_trs = np.where(good, r_tr, 100.0)
        B = M.cog_truncated(sp.family, shape, r50s.ravel(), r_trs.ravel(),
                            self.r_all)                       # (nR, n*nd)
        Bg = np.ascontiguousarray(B.T.reshape(n, nd, nR))
        W = np.where(good, dm, 0.0)[:, None, :] * self.em     # (n, nE, nd)
        out = np.matmul(W, Bg)                                # (n, nE, nR)

        dead = (ok.sum(1) < 5) | (good.sum(1) < 5) | ~np.isfinite(out).all((1, 2))
        out[dead] = np.nan
        return out


def check_nesting(cspec, recs, data, mask, theta_base):
    """With the conditioning OFF, this class must equal the parent EXACTLY —
    same values, same NaN pattern, no tolerance. Returns both predictions."""
    parent = S35.StackedProblem(cspec.base, recs, data, mask,
                                epochs=(0, 1, 2, 3, 4))
    child = ConditionedProblem(cspec, recs, data, mask, epochs=(0, 1, 2, 3, 4))
    pa = parent.predict(np.asarray(theta_base, float))
    pb = child.predict(cspec.nest(theta_base))
    if not np.array_equal(np.isnan(pa), np.isnan(pb)):
        raise SystemExit(f"NESTING FAILED for {cspec.label}: NaN patterns "
                         f"differ — the copy has drifted from the parent.")
    if not np.array_equal(pa[~np.isnan(pa)], pb[~np.isnan(pb)]):
        worst = np.nanmax(np.abs(pa - pb) / np.clip(np.abs(pa), 1e-30, None))
        raise SystemExit(f"NESTING FAILED for {cspec.label}: values differ "
                         f"(worst relative {worst:.2e}). Adding an exact-zero "
                         f"term cannot do this; the copy has drifted.")
    return parent, child


def _selfcheck():
    import stage33 as S33
    import stage37_size_epoch as S37
    print("exp59 conditioned.py self-check")
    recs, sel, data, mask, lmh, cuts = S37.build(True)      # smoke sample
    base = S33.spec_from_label("gompertz_log-E2-S2")
    theta = S37.incumbent()

    # (1) EXACT nesting for every law, values and NaN pattern
    for law in ("", "F1", "C1", "F1+C1"):
        sp = CSpec(base, law)
        parent, child = check_nesting(sp, recs, data, mask, theta)
        l0 = parent.loss(theta)
        l1 = child.loss(sp.nest(theta))
        assert l0 == l1, (law, l0, l1)
        print(f"  ({law or 'off':>5}) nesting EXACT — values, NaNs, and loss "
              f"({l0:.6f})")

    # (2) the mechanism is what the docstring says: with e_f = -1, each
    #     galaxy's z=2 log-mass change must track MINUS its deposit-weighted
    #     mean of (f_form - 0.5). This is a mechanics check, deliberately not
    #     a physics check — the first draft asserted a group contrast here and
    #     the full-sample leverage sweep (stage0_leverage.py) falsified the
    #     guess; which groups gain is an empirical question that stage
    #     answers, not something a self-check should hard-code.
    sp = CSpec(base, "F1")
    child = ConditionedProblem(sp, recs, data, mask, epochs=(0, 1, 2, 3, 4))
    p0 = child.predict(sp.nest(theta))
    th = sp.nest(theta).copy()
    th[-1] = -1.0
    p1 = child.predict(th)
    wmean = np.array([
        (np.sum(np.where(h.epoch_mask[4], h.dmh, 0.0) * (h.f_form - F_PIV))
         / max(np.sum(np.where(h.epoch_mask[4], h.dmh, 0.0)), 1e-30))
        for h in recs])
    with np.errstate(invalid="ignore", divide="ignore"):
        dlog = np.log10(p1[:, 4, -1] / p0[:, 4, -1])
    fin = np.isfinite(dlog) & np.isfinite(wmean)
    r = np.corrcoef(-wmean[fin], dlog[fin])[0, 1]
    print(f"  (mech) e_f = -1: correlation of each galaxy's z=2 mass change "
          f"with -1 x its\n         deposit-weighted (f_form - 0.5): "
          f"r = {r:+.3f}")
    assert r > 0.9, "the modulation must track the weighted conditioning mean"

    # (3) a_c moves with concentration excess likewise
    sp = CSpec(base, "C1")
    child = ConditionedProblem(sp, recs, data, mask, epochs=(0, 1, 2, 3, 4))
    p0 = child.predict(sp.nest(theta))
    th = sp.nest(theta).copy()
    th[-1] = +1.0
    p1 = child.predict(th)
    dc = np.array([np.nanmean(h.dlogc[h.has_c]) if h.has_c.any() else 0.0
                   for h in recs])
    with np.errstate(invalid="ignore", divide="ignore"):
        dlog = np.log10(p1[:, 4, -1] / p0[:, 4, -1])
    fin = np.isfinite(dlog)
    r = np.corrcoef(dc[fin], dlog[fin])[0, 1]
    print(f"  (dir ) a_c = +1: correlation of the mass change with the "
          f"galaxy's mean concentration excess r = {r:+.3f}")
    assert r > 0.5, "a_c > 0 must track concentration excess"
    print("  ALL CHECKS PASSED")


if __name__ == "__main__":
    _selfcheck()
