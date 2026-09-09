"""exp78 — the two candidate SIZE TERMS for the objective, as pure array
functions, with the probes that establish what each can and cannot see.

WHY. exp63's objective (A^2 + F^2 + S^2 + B^2 per epoch) scores no size at all.
It has now ranked a basin against the gates for the fifth time (C23: the 14.63
basin, a 25 kpc "compact" channel every gate rejects) and rejected a lever the
gates want (exp76's growth-rate split). A fit can only respect the size gate
if the loss can see it. Two candidates, both declared before any fit:

  (a) THE RADIUS TERM.  For f in {0.2, 0.5, 0.8}, R_f is the radius enclosing
      the fraction f of the galaxy's OWN mass inside 148 kpc -- the model's
      from its own curve, the truth's from the truth's, on exp73's merged
      0.673-148 kpc grid (`coordinate.extended_cog`), so the truth's R20 is
      measured rather than extrapolated. The residual is log10(R_f,model /
      R_f,truth) in dex. Per epoch: the MEDIAN of that residual in each
      halo-mass tercile (the same terciles as the binned term B, by the
      measured mass), then the rms over the 3 fractions x 3 terciles. This is
      the size gate's OFFSET sub-gate, written as a loss term; it is a MEDIAN
      term on purpose, so the per-galaxy size scatter no mean model can
      reproduce (C16) does not dominate it the way the centre's intrinsic
      scatter dominates the per-galaxy terms (memory
      `loss-is-blind-to-the-binned-gate`).
  (b) exp77's TRANSFORMED ANNULAR TERM.  On the 50-100 and 100-148 kpc annuli,
      T = asinh(M_ann / (0.001 x total)) / ln 10, the per-galaxy mean over the
      two annuli of (T_model - T_truth)^2, then the rms over galaxies at the
      epoch. exp77 divided BOTH sides by the truth's total (`divisor="truth"`);
      the variant dividing each side by its OWN total (`divisor="own"`) is
      carried because the first is an amplitude term as much as a shape term
      (probe 2 below shows it).

Each term is reported raw (dex) and NORMALISED to the nested incumbent on the
measured curves (1.000 at the null), exactly as S and B are.

THE PROBES (`run_probes`), asserted on the real truth before a term may enter
a loss: (1) the term is zero when the model IS the truth; (2) a pure
amplitude rescaling of the model must NOT move it; (3) a pure size shift
(the same galaxy 20 per cent larger) MUST move it, by about log10(1.2) dex;
(4) mass added inside the first radius (a point mass of a tenth of the total)
MUST move it; (5) a per-galaxy size scatter with zero median log should
barely move the radius term (it is an offset term, not a width term).

The model never sees the truth except inside these functions: they take a
model array and a truth array and return a number.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp73_size_relative"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import coordinate as C                                   # noqa: E402

FRACTIONS = (0.2, 0.5, 0.8)
ANNULUS_EDGES_KPC = (50.0, 100.0)                        # the third edge is the grid's last radius
ASINH_SCALE = 0.001                                      # exp77: asinh(M_ann / (0.001 x total)) / ln 10
LN10 = np.log(10.0)


# --------------------------------------------------------------------------- #
# ingredients
# --------------------------------------------------------------------------- #
def fractional_radius(cog, R, frac):
    """(...,) radius enclosing `frac` of M(<R[-1]) by linear interpolation of the
    running maximum of the curve -- `coordinate.size_radius`, vectorised (the
    loop version costs a second per call on the sample, which a loss cannot
    afford). NaN where the curve is not finite or has no mass. Linear inside
    R[0], as the original. `selftest` asserts agreement with the original."""
    cog = np.asarray(cog, float)
    R = np.asarray(R, float)
    flat = cog.reshape(-1, cog.shape[-1])
    ok = np.isfinite(flat).all(-1) & (flat[:, -1] > 0)
    out = np.full(len(flat), np.nan)
    if ok.any():
        f = flat[ok]
        cm = np.maximum.accumulate(f, axis=-1)
        tgt = frac * f[:, -1]
        j = (cm < tgt[:, None]).sum(-1)                  # first index with cm >= tgt
        res = np.empty(len(f))
        inner = j == 0
        res[inner] = R[0] * tgt[inner] / np.maximum(cm[inner, 0], 1e-300)
        jj = np.clip(j[~inner], 1, len(R) - 1)
        lo = cm[~inner, jj - 1]
        hi = cm[~inner, jj]
        t = (tgt[~inner] - lo) / np.where(hi > lo, hi - lo, 1.0)
        res[~inner] = R[jj - 1] + t * (R[jj] - R[jj - 1])
        out[ok] = res
    return out.reshape(cog.shape[:-1])


def cumulative_linear(cog, R, r):
    """M(<r) by LINEAR interpolation in radius and mass (the standard QA's
    convention, and exp77's), for one scalar radius `r`; flat beyond R[-1]."""
    cog = np.asarray(cog, float)
    R = np.asarray(R, float)
    if r >= R[-1]:
        return cog[..., -1]
    j = int(np.searchsorted(R, r, side="right") - 1)
    j = max(0, min(j, len(R) - 2))
    t = (r - R[j]) / (R[j + 1] - R[j])
    return cog[..., j] * (1 - t) + cog[..., j + 1] * t


def annular_masses(cog, R, edges=ANNULUS_EDGES_KPC):
    """(..., 2) masses in [edges[0], edges[1]] and [edges[1], R[-1]]."""
    m0 = cumulative_linear(cog, R, edges[0])
    m1 = cumulative_linear(cog, R, edges[1])
    m2 = np.asarray(cog, float)[..., -1]
    return np.stack([m1 - m0, m2 - m1], axis=-1)


def tercile_masks(lmh):
    """The binned term's three halo-mass terciles (the same construction as
    exp63's `Problem2`)."""
    lm = np.asarray(lmh, float)
    e = np.quantile(lm, [0, 1 / 3, 2 / 3, 1])
    return [(lm >= e[b]) & (lm <= e[b + 1] + 1e-9) for b in range(3)]


# --------------------------------------------------------------------------- #
# the two terms, one epoch at a time
# --------------------------------------------------------------------------- #
def radius_term(model, truth, R, lmh, fractions=FRACTIONS):
    """(a) for ONE epoch: model, truth (n, nR) on the same grid; lmh (n,) the
    binning mass. Returns (rms over fractions x terciles of the tercile-median
    log10 R_f ratio [dex], the (n_frac, 3) median table, n_bad)."""
    model, truth = np.asarray(model, float), np.asarray(truth, float)
    terc = tercile_masks(lmh)
    good = np.isfinite(model).all(1) & (model[:, -1] > 0)
    n_bad = int((~good).sum())
    med = np.full((len(fractions), 3), np.nan)
    for i, f in enumerate(fractions):
        d = np.log10(fractional_radius(model, R, f) / fractional_radius(truth, R, f))
        for b, t in enumerate(terc):
            sel = t & good & np.isfinite(d)
            if sel.sum() >= 5:
                med[i, b] = np.median(d[sel])
    return float(np.sqrt(np.nanmean(med ** 2))), med, n_bad


def annular_term(model, truth, R, divisor="truth", edges=ANNULUS_EDGES_KPC):
    """(b) for ONE epoch: exp77's transformed annular distance. Returns (rms
    over galaxies of the per-galaxy mean squared transformed error, the
    (2,) per-annulus rms, n_bad). `divisor="truth"` scales both sides by the
    truth's total (exp77); `"own"` scales each side by its own total."""
    model, truth = np.asarray(model, float), np.asarray(truth, float)
    good = np.isfinite(model).all(1) & (model[:, -1] > 0)
    n_bad = int((~good).sum())
    a_m, a_t = annular_masses(model, R, edges), annular_masses(truth, R, edges)
    tot_t = truth[:, -1:]
    tot_m = model[:, -1:] if divisor == "own" else tot_t
    T_m = np.arcsinh(a_m / (ASINH_SCALE * tot_m)) / LN10
    T_t = np.arcsinh(a_t / (ASINH_SCALE * tot_t)) / LN10
    e2 = (T_m - T_t)[good] ** 2
    if len(e2) < 5:
        return np.nan, np.full(2, np.nan), n_bad
    return float(np.sqrt(np.mean(e2))), np.sqrt(np.mean(e2, axis=0)), n_bad


TERMS = {
    "radius": lambda m, t, R, lmh: radius_term(m, t, R, lmh)[0],
    "annular (truth total, exp77)": lambda m, t, R, lmh: annular_term(m, t, R, "truth")[0],
    "annular (own total)": lambda m, t, R, lmh: annular_term(m, t, R, "own")[0],
}


# --------------------------------------------------------------------------- #
# the probes
# --------------------------------------------------------------------------- #
def stretched(truth, R, s):
    """The same galaxy `s` times larger: M'(R) = M(R / s), by log-log
    interpolation (`coordinate.cum_at`), linear inside the first radius."""
    truth = np.asarray(truth, float)
    R = np.asarray(R, float)
    lq, lR = np.log10(R / s), np.log10(R)
    out = np.empty_like(truth)
    for i in range(truth.shape[0]):
        c = truth[i]
        with np.errstate(divide="ignore"):
            lc = np.log10(np.where(c > 0, c, np.nan))
        m = 10 ** np.interp(lq, lR, lc)
        out[i] = np.where(lq < lR[0], c[0] * (R / s) / R[0], m)
    return out


def run_probes(truth, R, lmh, seed=78, verbose=True):
    """The five probes on one epoch's truth (n, nR). Returns {term: {probe:
    value}} and raises if a declared blind spot is violated."""
    rng = np.random.default_rng(seed)
    truth = np.asarray(truth, float)
    n = len(truth)
    models = {
        "identity": truth.copy(),
        "amplitude x1.3": 1.3 * truth,
        "size x1.2": stretched(truth, R, 1.2),
        "central +10% of total": truth + 0.1 * truth[:, -1:],
        "size scatter, zero median": None,
    }
    s = 10 ** (0.15 * rng.standard_normal(n))
    s = s / np.median(s)                                 # exactly zero median log
    scat = np.empty_like(truth)
    for i in range(n):
        scat[i] = stretched(truth[i][None], R, s[i])[0]
    models["size scatter, zero median"] = scat
    out = {}
    if verbose:
        print(f"  {'probe':<28}" + "".join(f"{k:>30}" for k in TERMS))
    for pname, m in models.items():
        row = {}
        for tname, fn in TERMS.items():
            row[tname] = fn(m, truth, R, lmh)
        out[pname] = row
        if verbose:
            print(f"  {pname:<28}" + "".join(f"{row[k]:>30.4f}" for k in TERMS))
    # the declared blind spots, asserted
    r = {k: out[k]["radius"] for k in out}
    assert r["identity"] < 1e-12, r
    assert r["amplitude x1.3"] < 1e-12, "the radius term must not see a pure amplitude rescaling"
    expect = np.log10(1.2)
    assert 0.5 * expect < r["size x1.2"] < 1.5 * expect, (r["size x1.2"], expect)
    assert r["central +10% of total"] > 0.02, r["central +10% of total"]
    # a median over a tercile of n/3 galaxies of a 0.15 dex lognormal scatter
    # has a sampling error of about 1.25 x 0.15 / sqrt(n/3); the smoke sample
    # (33 per tercile) sits at that floor, the full sample far below it
    floor = 1.25 * 0.15 / np.sqrt(n / 3)
    assert r["size scatter, zero median"] < max(0.25 * r["size x1.2"], 2.0 * floor), (r, floor)
    a = {k: out[k]["annular (truth total, exp77)"] for k in out}
    assert a["identity"] < 1e-12
    assert a["size x1.2"] > 0.01
    if verbose:
        print(f"\n  radius term: zero at identity and under a pure amplitude rescaling; moves {r['size x1.2']:.3f} dex "
              f"for a x1.2 size shift (log10 1.2 = {expect:.3f}); moves {r['central +10% of total']:.3f} for a central "
              f"point mass; {r['size scatter, zero median']:.4f} for a zero-median per-galaxy size scatter.  ASSERTED")
        print(f"  annular term (exp77 form): a pure AMPLITUDE rescaling moves it {a['amplitude x1.3']:.3f} "
              f"(it is an amplitude term as much as a shape term); a CENTRAL point mass moves it "
              f"{a['central +10% of total']:.4f} (blind: the annuli and the truth's total are unchanged)")
    return out


def selftest(truth, R):
    """`fractional_radius` against `coordinate.size_radius` on the real truth."""
    for f in FRACTIONS:
        a = fractional_radius(truth, R, f)
        b = C.size_radius(truth, R, f)
        ok = np.isfinite(a) & np.isfinite(b)
        assert np.array_equal(np.isfinite(a), np.isfinite(b))
        assert np.allclose(a[ok], b[ok], rtol=1e-9, atol=1e-9), np.max(np.abs(a[ok] - b[ok]))
    print(f"  selftest: fractional_radius == coordinate.size_radius on {int(np.isfinite(a).sum())} "
          f"galaxy-epochs at R20/R50/R80  OK")
