"""exp63 Stage 2 — the two-channel mean, fitted at z=0.4 ONLY, predicted at
z=0.7-2.0.

The model is `model2.py` (compact Sersic channel + extended gompertz_log
channel, the compact share a logistic in the halo mass at the deposit's own
time, twelve parameters, nested to the incumbent). This script:

  1. builds the objective (D4): at z=0.4, on the sane + mh-complete mask,
         L = score_A^2 + score_F^2 + score_S^2
     with score_A and score_F exactly exp54's (amplitude at 100 kpc over
     SIGMA_A; the production shape loss over L_F_REF) and score_S the
     shells/log objective on the amplitude-pinned profiles over its value for
     the NESTED INCUMBENT on the exact engine — so every term is ~1 at the
     null and the innermost aperture is priced (exp54's loss never saw it);
  2. fits by L-BFGS-B (`hongshao.fitting.minimize_loss`, finite differences)
     from seven starts — the nested incumbent (named), Stage 1's default,
     four seeded jitters of it, and the default with n_c = 2 — checkpointing
     after every start (`.PARTIAL.npz`);
  3. applies the gates G2a-G2f (thresholds frozen in the Stage 2 plan) to the
     best fit AND to the null, and records the verdicts (the gates judge the
     model; they do not stop the outputs);
  4. PREDICTS z=0.7-2.0 by truncating the same integral: the standard battery,
     the both-sample bias tables, the C8 decliner split, and whether the model
     reproduces Stage 1's leverage correlations;
  5. four figures.

Nothing at z > 0.4 enters the loss. The fit uses the reduced quadrature
(model2.FIT_NODES, 1.2e-5 dex from the full one); every reported number uses
the full nodes.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp63_analytic_growth/stage2_fit.py [--smoke] [--fit-only] [--eval-only]
       [--starts a-b] [--delay]

`--delay` fits the 13-parameter model with the extended channel's
accretion-to-deposition delay (Stage 2b); outputs carry the `_delay` tag.
`--tau-fixed X` holds tau_d at X (a physics-set value) and fits the other
twelve at z=0.4; outputs carry the `_tauX` tag.
`--growth` fits the 13-parameter model whose split also depends on the halo's
growth rate at the deposit (Stage 2c); outputs carry the `_growth` tag.
`--growth-rel` uses the growth rate RELATIVE to the population's typical rate at
that time (Stage 2d); outputs carry the `_growthrel` tag.
`--freeze name=value[,name=value...]` holds the named parameters at the given
values in every start (equal-bounds box; they are not fitted and not counted as
railed). The single-epoch round (2026-08-28, the user's direction) freezes the
time exponents a_z, a_Mz, b_e at the incumbent's values and b_c at 0 — one epoch
cannot constrain a time exponent. `--tag X` appends X to the output tag.
`--levers g_c,g_e,w_min` adds the named levers of `model2.LEVER_NAMES` (each
nests at zero; the nested start still reproduces the incumbent);
`--extended-family sersic` makes the extended deposit a Sersic profile whose
index is the `c_e` slot (then no start nests the incumbent). Both are recorded
in the fit file and re-read by `--eval-only`.
`--objective outer` adds a fourth, OUTSKIRT-AWARE term to the loss: the rms
over galaxies of log10(model/data) of the outer annulus mass M*(50-148 kpc),
divided by its value for the nested incumbent (so 1 at the null). The
production shape loss pins the amplitude at 100 kpc and sees only three radii
beyond it (memory: the loss is blind past its own pin); this term prices the
outskirts explicitly. Recorded in the fit file.
`--compact-kpc` sizes the compact channel in physical kpc (`model2.Spec2(compact_in_kpc=True)`;
`log_f_c` is then log10 kpc); recorded in the fit file.
`--epoch k` (0-4 for z = 0.4, 0.7, 1.0, 1.5, 2.0; default 0) fits at THAT epoch
alone: the loss sees that epoch's data on that epoch's sane + mh-complete mask
with that epoch's SIGMA_A and halo-mass terciles; the model is the same integral
truncated at that epoch's anchor, so the fit sees only the MAH before it. The
evaluation then predicts all five epochs as before ("fitted at one epoch, four
predicted"). Recorded in the fit file as `fit_epoch`.
`--joint` fits ALL FIVE EPOCHS with one theta: the loss is the sum over epochs
of that epoch's (A^2 + F^2 + S^2 [+ B^2]), each on its own sane + mh-complete
mask with its own sigma_A, halo-mass terciles and null references (so every
term is 1 at the null and the null loss is 5 x the per-epoch null). One
`predict2` call over all galaxies and epochs per evaluation. `--start-from TAG`
replaces the "default" start with that fit's best theta (mapped by parameter
name; missing names keep the default), and `--set name=val,...` edits that
start — e.g. the z=0.4 kpc solution with the time exponents read from the
per-epoch sequence (Stage 5e). Recorded as `fit_epoch = -1` (joint).
`--objective binned` adds instead the term the VISUAL GATE reads: the rms over
the three halo-mass terciles and the 24 radii of the tercile-MEDIAN fractional
residual (model-data)/data at z=0.4, divided by its value for the nested
incumbent. Why: the per-galaxy terms are dominated by the intrinsic scatter of
the centre (rms 0.18 dex at 2-5 kpc in the top tercile) that no mean model can
reduce, so they are nearly blind to the binned median offset (0.065 dex there)
that the average CoGs show; this term prices that offset directly.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term", HERE):
    sys.path.insert(0, str(p))

import engine as E                                       # noqa: E402
import fit as F                                          # noqa: E402
import model2 as M2                                      # noqa: E402
import stage0_cost as S0                                 # noqa: E402
import stage0_rebaseline as S0R                          # noqa: E402
from hongshao import qa                                  # noqa: E402
from hongshao.fitting import minimize_loss               # noqa: E402
from hongshao.objective import Objective                 # noqa: E402

RULE = "=" * 92
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
QADIR = FIGDIR / "qa"
ANCHOR_Z = E.ANCHOR_Z
TABLE_KEYS = ("kpc:M(<10)", "kpc:M(<100)", "kpc:M(50-100)")
#: the gates, frozen (Stage 2 plan, Step 3)
G2A_MEDIAN, G2A_TOL, G2A_WIDTH_DATA, G2A_WIDTH_FRAC = 1.00, 0.05, 0.38, 0.7
G2B_TOL = 0.05
G2C_SLOPE_TRUTH, G2C_TOL = 1.42, 0.10
G2E_LOG_F_C, G2E_TOL = np.log10(0.02), 0.5
I3 = 3                                                   # R_GRID[3] = 4.92 kpc
MAX_EVALS = {"smoke": 60, "full": 3000}


# --------------------------------------------------------------------------- #
class Problem2:
    """The z=0.4 objective on the fit mask."""

    def __init__(self, spec2, curves, data, fit_rows, R, l_s_ref=None, outer=False, l_o_ref=None,
                 binned=False, lmh=None, l_b_ref=None, epoch=0):
        self.spec2, self.R, self.epoch = spec2, R, epoch
        self.outer, self.l_o_ref = outer, l_o_ref
        self.binned, self.l_b_ref = binned, l_b_ref
        self.i50 = int(np.argmin(np.abs(R - 50.0)))
        if binned:
            lm = np.asarray(lmh, float)[fit_rows]
            e = np.quantile(lm, [0, 1 / 3, 2 / 3, 1])
            self.terciles = [(lm >= e[b]) & (lm <= e[b + 1] + 1e-9) for b in range(3)]
        self.curves = [curves[i] for i in fit_rows]
        d = data[fit_rows, epoch, :]
        self.d = d
        self.A_d = np.log10(d[:, F.I100])
        self.F_d = (d / d[:, F.I100][:, None])[:, None, :]
        self.D_msun = d[:, None, :]                      # for the shells/log term, in Msun
        self.obj_F = Objective()
        self.obj_S = Objective(quantity="shells", residual="log")
        self.sig = F.SIGMA_A[epoch]
        self.l_s_ref = l_s_ref
        self.n_eval = 0

    def scores(self, theta, nodes=M2.FIT_NODES):
        m = M2.predict2(self.spec2, theta, self.curves, self.R, epochs=(self.epoch,), nodes=nodes)[:, 0, :]
        return self.score_model(m)

    def score_model(self, m):
        """The scores from a precomputed (n_fit, 24) model at this epoch."""
        good = np.isfinite(m).all(1) & (m[:, F.I100] > 0)
        n_bad = int((~good).sum())
        if good.sum() < 10:
            return np.nan, np.nan, np.nan, n_bad
        mg = m[good]
        A_m = np.log10(np.clip(mg[:, F.I100], 1.0, None))
        score_A = float(np.sqrt(np.mean(((A_m - self.A_d[good]) / self.sig) ** 2)))
        F_m = (mg / np.clip(mg[:, F.I100], 1.0, None)[:, None])[:, None, :]
        score_F = float(self.obj_F(F_m, self.F_d[good], self.R)) / F.L_F_REF
        # the shells/log objective floors empty shells at 1 Msun, so it must see
        # profiles in SOLAR MASSES: the model pinned to the data's M(<100 kpc)
        P_m = F_m * self.d[good][:, None, F.I100][:, :, None]
        l_s = float(self.obj_S(P_m, self.D_msun[good], self.R))
        score_S = l_s / self.l_s_ref if self.l_s_ref else l_s
        if self.binned:
            rel = (mg - self.d[good]) / self.d[good]
            med = np.array([np.median(rel[t[good]], axis=0) for t in self.terciles])
            l_b = float(np.sqrt(np.mean(med ** 2)))
            return score_A, score_F, score_S, n_bad, (l_b / self.l_b_ref if self.l_b_ref else l_b)
        if not self.outer:
            return score_A, score_F, score_S, n_bad
        d = self.d[good]
        o_m = np.clip(mg[:, -1] - mg[:, self.i50], 1.0, None)
        o_d = np.clip(d[:, -1] - d[:, self.i50], 1.0, None)
        l_o = float(np.sqrt(np.mean(np.log10(o_m / o_d) ** 2)))
        score_O = l_o / self.l_o_ref if self.l_o_ref else l_o
        return score_A, score_F, score_S, n_bad, score_O

    def loss(self, theta):
        self.n_eval += 1
        sc = self.scores(theta)
        a, f, s, n_bad = sc[:4]
        if len(sc) < 5 and (self.outer or self.binned):
            return F.FAIL                                # scores() bailed: too few galaxies evaluate
        o = sc[4] if (self.outer or self.binned) else 0.0
        if not np.all(np.isfinite([a, f, s, o])):
            return F.FAIL
        return a * a + f * f + s * s + o * o + F.FAIL * n_bad / len(self.curves)


class JointProblem2:
    """One theta, five epochs: the sum of each epoch's Problem2 loss, each on its
    own mask with its own references; one predict2 call per evaluation."""

    def __init__(self, spec2, curves, data, mask, lmh, R, th_inc, outer=False, binned=False,
                 epochs=(0, 1, 2, 3, 4)):
        self.spec2, self.R, self.epochs = spec2, R, list(epochs)
        mask = np.asarray(mask, bool)
        self.rows = {}
        for k in self.epochs:
            self.rows[k] = np.where(mask[:, k] & np.isfinite(data[:, k]).all(1) & (data[:, k] > 0).all(1))[0]
        self.all_rows = np.unique(np.concatenate(list(self.rows.values())))
        self.curves = [curves[i] for i in self.all_rows]
        pos = {r: i for i, r in enumerate(self.all_rows)}
        self.index = {k: np.array([pos[r] for r in self.rows[k]]) for k in self.epochs}
        self.problems = {}
        for k in self.epochs:
            ref = Problem2(M2.Spec2(), curves, data, self.rows[k], R, l_s_ref=None, outer=outer, l_o_ref=None,
                           binned=binned, lmh=lmh[:, k], l_b_ref=None, epoch=k)
            sc = ref.scores(M2.nested_theta(th_inc))
            self.problems[k] = Problem2(spec2, curves, data, self.rows[k], R, l_s_ref=sc[2], outer=outer,
                                        l_o_ref=(sc[4] if outer else None), binned=binned, lmh=lmh[:, k],
                                        l_b_ref=(sc[4] if binned else None), epoch=k)
        # per-epoch references live in the problems; these keep the fit-file fields uniform
        self.l_s_ref = np.array([self.problems[k].l_s_ref for k in self.epochs])
        self.l_o_ref = np.array([self.problems[k].l_o_ref or np.nan for k in self.epochs])
        self.l_b_ref = np.array([self.problems[k].l_b_ref or np.nan for k in self.epochs])
        self.n_eval = 0

    def per_epoch(self, theta, nodes=M2.FIT_NODES):
        m = M2.predict2(self.spec2, theta, self.curves, self.R, epochs=tuple(self.epochs), nodes=nodes)
        return {k: self.problems[k].score_model(m[self.index[k], j]) for j, k in enumerate(self.epochs)}

    def loss(self, theta):
        self.n_eval += 1
        total = 0.0
        for k, sc in self.per_epoch(theta).items():
            pr = self.problems[k]
            a, f, s, n_bad = sc[:4]
            if len(sc) < 5 and (pr.outer or pr.binned):
                return F.FAIL * len(self.epochs)
            o = sc[4] if (pr.outer or pr.binned) else 0.0
            if not np.all(np.isfinite([a, f, s, o])):
                return F.FAIL * len(self.epochs)
            total += a * a + f * f + s * s + o * o + F.FAIL * n_bad / len(pr.curves)
        return total

    def report(self, theta, label):
        print(f"  {label}: per-epoch (A, F, S[, B]) and loss")
        tot = 0.0
        for k, sc in self.per_epoch(theta, nodes=M2.FULL_NODES).items():
            vals = [v for v in (sc[0], sc[1], sc[2]) + ((sc[4],) if len(sc) > 4 else ())]
            lk = sum(v * v for v in vals)
            tot += lk
            print(f"    z={ANCHOR_Z[k]}: " + " ".join(f"{v:.3f}" for v in vals) + f" -> {lk:.4f}")
        print(f"    total {tot:.4f}")
        return tot


def starts_for(spec2, th_inc, rng):
    base = M2.with_levers_theta(M2.default_theta(th_inc, delay=spec2.delay, growth=spec2.growth_split), spec2)
    lo, hi = np.array(spec2.bounds()).T
    nested = M2.with_levers_theta(M2.nested_theta(th_inc, delay=spec2.delay, growth=spec2.growth_split), spec2)
    if spec2.compact_in_kpc:
        base[spec2.index("log_f_c")] = np.log10(4.0)
        nested[spec2.index("log_f_c")] = np.log10(4.0)
    out = [("nested", M2.clip_to_bounds(spec2, nested)), ("default", base)]
    for k in range(4):
        j = base + rng.normal(0, 0.05, base.shape) * np.maximum(np.abs(base), 0.3)
        out.append((f"jitter{k}", np.clip(j, lo, hi)))
    n2 = base.copy(); n2[spec2.index("n_c")] = 2.0
    out.append(("default_n2", n2))
    return out


def inner_slope(cogs, R):
    return np.log10(cogs[..., I3] / cogs[..., 0]) / np.log10(R[I3] / R[0])


def railed(spec2, theta, tol=1e-3, bounds=None):
    lo, hi = np.array(bounds if bounds is not None else spec2.bounds()).T
    return [n for n, v, a, b in zip(spec2.theta_names, theta, lo, hi)
            if b - a > tol and (v - a < tol * max(abs(a), 1) or b - v < tol * max(abs(b), 1))]


def gates(name, cogs, data, lmh, good, spec2, theta, curves, bounds=None):
    """The G2 table for one product. Returns dict gate -> (value_str, passed)."""
    R = F.R_GRID
    out = {}
    sm, sd = inner_slope(cogs[good][:, 0], R), inner_slope(data[good][:, 0], R)
    med, wid = float(np.median(sm)), float(np.percentile(sm, 84) - np.percentile(sm, 16))
    out["G2a inner slope"] = (f"median {med:.3f} (data {np.median(sd):.3f}), width {wid:.2f} (data {np.percentile(sd, 84) - np.percentile(sd, 16):.2f})",
                              abs(med - G2A_MEDIAN) <= G2A_TOL and wid >= G2A_WIDTH_FRAC * G2A_WIDTH_DATA)
    lm = lmh[good][:, 0]; edges = np.percentile(lm, [0, 100 / 3, 200 / 3, 100])
    worst = 0.0
    for c in range(3):
        sel = (lm >= edges[c]) & (lm <= edges[c + 1])
        mm, dd = cogs[good][sel][:, 0], data[good][sel][:, 0]
        raw = np.median((mm - dd) / dd, axis=0)
        pin = mm / mm[:, F.I100][:, None] * dd[:, F.I100][:, None]
        worst = max(worst, float(np.max(np.abs(raw))), float(np.max(np.abs(np.median((pin - dd) / dd, axis=0)))))
    out["G2b tercile profiles"] = (f"worst |median residual| {100 * worst:.1f}%", worst <= G2B_TOL)
    t, m = qa.measure_all(cogs[good], data[good], R)[:2]
    x_t, y_t = np.log10(t["kpc:M(<30)"][:, 0]), np.log10(t["kpc:M(30-50)"][:, 0])
    x_m, y_m = np.log10(np.clip(m["kpc:M(<30)"][:, 0], 1, None)), np.log10(np.clip(m["kpc:M(30-50)"][:, 0], 1, None))
    st, sm_ = qa.plane_stats(x_t, y_t)["slope"], qa.plane_stats(x_m, y_m)["slope"]
    out["G2c plane slope"] = (f"M(30-50)|M(<30): model {sm_:.2f}, truth {st:.2f}", abs(sm_ - st) <= G2C_TOL)
    big = M2.predict2(spec2, theta, curves[:20], np.array([1e9]))[:, 0, 0]
    cm = M2.channel_masses(spec2, theta, curves[:20]).sum(1)
    cons = float(np.max(np.abs(big / cm - 1)))
    out["G2d conservation"] = (f"max |M(<1e9)/deposited - 1| = {cons:.1e}", cons < 1e-6)
    p = spec2.unpack(theta)
    out["G2e sizes physical"] = (f"log f_c {p['log_f_c']:+.2f} (0.02 -> {G2E_LOG_F_C:+.2f}), log f_e {p['log_f_e']:+.2f}",
                                 p["log_f_c"] < p["log_f_e"] and abs(p["log_f_c"] - G2E_LOG_F_C) <= G2E_TOL)
    share = float(M2.compact_share(p, 13.5))
    rl = railed(spec2, theta, bounds=bounds)
    out["G2f identifiable"] = (f"railed {rl or 'none'}; compact share at 10^13.5 = {share:.2f}",
                               not rl and 0.05 <= share <= 0.95)
    print(f"\n  GATES — {name}")
    for g, (v, ok) in out.items():
        print(f"    {g:<24} {'PASS' if ok else 'FAIL':<5} {v}")
    return out


def decliner_split(cogs, data, good, label):
    """C8: median log10(model/truth) of M*(<4.92 kpc) per epoch, split on whether
    the measured M*(<4.92) fell between z=2 and z=0.4."""
    d, m = data[good], cogs[good]
    dec = d[:, 4, I3] > d[:, 0, I3]
    err = np.log10(np.clip(m[:, :, I3], 1, None)) - np.log10(d[:, :, I3])
    print(f"    {label:<22}" + "".join(f"{np.median(err[:, j]):>+9.3f}" for j in range(5))
          + f"   span {np.ptp([np.median(err[:, j]) for j in range(5)]):.3f}")
    for nm, sel in (("  centre declined", dec), ("  did not decline", ~dec)):
        print(f"    {nm + f' ({100 * sel.mean():.0f}%)':<22}" + "".join(
            f"{np.median(err[sel, j]):>+9.3f}" for j in range(5))
              + f"   span {np.ptp([np.median(err[sel, j]) for j in range(5)]):.3f}")
    return err, dec


def leverage_model(spec2, theta, curves, lmh):
    from scipy.stats import spearmanr
    from stage1_deconvolve import halo_variables, partial_spearman
    cm = M2.channel_masses(spec2, theta, curves)
    share = cm[:, 0] / cm.sum(1)
    _, hv = halo_variables([None] * 0, curves) if False else (None, None)
    return share


# --------------------------------------------------------------------------- #
def _spec(delay, growth, growth_rel=False, levers=(), extended_family=M2.EXTENDED_FAMILY, compact_in_kpc=False):
    if delay:
        return M2.Spec2.with_delay()
    if growth_rel:
        return M2.Spec2.with_growth_rel()
    if growth:
        return M2.Spec2.with_growth_split()
    return M2.Spec2.with_levers(levers, extended_family=extended_family, compact_in_kpc=compact_in_kpc)


def spec_from_fit(fz):
    """Rebuild the Spec2 a fit file was made with."""
    names = tuple(str(n) for n in fz["theta_names"])
    fam = str(fz["extended_family"]) if "extended_family" in fz.files else M2.EXTENDED_FAMILY
    if "tau_d" in names:
        return M2.Spec2.with_delay()
    if "g_rel" in names:
        return M2.Spec2.with_growth_rel()
    if "g_split" in names:
        return M2.Spec2.with_growth_split()
    kpc = bool(fz["compact_in_kpc"]) if "compact_in_kpc" in fz.files else False
    return M2.Spec2.with_levers(tuple(n for n in names if n in M2.LEVER_NAMES), extended_family=fam,
                                compact_in_kpc=kpc)


def parse_freeze(arg):
    """'a_z=0.505,b_c=0' -> {'a_z': 0.505, 'b_c': 0.0}."""
    if not arg:
        return {}
    return {kv.split("=")[0]: float(kv.split("=")[1]) for kv in arg.split(",")}


def run_fit(smoke, starts_sel, tag, delay=False, tau_fixed=None, growth=False, growth_rel=False,
            freeze=None, levers=(), extended_family=M2.EXTENDED_FAMILY, objective="production",
            compact_in_kpc=False, epoch=0, joint=False, start_from=None, set_values=None):
    freeze = freeze or {}
    set_values = set_values or {}
    if epoch:
        print(f"  FIT EPOCH {epoch}: z = {ANCHOR_Z[epoch]} — the loss sees this epoch's data only; the integral "
              f"is truncated at its anchor (the MAH before it)")
    outer, binned = objective == "outer", objective == "binned"
    recs, data, mask, lmh, spec_inc, th_inc, _ = S0.build(smoke)
    curves = E.build_curves(recs, verbose=False)
    R = F.R_GRID
    spec2 = _spec(delay, growth, growth_rel, levers, extended_family, compact_in_kpc)
    if compact_in_kpc:
        print("  compact channel sized in PHYSICAL kpc (log_f_c = log10 kpc)")
    if levers or extended_family != M2.EXTENDED_FAMILY:
        print(f"  levers {list(levers) or 'none'}; extended kernel {extended_family}; "
              f"{spec2.n_theta} parameters")
    if growth_rel:
        lt_ref, a_ref = M2.set_alpha_ref(curves)
        print(f"  alpha_ref(t): population-median dlnM/dlnt at the nodes, {a_ref.min():.2f}-{a_ref.max():.2f}")
    bounds = spec2.bounds()
    if tau_fixed is not None:
        # tau_d set from physics, not fitted: an equal-bounds box freezes it
        bounds[spec2.index("tau_d")] = (tau_fixed, tau_fixed)
        print(f"  tau_d FIXED at {tau_fixed} Hubble times at accretion (not fitted)")
    for nm, val in freeze.items():
        bounds[spec2.index(nm)] = (val, val)
        print(f"  {nm} FROZEN at {val:+.4f} (not fitted)")
    fit_rows = np.where(np.asarray(mask, bool)[:, epoch] & np.isfinite(data[:, epoch]).all(1)
                        & (data[:, epoch] > 0).all(1))[0]
    print(f"  fit sample: {len(fit_rows)} of {len(curves)} galaxies (sane + mh-complete at z={ANCHOR_Z[epoch]})")
    if joint:
        pr = JointProblem2(spec2, curves, data, mask, lmh, R, th_inc, outer=outer, binned=binned)
        print(f"  JOINT FIT at all five epochs: masks " + " / ".join(f"{len(pr.rows[k])}" for k in pr.epochs)
              + f" galaxies (union {len(pr.all_rows)}); each epoch's terms are 1 at the null")
        fit_rows = pr.all_rows

    # the shells/log reference: the NESTED incumbent on the exact engine (frozen)
    # the references (shells/log, and the outer term if used) are ALWAYS the
    # gompertz_log nested incumbent on the exact engine, whatever the family
    if not joint:
      pr_ref = Problem2(M2.Spec2(), curves, data, fit_rows, R, l_s_ref=None, outer=outer, l_o_ref=None,
                      binned=binned, lmh=lmh[:, epoch], l_b_ref=None, epoch=epoch)
      ref = pr_ref.scores(M2.nested_theta(th_inc))
      pr = Problem2(spec2, curves, data, fit_rows, R, l_s_ref=ref[2], outer=outer,
                    l_o_ref=(ref[4] if outer else None), binned=binned, lmh=lmh[:, epoch],
                    l_b_ref=(ref[4] if binned else None), epoch=epoch)
      print(f"  references from the nested incumbent: shells/log {ref[2]:.5f}"
            + (f", outer log-rms M*(50-148) {ref[4]:.5f} dex" if outer else "")
            + (f", binned tercile-median rms {100 * ref[4]:.3f}%" if binned else "") + " (each -> 1 by definition)")
    th_nested = M2.with_levers_theta(M2.nested_theta(th_inc, delay=spec2.delay, growth=spec2.growth_split), spec2)
    if joint:
        l_null = pr.loss(th_nested)
        pr.report(th_nested, "the null (nested incumbent, exact engine)")
    else:
      sc = pr.scores(th_nested)
      a, f, s, nb = sc[:4]
      l_null = pr.loss(th_nested)
      print(f"  the null (nested incumbent{'' if spec2.extended_family == M2.EXTENDED_FAMILY else ' — NOT a nesting under this family'}, "
            f"exact engine, z=0.4): score_A {a:.4f}, score_F {f:.4f}, score_S {s:.4f}"
            + (f", score_O {sc[4]:.4f}" if outer else "") + (f", score_B {sc[4]:.4f}" if binned else "")
            + f", loss {l_null:.6f}, failures {nb}")
      sc2 = pr.scores(th_nested, nodes=M2.FULL_NODES)
      print(f"  the same on the full nodes: {sc2[0]:.4f} / {sc2[1]:.4f} / {sc2[2]:.4f} (fit nodes are adequate)")

    rng = np.random.default_rng(63)
    starts = starts_for(spec2, th_inc, rng)
    if start_from is not None:
        fz0 = np.load(OUTDIR / f"stage2_fit{start_from}.npz", allow_pickle=True)
        names0 = [str(n) for n in fz0["theta_names"]]
        th_from = starts[1][1].copy()
        for i, nm in enumerate(spec2.theta_names):
            if nm in names0:
                th_from[i] = float(fz0["theta_best"][names0.index(nm)])
        for nm, val in set_values.items():
            th_from[spec2.index(nm)] = val
        starts[1] = (f"from{start_from}", M2.clip_to_bounds(spec2, th_from))
        print(f"  start 'from{start_from}': that fit's theta" + (f" with {set_values}" if set_values else ""))
    if tau_fixed is not None:
        starts = [(nm, np.r_[th[:-1], tau_fixed]) for nm, th in starts]
    for nm, val in freeze.items():
        for _, th in starts:
            th[spec2.index(nm)] = val
    if starts_sel is not None:
        a_, b_ = starts_sel
        starts = starts[a_:b_ + 1]
    results = []
    partial = OUTDIR / f"stage2_fit{tag}.PARTIAL.npz"
    max_evals = MAX_EVALS["smoke" if smoke else "full"]
    for name, p0 in starts:
        t0 = time.time(); pr.n_eval = 0
        l0 = pr.loss(p0)
        r = minimize_loss(pr.loss, p0, method="lbfgsb", bounds=bounds,
                          max_evals=max_evals, fd_step=1e-5)
        dt = time.time() - t0
        rl = railed(spec2, r.x, bounds=bounds)
        print(f"  start {name:<11} loss {l0:.6f} -> {r.fun:.6f}  ({pr.n_eval} evals, {dt / 60:.1f} min)"
              f"{'  RAILED: ' + ','.join(rl) if rl else ''}", flush=True)
        results.append(dict(name=name, theta0=p0, theta=np.asarray(r.x), loss=float(r.fun),
                            loss0=float(l0), n_eval=pr.n_eval, railed=rl))
        OUTDIR.mkdir(parents=True, exist_ok=True)
        np.savez(partial, names=np.array([q["name"] for q in results]),
                 thetas=np.array([q["theta"] for q in results]),
                 losses=np.array([q["loss"] for q in results]),
                 losses0=np.array([q["loss0"] for q in results]),
                 l_s_ref=pr.l_s_ref, loss_null=l_null, theta_nested=th_nested,
                 fit_rows=fit_rows, theta_names=np.array(spec2.theta_names))
    best = min(results, key=lambda q: q["loss"])
    print(f"\n  best start: {best['name']} loss {best['loss']:.6f} (null {l_null:.6f}, "
          f"{100 * (1 - best['loss'] / l_null):+.1f}%)")
    print("  " + "  ".join(f"{n}={v:+.4f}" for n, v in zip(spec2.theta_names, best["theta"])))
    if joint:
        pr.report(best["theta"], "best (full nodes)")
    out = OUTDIR / f"stage2_fit{tag}.npz"
    extra = {}
    if growth_rel:
        extra = dict(alpha_ref_lt=M2._ALPHA_REF["lt"], alpha_ref=M2._ALPHA_REF["alpha"])
    np.savez(out, theta_best=best["theta"], loss_best=best["loss"], best_name=best["name"], **extra,
             frozen_names=np.array(list(freeze)), frozen_values=np.array(list(freeze.values())),
             bounds=np.array(bounds), extended_family=spec2.extended_family, objective=objective,
             compact_in_kpc=spec2.compact_in_kpc, fit_epoch=(-1 if joint else epoch),
             l_o_ref=(pr.l_o_ref if outer else np.nan), l_b_ref=(pr.l_b_ref if binned else np.nan),
             names=np.array([q["name"] for q in results]),
             thetas=np.array([q["theta"] for q in results]),
             losses=np.array([q["loss"] for q in results]),
             losses0=np.array([q["loss0"] for q in results]),
             l_s_ref=pr.l_s_ref, loss_null=l_null, theta_nested=th_nested, theta_incumbent=th_inc,
             fit_rows=fit_rows, theta_names=np.array(spec2.theta_names))
    print(f"  saved {out.relative_to(ROOT)}")


def run_eval(smoke, tag, tables_only=False, delay=False, growth=False, growth_rel=False, qadir=None):
    QADIR = FIGDIR / (qadir or "qa")               # `--qadir NAME` keeps a product's battery in its own folder
    recs, data, mask, lmh, spec_inc, th_inc, _ = S0.build(smoke)
    curves = E.build_curves(recs, verbose=False)
    R = F.R_GRID
    fz = np.load(OUTDIR / f"stage2_fit{tag}.npz", allow_pickle=True)
    spec2 = spec_from_fit(fz)
    fit_epoch = int(fz["fit_epoch"]) if "fit_epoch" in fz.files else 0
    FIT_EPOCH["k"] = fit_epoch
    print("  fitted at ALL FIVE EPOCHS jointly (nothing is a prediction)" if fit_epoch < 0 else
          f"  fitted at z = {ANCHOR_Z[fit_epoch]} only; every other epoch is a prediction")
    if spec2.growth_rel:
        M2.set_alpha_ref(curves)
    theta = np.asarray(fz["theta_best"], float)
    th_nested = np.asarray(fz["theta_nested"], float)
    print(f"  best theta ({str(fz['best_name'])}, loss {float(fz['loss_best']):.6f} vs null "
          f"{float(fz['loss_null']):.6f}):")
    print("  " + "  ".join(f"{n}={v:+.4f}" for n, v in zip(spec2.theta_names, theta)))
    cogs = {"baseline": M2.predict2(spec2, th_nested, curves, R),
            "stage2": M2.predict2(spec2, theta, curves, R)}
    good = np.isfinite(data).all(axis=(1, 2)) & (data > 0).all(axis=(1, 2))
    for m in cogs.values():
        good &= np.isfinite(m).all(axis=(1, 2)) & (m > 0).all(axis=(1, 2))
    msk = np.asarray(mask, bool)[good]
    logms = np.log10(np.clip(data[good][:, 0, -1], 1.0, None))
    print(f"  QA sample {int(good.sum())} galaxies; fit mask keeps "
          + "/".join(f"{int(msk[:, j].sum())}" for j in range(5)) + " per epoch")

    g_null = gates("baseline (nested incumbent, exact engine)", cogs["baseline"], data, lmh, good, spec2, th_nested, curves)
    g_fit = gates("stage2 two-channel fit", cogs["stage2"], data, lmh, good, spec2, theta, curves,
                  bounds=(fz["bounds"] if "bounds" in fz.files else None))

    print(f"\n{RULE}\nPREDICTIONS at z=0.7-2.0 (never fitted) — median relative bias [%], all / fit mask\n{RULE}")
    out = {}
    QADIR.mkdir(parents=True, exist_ok=True)
    for m in cogs:
        out[m] = qa.evaluate(cogs[m][good], data[good], R, list(ANCHOR_Z), name=f"exp63_stage2_{m}{tag}",
                             figdir=(None if (tables_only or m == "baseline") else QADIR),
                             figures=not (tables_only or m == "baseline"), verbose=(m == "stage2"),
                             bin_by=lmh[good][:, 0], bin_label=r"logM$_h$(z=0.4)",
                             bin_by_ms=logms, ms_label=r"logM$_*$ (total)")
    for key in TABLE_KEYS:
        print(f"\n  {key}")
        print(f"  {'product':<12}{'sample':<16}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
        for m in cogs:
            t, mm = out[m]["truth"][key], out[m]["model"][key]
            print(f"  {m:<12}{'all':<16}" + "".join(
                f"{100 * np.nanmedian(qa.relerr(mm[:, j], t[:, j])):>9.1f}%" for j in range(5)))
            print(f"  {'':<12}{'fit mask':<16}" + "".join(
                f"{100 * np.nanmedian(qa.relerr(mm[msk[:, j], j], t[msk[:, j], j])):>9.1f}%" for j in range(5)))
    print("\n  tier 3, profile max|rel| beyond 5 kpc (median)")
    for m in cogs:
        print(f"  {m:<12}" + "".join(f"{np.nanmedian(out[m]['mr_out'][:, j]):>10.4f}" for j in range(5)))
    print(f"\n  C8 — median log10(model/truth) of M*(<4.92 kpc) per epoch (z=0.4 ... 2.0), decliner split")
    errs = {}
    for m in cogs:
        errs[m] = decliner_split(cogs[m], data, good, m)

    # Stage 1's leverage, reproduced?
    from stage1_deconvolve import halo_variables, partial_spearman
    lmh0, hv = halo_variables(recs, curves)
    cm = M2.channel_masses(spec2, theta, curves)
    share = cm[:, 0] / cm.sum(1)
    print(f"\n  the model's compact share at z=0.4: median {np.median(share):.2f}, 16-84% "
          f"{np.percentile(share, 16):.2f}-{np.percentile(share, 84):.2f}; corr with logMh "
          f"{np.corrcoef(lmh0, share)[0, 1]:+.2f} (Stage 1 required: extended share 0.35->0.57, r=+0.53)")
    print("  partial Spearman at fixed logMh of the model's compact share (Stage 1's inner share in brackets):")
    s1 = np.load(OUTDIR / "stage1_deconvolve.npz", allow_pickle=True) if (OUTDIR / "stage1_deconvolve.npz").exists() else None
    ref = {}
    if s1 is not None:
        vars_ = list(s1["leverage_vars"]); stats_ = list(s1["leverage_stats"])
        L = s1["leverage_gompertz_c0.80"]
        ref = {v: L[stats_.index("inner share s<5"), vars_.index(v)] for v in vars_}
    print("    " + "  ".join(f"{v} {partial_spearman(share, x, lmh0):+.2f} [{ref.get(v, np.nan):+.2f}]"
                             for v, x in hv.items()))

    # figures
    paths = [S0R.compare_figure(cogs, data, lmh, good, f"exp63_stage2_residuals{tag}",
                                models=("baseline", "stage2"), styles={"baseline": "--", "stage2": "-"},
                                pretty={"baseline": "baseline", "stage2": "stage2"},
                                title="exp63 Stage 2 — the two-channel fit (solid) against the exact-engine incumbent (dashed); fitted at z=0.4 only")]
    paths += figures_stage2(cogs, data, lmh, good, spec2, theta, th_nested, curves, share, tag)
    for p in paths:
        print(f"  wrote {p.relative_to(ROOT)}")
    np.savez(OUTDIR / f"stage2_eval{tag}.npz", theta=theta, good=good,
             inner_slope_model=inner_slope(cogs["stage2"][good][:, 0], R),
             inner_slope_baseline=inner_slope(cogs["baseline"][good][:, 0], R),
             inner_slope_data=inner_slope(data[good][:, 0], R), share=share,
             gates_fit=np.array([[g, v, str(ok)] for g, (v, ok) in g_fit.items()]),
             gates_null=np.array([[g, v, str(ok)] for g, (v, ok) in g_null.items()]),
             **{f"bias_{m}_{k}_{s}": np.array([np.nanmedian(qa.relerr(out[m]["model"][k][sel[:, j], j],
                                                                       out[m]["truth"][k][sel[:, j], j]))
                                               for j in range(5)])
                for m in cogs for k in TABLE_KEYS for s, sel in (("all", np.ones_like(msk)), ("fitmask", msk))},
             **{f"c8_{m}": errs[m][0] for m in cogs}, decliner=errs["stage2"][1])
    print(f"  saved {(OUTDIR / f'stage2_eval{tag}.npz').relative_to(ROOT)}")


FIT_EPOCH = {"k": 0}                                     # set by run_eval: which epoch the figures shade


def _shade_fitted(a):
    if FIT_EPOCH["k"] < 0:
        a.axvspan(0.35, 2.05, color="#E69F00", alpha=0.08)
        return
    z = ANCHOR_Z[FIT_EPOCH["k"]]
    a.axvspan(z - 0.05, z + 0.05, color="#E69F00", alpha=0.15)


def figures_stage2(cogs, data, lmh, good, spec2, theta, th_nested, curves, share, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hongshao.plotting import set_style, save_fig
    from hongshao.qa import _tex, _pct
    set_style()
    R = F.R_GRID
    paths = []
    # gates figure: inner slope + the plane
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    sd = inner_slope(data[good][:, 0], R)
    bins = np.linspace(0, 2.2, 45)
    ax[0].hist(sd, bins=bins, alpha=0.4, color="#E69F00", label=f"TNG: median {np.median(sd):.2f}")
    for m, col in (("baseline", "#999999"), ("stage2", "#0072B2")):
        sm = inner_slope(cogs[m][good][:, 0], R)
        ax[0].hist(sm, bins=bins, histtype="step", lw=2, color=col, label=f"{m}: median {np.median(sm):.2f}")
    ax[0].set_xlabel(r"CoG slope $d\log M_*/d\log R$ over 2-4.9 kpc, z=0.4"); ax[0].set_ylabel("galaxies")
    ax[0].legend(fontsize=8); ax[0].set_title("G2a: the inner slope")
    t, m2 = qa.measure_all(cogs["stage2"][good], data[good], R)[:2]
    ax[1].scatter(np.log10(t["kpc:M(<30)"][:, 0]), np.log10(t["kpc:M(30-50)"][:, 0]), s=4, alpha=0.3, color="#E69F00", label="truth")
    ax[1].scatter(np.log10(m2["kpc:M(<30)"][:, 0]), np.log10(m2["kpc:M(30-50)"][:, 0]), s=4, alpha=0.3, color="#0072B2", label="stage2 mean")
    ax[1].set_xlabel(_tex("log M*(<30 kpc)")); ax[1].set_ylabel("log M*(30-50 kpc)"); ax[1].legend(fontsize=8)
    ax[1].set_title("G2c: the plane (mean model, no layer)")
    fig.tight_layout(); save_fig(fig, FIGDIR / f"exp63_stage2_gates{tag}"); paths.append(FIGDIR / f"exp63_stage2_gates{tag}.png")

    # channels figure: compact share vs logMh, and the two size laws vs z with Stage 1's requirement
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    lm0 = lmh[:, 0]
    ax[0].scatter(lm0, share, s=5, alpha=0.4, color="#0072B2", label="model: compact share of M*(z=0.4)")
    s1p = OUTDIR / "stage1_deconvolve.npz"
    if s1p.exists():
        s1 = np.load(s1p, allow_pickle=True)
        w = s1["w_gompertz_c0.80"]; sg = s1["s_grid"]
        inner = w[:, sg < 15].sum(1) / w.sum(1)
        ax[0].scatter(s1["lmh"], inner, s=5, alpha=0.25, color="#E69F00", label=_tex("Stage 1 required: share at s<15 kpc"))
    ax[0].set_xlabel(_tex("log Mh(z=0.4)")); ax[0].set_ylabel("compact share"); ax[0].legend(fontsize=8)
    ax[0].set_title("the split against halo mass")
    p = spec2.unpack(theta)
    zz = np.linspace(0.4, 8, 100)
    ax[1].plot(zz, p["log_f_c"] + p["b_c"] * np.log10(1 + zz), "-", color="#0072B2", lw=2, label="compact channel s_c/R200c")
    ax[1].plot(zz, p["log_f_e"] + p["b_e"] * np.log10(1 + zz), "-", color="#D55E00", lw=2, label="extended channel s_e/R200c")
    pn = spec2.unpack(th_nested)
    ax[1].plot(zz, pn["log_f_e"] + pn["b_e"] * np.log10(1 + zz), "k--", lw=1.2, label="incumbent's single law")
    if s1p.exists():
        ls_, zs_ = s1["log_s_over_r200_gompertz_c0.80"], s1["z_star"]
        zb = np.array([0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0, 15.0])
        zc, med, lo, hi = [], [], [], []
        for a_, b_ in zip(zb[:-1], zb[1:]):
            sel = (zs_ >= a_) & (zs_ < b_) & np.isfinite(ls_)
            if sel.sum() >= 20:
                zc.append(np.median(zs_[sel])); med.append(np.median(ls_[sel]))
                lo.append(np.percentile(ls_[sel], 16)); hi.append(np.percentile(ls_[sel], 84))
        ax[1].fill_between(zc, lo, hi, color="gray", alpha=0.2, label="Stage 1 required s*/R200c (16-84)")
        ax[1].plot(zc, med, "o", color="gray")
    ax[1].set_xscale("log"); ax[1].set_xlabel("redshift of the deposit"); ax[1].set_ylabel(r"$\log_{10}(s/R_{200c})$")
    ax[1].legend(fontsize=7); ax[1].set_title("the two size laws")
    fig.tight_layout(); save_fig(fig, FIGDIR / f"exp63_stage2_channels{tag}"); paths.append(FIGDIR / f"exp63_stage2_channels{tag}.png")

    # C8 figure: the central error against epoch, whole population and the decliner split
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    dec = data[good][:, 4, I3] > data[good][:, 0, I3]
    for a, (nm, col) in zip(ax, (("baseline", "#999999"), ("stage2", "#0072B2"))):
        e = np.log10(np.clip(cogs[nm][good][:, :, I3], 1, None)) - np.log10(data[good][:, :, I3])
        for sel, ls, lab in ((np.ones_like(dec, bool), "-", "all galaxies"),
                             (~dec, "--", "centre did NOT decline from z=2 to 0.4"), (dec, ":", "centre declined")):
            med = [np.median(e[sel, j]) for j in range(5)]
            a.plot(ANCHOR_Z, med, ls, color=col, lw=2.2, marker="o",
                   label=f"{lab} ({100 * sel.mean():.0f}{_pct()}): span {np.ptp(med):.3f} dex")
            if ls == "--":
                a.fill_between(ANCHOR_Z, [np.percentile(e[sel, j], 16) for j in range(5)],
                               [np.percentile(e[sel, j], 84) for j in range(5)], color=col, alpha=0.12)
        a.axhline(0, color="k", lw=0.8); _shade_fitted(a)
        a.set_xlabel("z"); a.legend(fontsize=7.5, loc="lower left")
        a.set_title("incumbent, exact engine (fitted at all five epochs)" if nm == "baseline"
                    else ("two-channel mean (fitted at all five epochs jointly)" if FIT_EPOCH["k"] < 0 else
                          f"two-channel mean (fitted at z={ANCHOR_Z[FIT_EPOCH['k']]} only, shaded)"), fontsize=9.5)
    ax[0].set_ylabel(_tex("median log10(model/truth) of M*(<4.92 kpc)")); ax[0].set_ylim(-0.3, 0.12)
    fig.suptitle("C8: the central error against epoch (band: 16-84% of the non-decliners)", fontsize=10.5)
    fig.tight_layout(); save_fig(fig, FIGDIR / f"exp63_stage2_c8{tag}"); paths.append(FIGDIR / f"exp63_stage2_c8{tag}.png")

    # predictions figure: bias vs epoch, both samples
    ev = None
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True)
    msk = np.asarray(S0.build(False)[2] if False else None) if False else None
    for a, key in zip(ax, TABLE_KEYS):
        for m, col in (("baseline", "#999999"), ("stage2", "#0072B2")):
            t, mm = qa.measure_all(cogs[m][good], data[good], R)[:2]
            b_all = [100 * np.nanmedian(qa.relerr(mm[key][:, j], t[key][:, j])) for j in range(5)]
            a.plot(ANCHOR_Z, b_all, "o-", color=col, lw=2, label=f"{m}, all")
        a.axhline(0, color="k", lw=0.8); _shade_fitted(a)
        a.set_title(_tex(key)); a.set_xlabel("z"); a.set_ylabel(f"median relative bias [{_pct()}]")
        a.legend(fontsize=7)
    fig.suptitle("fitted at all five epochs jointly (one theta)" if FIT_EPOCH["k"] < 0 else
                 f"fitted at z={ANCHOR_Z[FIT_EPOCH['k']]} (shaded); every other epoch is a prediction", fontsize=11)
    fig.tight_layout(); save_fig(fig, FIGDIR / f"exp63_stage2_predictions{tag}"); paths.append(FIGDIR / f"exp63_stage2_predictions{tag}.png")
    return paths


if __name__ == "__main__":
    args = sys.argv[1:]
    smoke = "--smoke" in args
    delay = "--delay" in args or "--tau-fixed" in args
    tau_fixed = float(args[args.index("--tau-fixed") + 1]) if "--tau-fixed" in args else None
    growth = "--growth" in args
    growth_rel = "--growth-rel" in args
    freeze = parse_freeze(args[args.index("--freeze") + 1]) if "--freeze" in args else {}
    levers = tuple(args[args.index("--levers") + 1].split(",")) if "--levers" in args else ()
    ext_fam = args[args.index("--extended-family") + 1] if "--extended-family" in args else M2.EXTENDED_FAMILY
    objective = args[args.index("--objective") + 1] if "--objective" in args else "production"
    compact_kpc = "--compact-kpc" in args
    epoch = int(args[args.index("--epoch") + 1]) if "--epoch" in args else 0
    joint = "--joint" in args
    start_from = args[args.index("--start-from") + 1] if "--start-from" in args else None
    set_values = parse_freeze(args[args.index("--set") + 1]) if "--set" in args else {}
    tag = (f"_tau{tau_fixed:g}" if tau_fixed is not None else ("_delay" if delay else ("_growthrel" if growth_rel else ("_growth" if growth else ""))))
    tag += (args[args.index("--tag") + 1] if "--tag" in args else "") + ("_smoke" if smoke else "")
    sel = None
    if "--starts" in args:
        a_, b_ = args[args.index("--starts") + 1].split("-")
        sel = (int(a_), int(b_))
    print(f"{RULE}\nexp63 STAGE 2 — the two-channel mean, fitted at z=0.4 only"
          f"{' (SMOKE)' if smoke else ''}\n{RULE}\n")
    if "--eval-only" not in args:
        run_fit(smoke, sel, tag, delay=delay, tau_fixed=tau_fixed, growth=growth, growth_rel=growth_rel,
                freeze=freeze, levers=levers, extended_family=ext_fam, objective=objective,
                compact_in_kpc=compact_kpc, epoch=epoch, joint=joint, start_from=start_from,
                set_values=set_values)
    if "--fit-only" not in args:
        run_eval(smoke, tag, tables_only="--tables-only" in args, delay=delay, growth=growth, growth_rel=growth_rel,
                 qadir=(args[args.index("--qadir") + 1] if "--qadir" in args else None))
