"""exp57 Stage 3 — fit the expansion laws.

One process per law (`--only X1`), so a kill loses at most one fit and the
three run in parallel on ten cores.

**THE NESTED START IS MANDATORY AND IS CHECKED, NOT TRUSTED.** Start 0 is the
exp54 incumbent with the expansion switched off, and `assert_nests` verifies it
reproduces the incumbent's loss of 1.874253 before the optimiser is allowed to
move. That identity is what makes "the enrichment did not help" a statement
about the model rather than about the optimiser: from that start the fit CANNOT
converge to a worse loss. exp54's own standing rule is to lift a nested start
BY NAME with an assertion that it reproduces the parent's loss, because
positional splicing was silently wrong once already.

**Why the other starts are placed where they are.** Nelder-Mead builds its
initial simplex by perturbing each coordinate by 5%, or by 0.00025 when the
coordinate is zero. The nested start has `A = 0` exactly, so on its own it
would explore the expansion axis in steps of 0.00025 — which is not a search,
it is a stall. The remaining starts therefore place `A` at 0.5, 2.0 and −0.3
explicitly: two magnitudes of expansion and one of CONTRACTION, since the
bounds are two-sided on purpose.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp57_expansion_term/stage3_fit.py --only X1 [--smoke]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP54 = ROOT / "experiments/exp54_unpinned_amplitude"
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink", EXP54, HERE):
    sys.path.insert(0, str(p))

import expand as X                                       # noqa: E402
import fit as F                                          # noqa: E402
import model as M                                        # noqa: E402
import stage0_cost as S0                                 # noqa: E402
import stage33 as S33                                    # noqa: E402
import stage35_time_law as S35                           # noqa: E402
import stage37_size_epoch as S37                         # noqa: E402

BASE = "gompertz_log-E2-S2"
LOSS_INCUMBENT = 1.874253
FITDIR = HERE / "outputs" / "stage3_fits"
#: where the extra starts put the expansion parameter. Two magnitudes of
#: expansion and one of contraction; see the module docstring for why a start
#: at exactly zero cannot search this axis on its own.
A_STARTS = (0.5, 2.0, -0.3)
RULE = "=" * 96


def assert_nests(pr, sp, theta_base, l0, tol=1e-9):
    """The nested start must reproduce the incumbent's loss, or stop."""
    th = sp.nest(theta_base)
    lx = pr.loss(th)
    print(f"    nested start: {dict(zip(sp.x_names, th[sp.base.n_theta:]))} "
          f"-> loss {lx:.12f} against the incumbent's {l0:.12f} "
          f"(d = {abs(lx - l0):.2e})")
    if abs(lx - l0) > tol:
        raise SystemExit(
            f"REFUSING TO RUN: {sp.label}'s nested start does not reproduce "
            f"the incumbent. The lift is wrong, so 'the enrichment did not "
            f"help' would be unfalsifiable.")
    return th


def starts_for(sp, theta_base, n_jitter=1, seed=7):
    """Nested first, then explicit expansion values, then a jittered base."""
    lo, hi = np.array(sp.bounds()).T
    nb = sp.base.n_theta
    out = [sp.nest(theta_base)]
    for a in A_STARTS:
        th = sp.nest(theta_base).copy()
        th[nb] = a
        if sp.law == "X0":                       # alpha lives on a different scale
            th[nb] = {0.5: 0.15, 2.0: 0.5, -0.3: -0.15}[a]
        out.append(np.clip(th, lo, hi))
    if sp.law == "X2":                           # the shape axis needs its own
        for a, p in ((1.0, 0.4), (1.0, 3.0)):
            th = sp.nest(theta_base).copy()
            th[nb], th[nb + 1] = a, p
            out.append(np.clip(th, lo, hi))
    if sp.law == "X3":
        # The core radius is the parameter the whole law turns on and it spans
        # two and a half decades, so it gets its own starts rather than being
        # left to Nelder-Mead's 5% simplex from a single value: 2, 10, 50 and
        # 200 kpc, each with a moderate expansion.
        for a, lrc in ((1.0, 0.3), (1.0, 1.7), (1.0, 2.3), (3.0, 0.7)):
            th = sp.nest(theta_base).copy()
            th[nb], th[nb + 1] = a, lrc
            out.append(np.clip(th, lo, hi))
    rng = np.random.default_rng(seed)
    for _ in range(n_jitter):
        th = sp.nest(theta_base).copy()
        th[:nb] = th[:nb] + rng.normal(0, 0.08, nb) * np.maximum(
            np.abs(th[:nb]), 0.3)
        th[nb] = 1.0
        out.append(np.clip(th, lo, hi))
    return out


class FrozenProblem:
    """The problem with one named parameter held fixed — gate G4's machinery.

    exp53 measured that the first model's transport SUBSTITUTED for a heavy
    deposit tail, and exp54 Stage 3.9 measured that the deposit shape `c` is the
    only one of the seven parameters the others cannot absorb. Adding an
    expansion term is the most likely thing to destroy that, because both
    control how far a deposit's mass reaches. Freezing `c` and refitting says
    how much of the expansion term's gain is really a change in deposit shape.

    Presents the same interface `fit.fit` needs, exactly as Stage 3.9's
    `FixedProblem` did, so the same optimiser runs on both.
    """

    class _Spec:
        def __init__(self, b):
            self._b = b

        def bounds(self):
            return self._b

    def __init__(self, problem, names, frozen):
        self.p = problem
        self.j = [names.index(k) for k in frozen]
        self.v = [float(frozen[k]) for k in frozen]
        full = list(problem.spec.bounds())
        self.spec = FrozenProblem._Spec(
            [b for k, b in enumerate(full) if k not in self.j])
        self.n_eval = 0

    def lift(self, x):
        out = np.asarray(x, float)
        for j, v in sorted(zip(self.j, self.v)):
            out = np.insert(out, j, v)
        return out

    def drop(self, x7):
        return np.delete(np.asarray(x7, float), self.j)

    def loss(self, x):
        self.n_eval += 1
        return self.p.loss(self.lift(x))


def run(law, smoke=False, freeze=None):
    print(f"{RULE}\nexp57 STAGE 3 — fitting the expansion law {law}\n{RULE}\n")
    recs, data, mask, lmh, base, theta, slow = S0.build(smoke)
    l0 = slow.loss(theta)
    print(f"  sample   : {len(recs)} galaxies, {int(mask.sum())} galaxy-epochs")
    print(f"  incumbent: {BASE}, loss {l0:.6f}")
    if not smoke and abs(l0 - LOSS_INCUMBENT) > 1e-5:
        raise SystemExit(f"REFUSING TO RUN: incumbent loss is {l0:.6f}")

    sp = X.XSpec(base, law)
    pr = X.ExpandingProblem(sp, recs, data, mask, epochs=(0, 1, 2, 3, 4))
    label = sp.label + (f"+frozen_{'_'.join(freeze)}" if freeze else "")
    frozen = ({k: float(theta[sp.theta_names.index(k)]) for k in freeze}
              if freeze else None)
    if frozen:
        print(f"  GATE G4  : {', '.join(f'{k} FROZEN at {v:+.4f}' for k, v in frozen.items())}")
    print(f"  fitting  : {label}, "
          f"{sp.n_theta - (len(freeze) if freeze else 0)} free parameters "
          f"({', '.join(sp.theta_names)})")
    print(f"  bounds   : " + "  ".join(
        f"{n}=({a:g},{b:g})" for n, (a, b) in zip(sp.x_names,
                                                  sp.bounds()[base.n_theta:])))
    assert_nests(pr, sp, theta, l0)

    st = starts_for(sp, theta, n_jitter=(0 if smoke else 1))
    if smoke:
        st = st[:2]
    print(f"  {len(st)} starts; expansion values "
          + ", ".join(f"{s[base.n_theta]:+.3f}" for s in st) + "\n", flush=True)

    t0 = time.time()
    best_th, best_l, per_start = None, np.inf, []
    fp = FrozenProblem(pr, sp.theta_names, frozen) if frozen else None
    for i, p0 in enumerate(st):
        ts = time.time()
        if fp is None:
            th, l = F.fit(pr, starts=[p0],
                          maxiter=(400 if smoke else 600 * sp.n_theta),
                          verbose=False)
        else:
            x, l = F.fit(fp, starts=[fp.drop(p0)],
                         maxiter=(400 if smoke else 600 * sp.n_theta),
                         verbose=False)
            th = fp.lift(x)
        per_start.append((float(l), th.copy()))
        flag = ""
        if l < best_l:
            best_th, best_l, flag = th, l, "   <- best so far"
        print(f"    start {i} (expansion {p0[base.n_theta]:+.3f}): loss "
              f"{l:.6f}, {', '.join(f'{n}={v:+.4f}' for n, v in zip(sp.x_names, th[base.n_theta:]))}"
              f"  [{(time.time() - ts) / 60:.1f} min]{flag}", flush=True)

    spread = max(l for l, _ in per_start) - min(l for l, _ in per_start)
    print(f"\n  best loss {best_l:.6f} against the incumbent's {l0:.6f}: "
          f"{100 * (1 - best_l / l0):+.2f}%")
    print(f"  start-to-start spread {spread:.2e}"
          + ("   <- ROUGH surface, a single start would have been unsafe"
             if spread > 1e-4 else ""))
    print(f"  fitted expansion: "
          + ", ".join(f"{n} = {v:+.4f}"
                      for n, v in zip(sp.x_names, best_th[base.n_theta:])))
    lo, hi = np.array(sp.bounds()).T
    at_b = [n for i, n in enumerate(sp.theta_names)
            if abs(best_th[i] - lo[i]) < 1e-3 or abs(best_th[i] - hi[i]) < 1e-3]
    if at_b:
        print(f"  AT A BOUND: {', '.join(at_b)} — report as a bound, not as a "
              f"fitted value")
    if law == "X1":
        A = float(best_th[base.n_theta])
        print(f"  IN WORDS: the oldest deposits end up {1 + A:.3f} times "
              f"larger than they were laid down"
              + ("  (i.e. they CONTRACT)" if A < 0 else ""))

    sa, sf = pr.per_epoch(best_th)
    print(f"  profile-shape error {S35.shape_pct(sf):.3f}% "
          f"(incumbent 13.787%), score_A {pr.scores(best_th)[0]:.4f} "
          f"(incumbent 1.0558)")
    print(f"  total {(time.time() - t0) / 60:.1f} min")

    if not smoke:
        FITDIR.mkdir(parents=True, exist_ok=True)
        np.savez(FITDIR / f"{label}.npz", label=label, law=law,
                 frozen=np.array(sorted(frozen)) if frozen else np.array([]),
                 theta=best_th, loss=best_l, loss_incumbent=l0,
                 theta_incumbent=theta,
                 names=np.array(sp.theta_names), n_starts=len(st),
                 start_losses=np.array([l for l, _ in per_start]),
                 start_thetas=np.array([t for _, t in per_start]),
                 sA=sa, sF=sf, shape_pct=S35.shape_pct(sf))
        print(f"  wrote {FITDIR.name}/{label}.npz")
    return best_th, best_l


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    law = (sys.argv[sys.argv.index("--only") + 1]
           if "--only" in sys.argv else "X1")
    frz = (sys.argv[sys.argv.index("--freeze") + 1].split(",")
           if "--freeze" in sys.argv else None)
    run(law, smoke, frz)
