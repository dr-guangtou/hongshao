"""exp61 — single-epoch baselines, the parameter drift, and the z ≤ 1.0 scope.

The user's two directions after the exp59 verdict, verbatim: "1. independent
single-epoch fit to establish the model's performance baseline, and also
explore the evolution of the key free parameters with the redshift; 2. reduce
the redshift range. A well-motivated retreat. If hongshao could work well
under z <= 1.0, it is still a very good 1st step."

WHAT THIS MEASURES. `gompertz_log-E2-S2` (exp54's adopted seven-parameter
model) fitted SEVEN times on the same clean sample under the same
mh-complete + sane mask:

  * five INDEPENDENT single-epoch fits (the loss sees one epoch's
    galaxy-epochs only). Note a single-epoch fit still constrains the time
    parameters, because the efficiency and size laws act at DEPOSIT time and
    one epoch's profile integrates the whole deposit history — what it drops
    is the requirement that one theta serve five epochs at once.
  * one z <= 1.0 fit (epochs 0.4 / 0.7 / 1.0).
  * the shared incumbent (no fit — its per-epoch scores are the reference).

DELIVERABLES.
  1. The BASELINE table: per-epoch score_A / score_F for the shared
     incumbent, for each single-epoch fit at its own epoch, and for the
     z <= 1.0 fit at its three epochs (plus, for information only, that fit
     evaluated OUT of scope at z = 1.5 / 2.0). The single-epoch column is the
     model CLASS's per-epoch ceiling at shared-law-free timing; the gap to
     the shared row is what one-theta-for-five-epochs costs, epoch by epoch.
  2. The DRIFT table: the seven fitted parameters against redshift, with
     multi-start spread and at-bound flags. Read as an INDICATION of where
     the shared law bends, not as per-parameter measurements — exp54 Stage
     3.9 measured how strongly these parameters absorb each other, and no
     profile likelihood is run here.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp61_epoch_baselines/stage0_fits.py [--smoke]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP54 = ROOT / "experiments/exp54_unpinned_amplitude"
EXP57 = ROOT / "experiments/exp57_expansion_term"
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink", EXP54, EXP57):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import stage0_cost as S0                                 # noqa: E402
import stage35_time_law as S35                           # noqa: E402

ANCHOR_Z = (0.4, 0.7, 1.0, 1.5, 2.0)
N_STARTS = 4
SEED = 11
OUT = HERE / "outputs" / "stage0_fits.npz"
RULE = "=" * 96


def starts_for(theta, bounds, rng):
    """The incumbent, plus jittered copies of it (8% relative, the exp57
    convention). The incumbent start makes 'worse than shared' impossible for
    a converged fit, since the shared theta is admissible in every scope."""
    lo, hi = np.array(bounds).T
    out = [np.asarray(theta, float)]
    for _ in range(N_STARTS - 1):
        th = theta + rng.normal(0, 0.08, len(theta)) * np.maximum(
            np.abs(theta), 0.3)
        out.append(np.clip(th, lo, hi))
    return out


def fit_scope(name, epochs, base, recs, data, mask, theta, rng, smoke):
    pr = S35.StackedProblem(base, recs, data, mask, epochs=list(epochs))
    l_inc = pr.loss(theta)
    st = starts_for(theta, base.bounds(), rng)
    if smoke:
        st = st[:2]
    best_th, best_l, losses = None, np.inf, []
    t0 = time.time()
    for i, p0 in enumerate(st):
        th, l = F.fit(pr, starts=[p0],
                      maxiter=(300 if smoke else 600 * base.n_theta),
                      verbose=False)
        losses.append(float(l))
        if l < best_l:
            best_th, best_l = th, l
    sa, sf = pr.per_epoch(best_th)
    lo, hi = np.array(base.bounds()).T
    railed = [n for j, n in enumerate(base.theta_names)
              if abs(best_th[j] - lo[j]) < 1e-3 or abs(best_th[j] - hi[j]) < 1e-3]
    print(f"  {name:<10} loss {best_l:.6f} (incumbent theta gives {l_inc:.6f}"
          f", {100 * (1 - best_l / l_inc):+.2f}%), spread "
          f"{max(losses) - min(losses):.2e}, {time.time() - t0:.0f} s"
          + (f", AT BOUND: {','.join(railed)}" if railed else ""), flush=True)
    return dict(name=name, epochs=list(epochs), theta=best_th, loss=best_l,
                loss_at_incumbent=l_inc, start_losses=losses, sa=sa, sf=sf,
                railed=railed)


def main(smoke=False):
    print(f"{RULE}\nexp61 — single-epoch baselines and the z <= 1.0 scope"
          f"\n{RULE}\n")
    recs, data, mask, lmh, base, theta, slow = S0.build(smoke)
    print(f"  sample: {len(recs)} galaxies; shared incumbent loss "
          f"{slow.loss(theta):.6f}")
    sa0, sf0 = slow.per_epoch(theta)
    rng = np.random.default_rng(SEED)

    fits = []
    for k in range(5):
        fits.append(fit_scope(f"z={ANCHOR_Z[k]}", (k,), base, recs, data,
                              mask, theta, rng, smoke))
    fits.append(fit_scope("z<=1.0", (0, 1, 2), base, recs, data, mask, theta,
                          rng, smoke))
    # the z<=1.0 theta evaluated at ALL five epochs, out-of-scope included
    pr_all = S35.StackedProblem(base, recs, data, mask, epochs=[0, 1, 2, 3, 4])
    sa_r, sf_r = pr_all.per_epoch(fits[-1]["theta"])

    print(f"\n{RULE}\n  THE BASELINE TABLE — per-epoch score_A / score_F "
          f"(lower is better; ~1 = as good as\n  the halo-mass regression "
          f"benchmark refitted at that epoch)\n{RULE}")
    print(f"  {'model':<26}" + "".join(f"{f'z={z}':>14}" for z in ANCHOR_Z))
    print(f"  {'shared incumbent':<26}" + "".join(
        f"{sa0[j]:>7.3f}/{sf0[j]:<6.3f}" for j in range(5)))
    row = []
    for j in range(5):
        row.append(f"{fits[j]['sa'][0]:>7.3f}/{fits[j]['sf'][0]:<6.3f}")
    print(f"  {'single-epoch (each own)':<26}" + "".join(row))
    print(f"  {'z<=1.0 refit':<26}" + "".join(
        f"{sa_r[j]:>7.3f}/{sf_r[j]:<6.3f}"
        + ("*" if j >= 3 else "") for j in range(5)))
    print(f"  (* = out of the z<=1.0 fit's scope, evaluated for information "
          f"only)")

    print(f"\n{RULE}\n  THE DRIFT TABLE — fitted parameters against redshift "
          f"(single-epoch fits;\n  indicative, not per-parameter "
          f"measurements)\n{RULE}")
    print(f"  {'parameter':<12}{'shared':>10}"
          + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z) + f"{'z<=1.0':>10}")
    for j, n in enumerate(base.theta_names):
        print(f"  {n:<12}{theta[j]:>+10.4f}"
              + "".join(f"{fits[k]['theta'][j]:>+10.4f}" for k in range(5))
              + f"{fits[5]['theta'][j]:>+10.4f}")

    if not smoke:
        OUT.parent.mkdir(exist_ok=True)
        np.savez(OUT, names=np.array(base.theta_names),
                 theta_shared=theta, sa_shared=sa0, sf_shared=sf0,
                 sa_scope=sa_r, sf_scope=sf_r,
                 **{f"fit_{f['name'].replace('<=', 'le')}_{k}": v
                    for f in fits for k, v in f.items()
                    if k in ("theta", "loss", "start_losses", "sa", "sf")})
        print(f"\n  saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv[1:])
