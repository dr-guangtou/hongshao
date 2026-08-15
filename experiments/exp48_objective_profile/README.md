# exp48 — the objective, the metric, and the deposit profile

Three axes that exp47 left entangled, separated and measured. **One change
survives: the objective. The deposit profile does not matter.**

## Context

exp47 established that the kernel's plane failure and its poor central
masses are ONE defect, living on COMPACT galaxies at every epoch, and that
re-weighting the objective by a compact/extended split largely fixes it.
That fix was rejected on portability — it needs an R50 threshold and
TNG300's population fractions, so it cannot be retrained on another
simulation. exp48 pursues the same gain through changes that carry no
simulation-specific calibration.

## Step A — qa tier 2e (`hongshao/qa.py`, now STANDARD)

Tiers 1-2 report a median bias and a dex scatter: two moments of a
distribution that can differ in shape, tails or skew while matching both.
Tier 2e compares the DISTRIBUTIONS, per quantity per epoch, via their
CDFs — KS (largest vertical gap) and 1-Wasserstein (mean horizontal gap,
in dex), each against the truth's own split-half floor.

It immediately isolated a defect the mass tables hide: for the adopted
kernel the CUMULATIVE masses sit at the floor (M(<30): 0.4-0.9 at every
epoch) while the ANNULI are 4.4-4.8 at z >= 1.5. The model gets the totals
right and the radial distribution wrong.

Documented and asserted: tier 2e is PAIRING-BLIND like `plane_energy`, and
its ratio is unstable when the truth distribution is very narrow.

## Step B — the objective study (20 objectives)

Five portable axes: quantity (CoG / density), residual (fractional /
absolute / log), radial aggregation (RMS / mean-abs / max), radial
weighting, galaxy aggregation (mean / median / CVaR).

**Winner: compare SURFACE DENSITY profiles with a LOG residual.**

| objective | mean E/floor z<=1.5 | M5 compact | differential |
|---|---|---|---|
| **density x log** | **1.95** | **-31.4%** | 0.44/0.14 |
| cog x abs x absmean | 2.42 | -37.7% | 0.43/0.14 |
| *baseline (adopted)* | *2.72* | *-43.0%* | 0.40/0.13 |
| density x **frac** | 14.37 | +88.7% | BROKEN |

Three findings that argument alone would have got wrong:

1. **Density only works in log.** With a fractional residual it is the
   WORST objective tested; with a log residual it is the best. The
   one-at-a-time screen would have discarded density permanently — the
   interaction is the result. (Fractional residuals diverge where Sigma
   is small in the outer bins; a log residual equalizes the dynamic range
   and compares shape at every radius on equal terms.)
2. **CVaR failed**, and cleanly. Optimizing the worst-fit 20% was the
   portable replacement for exp47's rebalancing; it scored BELOW baseline
   and made compact galaxies worse. Measured reason: the worst-fit
   galaxies have median R50 = 14.9 kpc against 11.1 for the rest. **The
   badly-fit tail is the BIGGEST galaxies, not the compact ones.**
3. The absolute-linear residual improved the INNER masses and was the
   best single-axis change — the direction the user predicted and the
   plan's author did not.

## Step C — the deposit profile does NOT matter

Six families fitted inside the FULL kernel (conditioning, migration,
M(<500) normalization), each a one-line swap of the `cog(rc)` closure so
the profile is genuinely the only variable. Reproducibility pinned:
`moffat::current` returned 0.15362 against the screen's 0.15362, three
times, through a different code path.

**Loss ranking (winner objective):** richards 0.18282 < loglogistic
0.18289 < sersic_r50 0.18294 < cuspy 0.18379 < **moffat 0.18514** <
gompertz_log 0.18642.

**Judge ranking — NEARLY THE REVERSE:**

| candidate | mean E/floor | M5 compact | M5 extended | differential |
|---|---|---|---|---|
| **moffat (incumbent)** | **1.95** | -31.4% | **+3.9%** | 0.44/0.14 |
| gompertz_log | 2.05 | **-27.7%** | +7.6% | **0.39/0.13** |
| cuspy | 2.09 | -31.6% | +4.4% | 0.47/0.15 |
| loglogistic | 2.28 | -31.8% | +8.8% | 0.45/0.15 |
| richards | 2.29 | -31.4% | +10.4% | 0.44/0.14 |
| sersic_r50 | 2.44 | -29.0% | +14.2% | 0.42/0.14 |

The extra profile freedom is spent OVER-FILLING extended galaxies' centres
(+3.9% -> +14.2% down the ranking) while the compact deficit does not move
(-31.4% to -31.8%). **Had we ranked by loss, as exp38's shootout did, we
would have adopted richards or loglogistic and made the planes worse.**

- **The objective is the lever**: compact M5 -43.6% -> -31.4% from the
  objective alone; at best -31.4% -> -27.7% from any profile on top.
- **Moffat is vindicated.** exp38 chose it with a metric we showed was
  blind to profile structure, and it still wins on observables.
- **The cuspy hypothesis is dead.** It fits alpha=+0.814 under the winning
  objective and still scores worse, with no gain in compact central mass.
- **The objective REVERSES the profile ranking**: sersic_r50 is worst
  under the current objective (0.15767 vs moffat 0.15362) and tied-best
  under the winner. The two questions were never independent.
- **Descriptive quality does not transfer to primitive quality**:
  `gompertz_log` won the per-galaxy CoG benchmark (`gompertz.py`: 2.31% vs
  moffat's 3.00% at z=2) and is LAST on loss as a deposit — yet has the
  best compact central mass and the most comfortable gate. Worth a
  revisit; not adoptable on this evidence.

## Step D — held-out cross-validation: THE GAIN IS REAL

5 folds, every galaxy predicted out-of-sample, both objectives
cross-validated the same way:

| | z=0.4 | mean z<=1.5 | M5 compact | M5 extended | differential |
|---|---|---|---|---|---|
| baseline, held-out | 2.1 | 2.71 | -43.1% | -11.0% | 0.40/0.13 |
| **density-log, held-out** | **1.3** | **1.96** | **-30.9%** | +3.2% | 0.45/0.14 |
| *density-log, in-sample* | *1.3* | *1.95* | *-31.4%* | *+3.9%* | *0.44/0.14* |

Held-out and in-sample agree to 0.01 in plane score. No generalization
gap, consistent with exp32's precedent for population-shared thetas.

**THE COST, stated plainly: the differential gate moves 0.40/0.13 ->
0.45/0.14 against a data value of 0.37/0.11.** It passes, but 0.45 is at
the very top of the acceptable band where the baseline had margin. Given
exp38's history of components that bought inner mass and broke this test,
this is a genuine trade, not a free win.

## Recommendation (NOT adopted — the user's call)

Keep the adopted 1ch-mof profile exactly as it is. Change only the
objective, to log residuals on the surface-density profile. No new
parameters, no new functional form, no simulation-specific weighting.

Open: whether the differential margin is acceptable, and whether this
becomes the default or a documented option.

## Figures

`figures/qa_*_exp48_{adopted_objective,density_log}.*` — the full standard
set for both, including the new tier 2e CDF panels;
`figures/exp48_profiles` — the candidate deposit profiles at fixed R50,
which is also the correction to an earlier overclaim: at fixed R50 the
cusp buys 1.32x at alpha=1.5 (non-monotonically), not the large lever
first reported at fixed scale radius.

## Run

```bash
export HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof
PYTHONPATH=. uv run python experiments/exp48_objective_profile/objective.py {demo|screen|factorial|report}
PYTHONPATH=. uv run python experiments/exp48_objective_profile/shootout.py {demo|fit|report}
PYTHONPATH=. uv run python experiments/exp48_objective_profile/cv.py {demo|run}
PYTHONPATH=. uv run python experiments/exp48_objective_profile/gompertz.py {demo|run}
PYTHONPATH=. uv run python experiments/exp48_objective_profile/profile_figure.py
```

**Serial by default.** The pooled path hangs after a few fits in one
process — workers die, the parent blocks in `pool.map` at ~0% CPU, and the
workers report BrokenPipeError on multiprocessing's error path. It cost
hours twice before a CPU-based stall alarm caught the third occurrence in
15 minutes. `--pool` opts back in at your own risk.
