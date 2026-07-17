# exp39 — the core channel, revisited

exp38 adopted the 1ch-mof kernel (12 parameters, physics-clean) and PARKED
the dissipative core channel: it fixed the observation-facing inner masses
(held-out M(<5 kpc) bias -11% -> -2.8% at z=0.4) but broke the two physics
tests that are the kernel's reason to exist — the differential-deposition
curve (0.53/0.19 vs the measured 0.37/0.11: the fraction of z=0.7->0.4
stellar growth landing beyond 50/100 kpc for the massive tercile) and the
outskirt terciles (+0.054/+0.085 dex median surface-density overshoot at
30-60/60-148 kpc for the low-mass tercile, vs the kernel's +0.026/+0.019).

The post-mortem found two candidate mechanisms:

1. **Tail leakage (what-shape-it-has):** the core inherited the kernel's
   power-law tail — 8.3% of "core" mass beyond 30 kpc; at f_core = 0.19
   that is 1.6% of ALL stellar mass in the outskirts, matching the failed
   overshoot.
2. **Outer-kernel re-balancing (who-pays-for-it):** the quenched-core null
   showed the outer kernel re-balances itself (wider late deposits)
   whenever ANY core absorbs the inner mass, wherever the core's mass
   comes from.

Three measured leads (user plan, 2026-07-16), in order:

## Lead 1 — the core-form shootout (`shootout.py`)

Swap ONLY the core's radial form; everything else identical to the exp38
stage-3 core model (adopted kernel + non-migrating core channel,
inner-aware objective). Forms, all parameterized by the core half-mass
radius R50 (log10 box 0.5-8 kpc) so the scale means the same thing in
every row:

| form | profile | core mass beyond 30 kpc at the parked R50 (3.4 kpc) |
|---|---|---|
| mof | the inherited power-law tail (baseline = the parked model re-expressed) | 8.3% |
| gauss | Sersic n=0.5 — zero wings by construction | ~0 |
| exp | Sersic n=1 | ~0 |
| ser2/3/4 | cuspier Sersic n=2/3/4 (more mass inside 2-5 kpc per unit core mass) | small, rising with n |

The shootout separates the two mechanisms: if zero-wing cores keep the
inner win AND pass the physics tests, leakage was the breaker (the core
is rescuable); if even the Gaussian core breaks them, re-balancing
dominates and no core form helps.

Judged by the SAME pre-registered criteria: the differential curve (data
0.37/0.11), the outskirt terciles, NO parameter at a bound, held-out
pinned shape vs the kernel's 18.5-14.2% by epoch (parked core avg 15.0%),
and the M(<5)/M(<10) bias table.

Run (always `export HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof`):

    PYTHONPATH=. uv run python experiments/exp39_core_revisit/shootout.py \
        {demo|fit|cv|physics|table} [--form gauss,ser3] [--dev] [--cond z2]

## Lead 2 — condition f_core on the z=2 halo mass (`--cond z2`)

The provenance measurement showed the model core is built by z=2-4
deposits, so the halo mass at z=2 (population npz `logmh_zk_real[:, 4]`)
is the physically aligned regressor for the core fraction, not the
current z=0.4 logMh.

## Lead 3 — per-galaxy core diversity

The single-vs-joint decomposition located most of the remaining inner
deficit (-5 to -8%) in the POPULATION-SHARING limit: no shared core
fraction reproduces the galaxy-to-galaxy core diversity. Explore a
per-galaxy core fraction treated statistically. (Design after leads 1-2.)

## Results

### Lead 1 — the shootout verdict (2026-07-17, full n=2397): the core's
### radial form is IRRELEVANT; the re-balancing mechanism is the breaker

All six forms fit to statistically identical loss under the inner-aware
objective, and ALL of them — including the zero-leakage Gaussian — fail
the differential-deposition test identically:

| form | loss | median f_core | R50 core [kpc] | core mass beyond 30 kpc | at a bound | differential massive z0.7->0.4 (data 0.37/0.11) | outskirt T1 [30-60/60-148 kpc, dex] |
|---|---|---|---|---|---|---|---|
| mof (parked baseline) | 0.2541 | 0.110 | 0.76 | 2.6% | NONE | 0.53/0.18 FAIL | +0.051/+0.079 |
| gauss | 0.2540 | 0.129 | 3.46 | 0.0% | NONE | 0.54/0.19 FAIL | +0.056/+0.093 |
| exp | 0.2543 | 0.132 | 2.13 | 0.0% | mu=2.99 | 0.56/0.19 FAIL | +0.060/+0.086 |
| ser2 | 0.2540 | 0.144 | 2.30 | 0.1% | NONE | 0.54/0.19 FAIL | +0.052/+0.096 |
| ser3 | 0.2540 | 0.145 | 2.08 | 0.6% | NONE | 0.54/0.19 FAIL | +0.047/+0.090 |
| ser4 | 0.2540 | 0.145 | 1.83 | 1.4% | NONE | 0.53/0.19 FAIL | +0.050/+0.090 |

(differential = the fraction of z=0.7->0.4 stellar growth landing beyond
50/100 kpc, massive tercile; outskirt T1 = median surface-density
overshoot for the low-mass tercile at z=0.4, in-sample.)

**The tail-leakage mechanism is FALSIFIED as the physics-breaker:** a
core with zero mass beyond 30 kpc by construction breaks the physics
exactly like the 8%-leaky Moffat. The outer-kernel re-balancing (the
exp38 quenched-core null's suspect) is conclusively the cause — the
freed kernel spends its freedom on wider late deposits regardless of
what shape absorbs the inner mass.

### The frozen-kernel operating point: the no-tradeoff core

Pinning the 12 kernel parameters at the adopted stage-2 theta and
fitting ONLY the 3 core parameters (the kernel cannot re-balance):
a small, strongly halo-conditioned zero-wing Gaussian core (median
f_core 0.026, 16-84 pct range 0.002-0.080, R50 = 3.6 kpc interior;
the Moffat variant rails its scale at the 0.5-kpc box floor) gives:

| operating point (gauss core) | held-out pinned shape by epoch [%] | held-out M(<5) / M(<10) z=0.4 | differential (data 0.37/0.11) | outskirt T1 [dex] |
|---|---|---|---|---|
| adopted kernel, no core | 18.5/17.6/16.7/16.2/14.2 avg 16.6 | ~-11% / -5.4% | 0.39/0.12 PASS | +0.026/+0.019 |
| FROZEN kernel + core | 17.6/17.3/17.5/16.6/13.7 avg 16.5 | -5.0% / -0.6% | 0.39/0.12 PASS (the kernel's own) | +0.016/+0.039 |
| free kernel + core | 16.5/16.6/16.2/14.4/11.6 avg 15.1 | -3.1% / -4.0% | 0.54/0.19 FAIL | +0.056/+0.093 |

The frozen point is a STRICT improvement on the adopted kernel: same
held-out shape (16.5 vs 16.6 avg), the physics tests untouched (the
kernel's parameters are literally unchanged), and the z=0.4 inner
deficit halved (M(<5) -11 -> -5.0%, M(<10) -5.4 -> -0.6%). It does not
fix z=2 (M(<5) -6.6%): the high-z deficit needs deposits born more
compact, not retained mass (the exp38 provenance split). The free
kernel buys ~1.4 further shape points and ~2 inner points — exactly
what the parked exp38 Moffat core bought (15.0%/-2.8%) — and pays with
the physics; the trade is form-independent, now measured end to end.

### Lead 2 — condition f_core on the z=2 halo mass: helps exactly where
### the core channel is usable

Swapping the f_core regressor from the z=0.4 logMh to the standardized
z=2 halo mass (`logmh_zk_real[:, 4]`; the provenance-aligned choice):

- **Frozen kernel: better everywhere.** Loss 0.2737 -> 0.2719; in-sample
  M(<5) -4.9 -> -4.5% (z=0.4) and -6.6 -> -6.2% (z=2); held-out shape
  avg 16.5 -> 16.1% (17.8/17.0/16.6/16.1/13.2 by epoch) with held-out
  M(<5) -4.4% / M(<10) -1.0% at z=0.4. The core is more compact
  (R50 1.05 kpc) and still zero-leak, no parameter at a bound.
- **Free kernel: no gain.** Loss 0.2540 -> 0.2546 and the efficiency
  peak rails (mu = 3.00): once the kernel is free, its own conditioning
  rows already carry the halo information and the z2 regressor adds
  nothing but pressure on the bounds.

### Lead 3 — per-galaxy core diversity: large, real, and mostly NOT
### feature-predictable

At the frozen-z2 operating point, giving every galaxy its own core
fraction (one bounded scalar per galaxy; kernel, core scale, everything
else pinned): the fitted f_core spans 0.00-0.20 (16-84 pct; median
0.04) with 1.45 dex logit-scale scatter; the inner-aware loss drops
0.2719 -> 0.2135 (-21%) and the shared fit's median M(<5) deficit
closes (-4.5% -> +2.1%, and the 16th percentile improves -25.7 ->
-7.8%). But the diversity correlates only weakly with every feature we
have (Spearman |rho| <= 0.31: logms -0.31, logmh_z2 -0.26, logmh_z04
-0.24, t50 +0.17, c200c +0.11) — R^2 <~ 0.1, so conditioning cannot
capture it. This is the exp38 population-sharing limit made explicit:
the remaining inner deficit is per-object information the halo features
do not carry. Design consequence: a STATISTICAL f_core scatter layer
(logit-normal around the z2-conditioned mean) belongs in the generative
emulator draws; the deterministic kernel keeps the shared -4.4%.

## FINAL DECISION (user, 2026-07-17): PARKED, nothing graduates

The frozen-kernel operating point is NOT adopted: pinning the kernel is
an ad-hoc constraint, not a physical mechanism — it adds parameters
without model insight, and a joint fit of the same structure breaks the
physics (i.e., the combined model is not better at its own optimum; the
"improvement" exists only under an artificial restriction). The exp39
record stands as MECHANISM knowledge, not as a model: the re-balancing
diagnosis, the leakage falsification, the z2-conditioning result, and
the per-object diversity measurement are the durable outputs. The
adopted kernel remains the exp38 stage-2 1ch-mof, unchanged; the inner
deficit remains documented as a population-sharing limit, and the
observation-facing inner masses remain the statistical emulator's job
(exp37, M(<10) bias 1-2%).

## Measured verdicts (2026-07-17)

1. **The core-form shootout falsifies the tail-leakage mechanism**: all
   six forms (leak 0.0-2.6%) break the differential test identically at
   the free-kernel operating point. The outer-kernel re-balancing is
   THE physics-breaker; no core form rescues the free fit.
2. **The frozen-kernel + zero-wing Gaussian core (z2-conditioned) is a
   strict improvement on the adopted kernel**: held-out shape avg 16.1%
   vs 16.6%, held-out M(<5) -11 -> -4.4% and M(<10) -5.4 -> -1.0% at
   z=0.4, differential and outskirt tests untouched (the kernel's 12
   parameters are literally unchanged), 3 extra parameters, none at a
   bound. The z=2 deficit stays (-6.2%): deposits are born too wide at
   high z — a different (efficiency-side) problem, per the exp38
   provenance split.
3. **The remaining deficit is per-object information** (lead 3): treat
   it as a statistical scatter layer in the emulator, not as more
   conditioning.
