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

(pending)
