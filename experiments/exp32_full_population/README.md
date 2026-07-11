# exp32 — the full-population emulator (n=2397, logM* 10.66–12.36)

> **RESULT (2026-07-11, steps 1–3).** The universal-theta capacity limit (~30%
> held-out worst-radius, zero generalization gap) holds across the ENTIRE mass
> range; mass-conditioning buys 1 point (continuous logMh-slope = binning);
> the per-galaxy individuality lives in a degenerate width-growth/efficiency
> subspace, NOT the size normalization; and **epoch-matched history features
> overturn the "MAH stops mattering at high z" reading** — direct-epoch is the
> best per-quantity regression at every tier, while every regression family
> fails the observational planes that the transport family reproduces 2–3x
> better. The two model families are now cleanly complementary.

## Step 1 — foundation
- `dataset.py`: population cache, n=2397 (use + finite CoG + exp29 crossmatch
  quality flag + valid real/DiffMAH MAH + valid 5-epoch CoG; 100% of spot-checked
  eligibles pass). **Discovery: the historical "n=45" was a stratified
  every-41st subsample of the mass ranking (rows 0,41,...,1787), not the top-45**
  — why its results extrapolated so well.
- `theta_atlas.py`: per-galaxy 7-param fits for all galaxies, both configs
  (diffmah 37 min / real 24 min, 6 workers). **The ~10% per-galaxy floor holds
  over the whole mass range** (diffmah median per epoch 6.9–11.3%; floor rises
  mildly with mass: epoch-avg 9.7% at logM*~10.7 to 13.7% at the massive end).

## Step 2 — the mass structure of theta (`universal_mass.py`, diffmah primary)
10-fold mass-stratified CV, n=2397; epoch-avg max|rel|, per-galaxy median:

| model | in-sample | held-out | note |
|---|---|---|---|
| per-galaxy floor | 12.4% | — | atlas reference |
| global universal (7p) | 30.5% | 30.4% | ZERO gap — capacity, not overfitting |
| mass-binned (4x7p, logMh quartiles) | 27.9%* | 29.4% | *mean of bin losses |
| continuous theta(logMh) slope (14p) | 29.0% | 29.4% | = binning, half the params, differentiable |
| global universal, real-MAH config | 35.2% | 35.2% | DiffMAH stays ~5 points better |

- Mass-conditioning helps mostly BELOW logM*~11.3 (held-out Q1 32.5→29.7%);
  the massive quartile barely moves (35.6→34.7%). The fitted theta shifts
  regime around logMh~13.4 (panel B of the figure).
- **Anatomy** (free ONE component per galaxy around the global theta; median
  fraction of the galaxy's gap-to-floor closed): log_s0 **0.2%**, log_alpha 15%,
  and **g / q / b_early / b_late / z_c each 35–40%**. The individuality is NOT
  the width normalization (size) and NOT one knob — it is a degenerate
  ~2–3-dimensional subspace of width-GROWTH x efficiency-SHAPE directions.
  Step 4's stochastic layer should draw from that subspace.

## Step 3 — the honest scoreboard vs mass (`scoreboard.py`)
All held-out, halo-only; transport = CV universal-theta CoGs x LOO SHMR;
regressions = exact hat-matrix LOO per radius/epoch. Epoch-avg over the full
sample (per-quartile numbers in the output):

| tier | transport | transport-bins | transport-real | logmh-only | direct | direct-epoch |
|---|---|---|---|---|---|---|
| kpc apertures [dex] | 0.149 | 0.146 | 0.162 | 0.163 | 0.150 | **0.139** |
| Re envelopes [dex] | 0.259 | 0.246 | 0.359 | 0.303 | 0.293 | **0.251** |
| max\|rel\| R>5 kpc | 30.8% | 29.5% | 35.6% | 30.5% | 28.3% | **26.9%** |
| plane fidelity \|Δscatter\| [dex] | 0.209 | 0.193 | **0.169** | 0.429 | 0.417 | 0.542 |

1. **The epoch-matched features settle the MAH question**: `direct-epoch`
   (logMh(z_k), t50(z_k)/t(z_k), Mh(t_k/2)/Mh(t_k) — history up to z_k only,
   no z=0.4-anchored summaries, no c200c) is the best regression at EVERY tier
   and mass quartile. The exp31 "MAH adds nothing at high z" was feature
   misalignment, as suspected — the assembly history carries per-quantity
   information at all epochs when summarized relative to the target epoch.
2. **The per-quantity/distribution trade-off is now sharp**: the better a
   regression's per-galaxy error, the worse its population distribution
   (direct-epoch has the WORST plane fidelity, 0.542 — strongest
   regression-to-the-mean). The transport family reproduces the observational
   planes 2–3x more faithfully at slightly worse per-quantity scatter. No
   current model does both; that is exactly step 4's job.
3. **Config check down-mass**: DiffMAH input beats the real MAH at every tier
   and quartile except plane fidelity (real 0.169 — its bursty diversity
   propagates); the differentiable configuration stands.
4. Everyone degrades on the massive quartile (apertures 0.19–0.21 dex) — the
   massive end is intrinsically the hardest, also at the per-galaxy floor.

## Decisions / next (step 4–5)
- Population model: **continuous theta(logMh) slope, DiffMAH input** (29.4%,
  differentiable, 14 params).
- Step 4 (stochastic layer) has a measured target: close the plane-fidelity
  gap (0.19 -> ~0.07-0.10, the truth-scatter scale) by drawing correlated
  theta-deviations in the anatomy subspace; judged by tier 2b + exp07
  CRPS/calibration.
- Open question for step 4/5: can the transport structure absorb the
  direct-epoch information (epoch-matched features conditioning the efficiency
  f(z) per galaxy) without losing plane fidelity?

## Files
- `dataset.py` (cache builder + demo), `theta_atlas.py` (pooled per-galaxy
  fits + demo), `universal_mass.py` (global/bins/slope/anatomy/cv/report
  subcommands + demo), `scoreboard.py` (LOO regressions + tiered QA + demo).

Run: `PYTHONPATH=. uv run python experiments/exp32_full_population/<script>.py --help-less; see module docstrings`
