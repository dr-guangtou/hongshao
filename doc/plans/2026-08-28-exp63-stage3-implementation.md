# exp63 Stage 3 Implementation Plan — noise inside the deposition process

> **For agentic workers:** execute inline in the primary session. Steps use checkbox (`- [ ]`) syntax. Depends on Stage 2's fitted mean (`outputs/stage2_fit.npz[theta_best]`); the machinery is built and self-checked on the nested incumbent first.

**Goal:** Make the two-channel deposition model generative by putting three physically named noise sources *inside the history* — lumpy extended-channel accretion, a correlated efficiency history, per-galaxy size offsets — with analytic moments; sweep each source's leverage on the population widths by evaluation before any fit; fit the surviving noise parameters to held-out population statistics with a proper scoring rule, the mean frozen; then propagate the same draws to the earlier epochs as predictions (coherence, decliners).

**Architecture:** `process3.py` holds `Noise3` (the five noise parameters), `draw(spec2, theta, noise, curves, R, rng, n_draw)` returning `(S, n, 5, nR)` coherent multi-epoch draws, and `moments(...)` returning the analytic mean and the radius covariance of the lumpy-accretion source (Campbell's theorem) as the check on the draws. `stage3_process.py` holds the leverage sweep, the held-out energy-score fit, the gates G3a–G3b, the predictions and the figures.

**Tech Stack:** `numpy` Poisson and Cholesky draws on the shared node grid; `hongshao.qa.plane_energy`, `qa.size_planes`, `qa.mass_cdf`-style tier-2e (via `qa.evaluate(draw_cogs=...)` where possible), `scipy.optimize.minimize` Nelder-Mead with common random numbers.

**Spec:** science plan §2.2, §3 Stage 3; open question C16 (tier 2e from draws).

## Global Constraints

As before, plus: the mean model's parameters are FROZEN during the noise fit; every draw uses a seeded generator so that the objective is deterministic in the noise parameters (common random numbers); every population statistic is computed on held-out halves (calibrate on A, score on B, both swaps); the permutation control of exp60 is run (draws assigned to shuffled galaxies must NOT score as well).

---

### Task 1: `process3.py` — the three sources, their draws and their moments

**Interfaces:**
- `Noise3(n_eff, sigma_eta, ell, sigma_sc, sigma_se)`; `NOISE_NAMES`, `NOISE_BOUNDS` (n_eff ∈ [0.5, 200], sigma_eta ∈ [0, 1.0] dex, ell ∈ [0.02, 2] in ln t, sigma_sc, sigma_se ∈ [0, 0.6] dex); `Noise3.off()` = (1e6, 0, 1, 0, 0) — draws equal the mean bit for bit.
- `draw(spec2, theta, noise, curves, R, rng, n_draw=8, epochs=(0,1,2,3,4), nodes=FULL_NODES) -> (S, n, 5, nR)`:
  - lumpy accretion: on each node interval the extended channel's mass $m_i$ is replaced by $m_i K_i/\mu_i$ with $K_i\sim{\rm Poisson}(\mu_i)$, $\mu_i = n_{\rm eff}\,w_i\,{\rm d}\ln M/{\rm d}\ln t|_i$ (events per e-fold of halo growth); mean preserved exactly in expectation;
  - correlated efficiency: $\eta$ a zero-mean Gaussian process on the node grid with covariance $\sigma_\eta^2\exp(-|\Delta\ln t|/\ell)$ (Cholesky once per call), applied as $\varepsilon\to\varepsilon\,10^{\eta}$ to BOTH channels, then divided by the population-mean factor $\langle10^{\eta}\rangle=10^{\sigma_\eta^2\ln10/2}$ so the mean is preserved;
  - size offsets: per galaxy per draw, $\log f_c\to\log f_c+\mathcal N(0,\sigma_{sc})$, $\log f_e\to\log f_e+\mathcal N(0,\sigma_{se})$.
  - all three are drawn once per (draw, galaxy) and shared by every epoch — coherence by construction.
- `lumpy_covariance(spec2, theta, noise, curves, R) -> (n, nR, nR)` at $z=0.4$: $\sum_i m_i^2/\mu_i\,F_e(R/s_i)F_e(R'/s_i)$ (Campbell).

- [ ] Self-check: `Noise3.off()` reproduces `predict2` bit for bit; the draw mean over 200 draws of the lumpy source alone matches `predict2` to 1 per cent and its variance matches `lumpy_covariance` diagonal to 10 per cent on 20 galaxies; the efficiency source alone preserves the mean of $M_*(<100)$ to 1 per cent over 200 draws; draws are monotone in radius and time; timing for 8 draws × 2397 galaxies.

### Task 2: `stage3_process.py`

- [ ] **Step 1: the targets** at $z=0.4$ from the truth: tier-2b plane scatters (`kpc:M(30-50)|M(<30)`, `kpc:M(50-100)|M(<30)`), tier-2d $R_{50}|M_*$ scatter, the tier-2e width ratios; the split-half floors from `qa.plane_energy`.
- [ ] **Step 2: the leverage sweep.** Each source alone at three amplitudes ($n_{\rm eff}\in\{2,5,20\}$; $\sigma_\eta\in\{0.1,0.2,0.4\}$ dex with $\ell\in\{0.1,0.5\}$; $\sigma_{sc},\sigma_{se}\in\{0.1,0.2,0.4\}$), 4 draws each on the full sample: the plane widths and the mean shift ($M_*(<10)$, $M_*(<100)$ medians). A source that cannot move the `M(30-50)|M(<30)` width by ≥ 0.03 dex without shifting a median by > 2 per cent is dropped. Printed as a table and drawn.
- [ ] **Step 3: the fit.** Objective = sum over the $z=0.4$ tier-2b kpc planes and the tier-2d $R_{50}$ plane of `plane_energy` E/floor (draw population vs truth) + sum of tier-2e W1 ratios, on half A; Nelder-Mead over the surviving parameters with common random numbers (8 draws); scored on half B; both swaps; the permutation control.
- [ ] **Step 4: gates.** G3a: kpc-plane E/floor ≤ 1.5 held out at $z=0.4$ (exp60: 1.69/2.17). G3b: tier-2e width ratios within 0.15 of 1 at $z=0.4$ from draws (C16 closed). Both reported for the mean alone as the null.
- [ ] **Step 5: predictions.** The same draws at $z=0.7$–2.0: tier 2c size-rank persistence $z=0.4\leftrightarrow2$ (truth 0.33; exp60 0.97), the declining fraction (target 41.8 per cent) and median depth (−0.066 dex; C14's exchangeable ceiling −0.120), the tier-2e widths per epoch — reported, not fitted.
- [ ] **Step 6: figures** (`sweep`, `planes` with drawn population vs truth, `predictions`), README, todo, commit.
