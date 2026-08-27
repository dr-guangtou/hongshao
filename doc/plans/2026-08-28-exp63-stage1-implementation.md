# exp63 Stage 1 Implementation Plan — read the deposit-size distribution and the size law from the data

> **For agentic workers:** execute inline in the primary session. Steps use checkbox (`- [ ]`) syntax. Stage 2's plan is written after this stage's gate G1 is answered.

**Goal:** For every $z=0.4$ galaxy, recover the non-negative distribution of stellar mass over deposit size $W$ that its measured curve of growth requires under a fixed kernel (a deconvolution), then match its cumulative against the cumulative stellar mass the incumbent's efficiency law deposits in time to obtain the size-versus-time law each galaxy demands, $s^\ast(t)$; measure whether one size curve or two are needed (gate G1), and which halo variables have leverage on $W$ (1c, decision D2).

**Architecture:** One script `stage1_deconvolve.py` with three parts (1a representation, 1b quantile matching, 1c leverage), three kernels (cored `gompertz_log` c=0.80 — the incumbent's; cuspy Sérsic n=4; Sérsic n=1 — exp55's compact component), one output npz and four figures. Uses `engine.py` for the cumulative deposition in time and the analytic $R_{200c}(t)$.

**Tech Stack:** `scipy.optimize.nnls` with a second-difference Tikhonov penalty appended as rows (still non-negative least squares), `scipy.stats.spearmanr`, `exp53/families.cog_unit`, `engine.nodes/_deposits`.

**Spec:** `doc/plans/2026-08-28-exp63-analytic-growth-model.md` §3 Stage 1, §4 D2; tech note 04 §2 and §4.

## Global Constraints

As Stage 0's plan, plus: the deconvolution is a REPRESENTATION exercise (per-object freedom, no halo input except in 1b's time labels and 1c's correlations); every number is quoted per kernel; the untruncated kernels are used (Stage 0 measured the truncation at 3 per cent of the mass, at the far outskirts — irrelevant to the size distribution inside 148 kpc; stated in the README).

---

### Task 1: `stage1_deconvolve.py`

**Files:**
- Create: `experiments/exp63_analytic_growth/stage1_deconvolve.py`
- Output: `outputs/stage1_deconvolve.{npz,log}`, `figures/exp63_stage1_{representation,w_population,size_law,leverage}.{png,pdf}`
- Modify: `README.md` (Stage 1 section), `doc/todo.md`

**Interfaces:**
- Consumes: `S0.build` (recs, data, mask, lmh, spec, theta), `engine.build_curves`, `engine.nodes`, `engine._deposits`, `engine.r200c_analytic`, `engine.z_of_t`, `families.cog_unit`, `population.npz` (`t50`, `c200c`), `diffmah_combined.fits` official columns.
- Produces: `stage1_deconvolve.npz` with `s_grid (K,)`, per kernel `w_<k> (n, K)`, `rms_dex_<k> (n,)`, `rms_dex_perm_<k> (n,)`, `n_modes_<k> (n,)`, `q_grid (Q,)`, `log_s_star_over_r200_<k> (n, Q)`, `z_star (n, Q)`, `leverage_<k>` table, `g1_verdict_<k>` (str).

- [ ] **Step 1: the deconvolution.** Grid $s_k$: 20 log-spaced sizes from 0.3 to 600 kpc. Design matrix $A_{jk}=F(R_j/s_k)$ with the untruncated kernel; rows scaled by $1/d_j$ (fractional residual, the production loss's residual); penalty rows $\lambda D_2$ (second difference on $w$, scaled by the row-norm of $A$) with $\lambda$ chosen on the smoke sample as the largest value that degrades the median fractional RMS by less than 10 per cent relative to $\lambda=0$, then frozen; `nnls`. Report per kernel: median CoG RMS in dex, 16–84 per cent; the permuted control (the RMS of galaxy $i$'s CoG against the fit of galaxy $\pi(i)$, one fixed permutation); best/typical/worst galaxies drawn.
- [ ] **Step 2: quantile matching.** $C_W(\ln s)$: cumulative of $w$ over the grid, interpolated linearly in $\ln s$ between bin centres, normalised. $C_t(t)$: cumulative of $\varepsilon\,\mathrm{d}M$ over the engine's $z=0.4$ nodes at the incumbent's $\theta$, normalised. For $q\in\{0.05,0.1,\dots,0.95\}$: $\ln s^\ast(q)=C_W^{-1}(q)$, $t^\ast(q)=C_t^{-1}(q)$, $z^\ast=z(t^\ast)$, and $\log_{10}[s^\ast/R_{200c}(t^\ast)]$ with the analytic radius. Report the population median and 16–84 per cent band against $z^\ast$, per kernel, against the incumbent's $\log_{10} f_0+b\log_{10}(1+z)$ with $f_0=10^{-0.843}$, $b=-0.896$.
- [ ] **Step 3: gate G1.** Two measurements, per kernel: (i) the fraction of galaxies whose $w$ has two or more separated modes (local maxima of $w$ after a 3-bin running mean, each holding $\ge10$ per cent of the total, separated by $\ge0.5$ dex); (ii) the spread of $\log(s^\ast/R_{200c})$ across the population at fixed $z^\ast$ (the 16–84 band width, dex) and its trend with $z^\ast$ (the median at $z^\ast>3$ minus at $z^\ast<0.7$). Pre-registered prediction: two modes in $>50$ per cent, and a trend of $\ge0.7$ dex (0.02 early to 0.1–0.3 late). Verdict rule: "two scales warranted" iff the trend exceeds 0.5 dex OR the two-mode fraction exceeds 50 per cent, under the incumbent's own kernel.
- [ ] **Step 4: leverage (1c, D2).** Per kernel, partial Spearman at fixed $\log M_h$ (residuals after a quadratic) of: the mass-weighted mean $\log s$, the inner fraction $\sum_{s_k<5}w_k/\sum w_k$, the outer fraction $\sum_{s_k>50}w_k/\sum w_k$, with `early`, `late`, `logtc`, `t50`, `c200c`, `f_form(z=0.4)`. A variable is offered to Stage 2 only if $|\rho|\ge0.1$ on the inner fraction or the mean size; D2 applies that rule to `c200c`.
- [ ] **Step 5: figures.** `representation`: RMS histograms per kernel with the permuted control + three example galaxies (data, fit, and $w$ as bars); `w_population`: median $w$ per halo-mass tercile per kernel; `size_law`: $\log(s^\ast/R_{200c})$ vs $z^\ast$ per kernel with the incumbent's law; `leverage`: bar chart of the partial correlations.
- [ ] **Step 6: run `--smoke`** (every 24th galaxy, $\lambda$ scan printed), then full; README Stage 1 section with every number three ways; todo; commit.

## Self-review

- Spec coverage: 1a (Step 1), 1b (Step 2), G1 (Step 3), 1c + D2 (Step 4), figures (Step 5), the caveat on time labels stated in the README (Step 6).
- Placeholder scan: grid, λ rule, mode rule, verdict rule, correlation threshold are all numbers.
- Type consistency: `w_<kernel>` is `(n, K)` everywhere; quantile arrays `(n, Q)`.
