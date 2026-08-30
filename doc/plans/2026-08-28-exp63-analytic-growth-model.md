# exp63 — growing a massive galaxy along its halo's growth curve: the analytic, two-scale, stochastic deposition model

*Written 2026-08-28, before any code, at the user's direction: "pivot to something
different and rethink the model design" after exp60/61 showed the deposition-only
incumbent at its class ceiling. Three requests were made at once — (1) compress the
"follow the halo mass increments, then sum" recipe into an analytic integral, (2) a
new single-epoch ($z=0.4$) model that grows a galaxy along its mass accretion
history and explains the data better without unphysical transport, (3) a model that is
statistical from the start, so the diversity of profiles and the stellar-mass and
size planes are part of the prediction. The mathematics behind (1) is proved in
`doc/tech_note/04_analytic_deposition_integral.md`; this plan builds on it.
Nothing below is implemented. Branch `exp63-analytic-growth`, folder
`experiments/exp63_analytic_growth/`. `exp62` is taken by another worktree.*

**Reading guide.** §1 says what changes and why, in plain language. §2 is the model.
§3 is the experiment, stage by stage, with the gates and the numbers that decide
each stage written down before anything runs. §4 lists the decisions only the user
can take. §5 is what would count as failure.

---

## 1. What the evidence says, and what this plan changes

Three facts from the record, one from this week's probe:

- **The incumbent's own centre is partly a numerical artefact.** Its 72-step sum is a
  first-order approximation of an integral it never wrote down, biased −0.027 dex at
  2 kpc and −0.007 dex at 148 kpc (tech note 04, §1.1). About half of the "7–13 per
  cent low inside a few kpc at every epoch scope" (exp61) is this. The first thing
  exp63 does is remove it and re-measure.
- **The class has one radial coordinate.** Written as an integral, the model's
  profile is the distribution of stellar mass over deposit size, $W(v)$, smoothed by
  the deposit kernel (tech note 04, Eq. 3). One size law $s(t)=f_0(1+z)^bR_{200c}(t)$
  means centre and outskirts are the same law read at two times; exp54 Stage 3.7
  measured that this cannot serve both ends. On the measured grid the data rise as
  $R^{1.00}$ over 2–4.9 kpc (16–84 per cent 0.76–1.14), the incumbent as $R^{0.88}$
  with a distribution less than half as wide (0.77–0.94) (tech note 04, §2–3,
  corrected 2026-08-28: an earlier version of this bullet quoted 1.45 from a wrong
  radius grid).
- **The population is too narrow, and fitting harder makes it narrower**
  (exp61 Stage 4: width 0.70 → 0.54 of the truth's at $M(50\text{–}100)$; per-epoch
  fits 0.46). A per-galaxy loss rewards the conditional mean and nothing rewards
  width. The exp60 layer, applied to the output, reached 1.69× the split-half floor
  on the kpc planes and put drawn decliners twice too deep (C14); its drawn sizes
  persisted across epochs at rank 0.97 against the truth's 0.33 (tier 2c) because the
  noise was added per epoch to a finished profile rather than to the history.
- **The halo history knows a little about the core, in the direction physics
  expects.** At fixed $\log M_h(z=0.4)$, the measured $\log M_*(<4.9)/M_*(<33)$
  correlates with the DiffMAH early index at $-0.21$, the late index at $+0.21$,
  $t_{50}$ at $+0.08$ (partial Spearman, 2397 galaxies, measured grid): late-forming
  haloes hold more centrally concentrated stars, early-forming ones have built larger
  envelopes ($\log M_*(<148)/M_*(<33)$: early $+0.16$, late $-0.15$, $t_{50}$ $-0.17$).
  Weak ($|\rho|\le0.23$, i.e. $\lesssim5$ per cent of the variance), so most of the
  0.097 dex scatter in the inner ratio at fixed halo mass is intrinsic to the halo
  record and must be carried by a stochastic component.

What this plan changes, in order of how much it departs from the incumbent:

1. **The engine**: the deposition sum becomes the exact integral along the DiffMAH
   curve, with closed-form cosmology and (decision D1) closed-form $R_{200c}$. No
   catalog, no snapshot grid, smooth in every parameter.
2. **The inversion before the model**: the deposit-size distribution each galaxy's
   $z=0.4$ profile requires is read off the data first (a non-negative deconvolution),
   and the size-versus-time law is obtained by quantile matching against the
   efficiency law's cumulative deposition — a measurement of what any size law must
   look like, made before one is assumed.
3. **Two channels with two scales**, in-situ and ex-situ, both deposition (no
   transport), with the split between them an empirical function of halo mass and
   the in-situ deposits allowed a cuspy kernel. This is the "Direction 3" mean model.
4. **Noise inside the process, not on its output**: lumpy ex-situ accretion, a
   correlated efficiency history, and per-galaxy size offsets — three sources whose
   moments are integrals of the same form as the model itself — fitted to population
   statistics with a proper scoring rule, held out.
5. **One epoch fitted, four predicted.** Everything is fitted at $z=0.4$. The earlier
   epochs are the same integral truncated at $t(z_k)$, reported on both samples
   ([[per-epoch-ceiling-does-not-transfer]]), decliners split out, never fitted.

## 2. The model

### 2.1 The mean: two deposition channels along one history

For a halo with DiffMAH parameters $\theta_{\rm MAH}=(\log m_p,\log t_c,\alpha_e,\alpha_l)$,

$$
M_*(<R,\,t)=\int_{t_{\min}}^{t}\frac{\mathrm{d}M}{\mathrm{d}t'}
\Big[\varepsilon_{\rm is}\,F_{\rm is}\!\Big(\tfrac{R}{s_{\rm is}(t')}\Big)
+\varepsilon_{\rm ex}\,F_{\rm ex}\!\Big(\tfrac{R}{s_{\rm ex}(t')}\Big)\Big]\mathrm{d}t',
$$

with, per channel $k\in\{\rm is,ex\}$, evaluated at the deposit's own time:

- **efficiency** $\log_{10}\varepsilon_k=a_{0,k}+a_{M}\,m+a_{z}\,x+a_{Mz}\,m\,x$
  shared slopes, channel-specific amplitude, and the in-situ share of the deposited
  stellar mass $w_{\rm is}(m)=\sigma\big((m_{1/2}-m)/\Delta\big)$, a logistic in halo
  mass falling from ~1 at low mass to ~0 at cluster mass (the empirical recipe; the
  literature's ex-situ fraction rising from ~0.2 at $10^{12}$ to ~0.8 at $10^{14}$
  is a *prior* for the starting point, not a constraint);
- **sizes** $s_k=f_k\,(1+z)^{b_k}\,R_{200c}(t')$, with the prior expectation
  $f_{\rm is}\approx0.015$–$0.03$ (Kravtsov 2013: $R_{1/2}\simeq0.015\,R_{200c}$;
  the incumbent's single $f_0=0.14$ is what a one-scale law must compromise to) and
  $f_{\rm ex}\approx0.1$–$0.3$;
- **kernels** $F_{\rm is}$ a Sérsic deposit in the $R_{50}$ parametrisation of
  `exp53/families.py` with index $n_{\rm is}$ free in $[1,6]$ (cuspy: dissipative
  in-situ formation), $F_{\rm ex}$ the `gompertz_log` or Moffat family (cored: stars
  stripped from satellites land in shells). Both have closed-form Mellin transforms
  (tech note 04, §2), so the power-law regime of each channel is analytic;
- **truncation** at $3R_{200c}(t')$ as before, or none (decision D3);
- **no transport**: nothing moves after deposition. The only optional
  post-deposition term is the physically *bounded* expansion from stellar mass loss,
  $g(\Delta t)=1/\big(1-f_{\rm loss}(\Delta t)\big)\le1.6$ with $f_{\rm loss}$ the
  standard return fraction of a simple stellar population — a factor with no free
  reach, applied to the in-situ channel only, switched on only if Stage 2's decliner
  report asks for it and only after it passes exp57's G2 (reach) gate.

Parameter count: 4 (efficiency slopes and one amplitude) + 1 (second amplitude) + 2
(split) + 4 (two sizes) + 2 (two kernel shapes) = **13**, every one nested so that
$w_{\rm is}\equiv1$ or $\equiv0$, $n_{\rm is}$ fixed, or $f_{\rm is}=f_{\rm ex}$
reproduces the incumbent's class. Identifiability is measured by profiling
([[conditional-slices-overstate-every-parameter]]), not by conditional slices.

### 2.2 The noise: three sources with analytic moments

Each source is a random modification of the *history*, so a draw is a whole galaxy
at every epoch, coherent by construction (tier 2c becomes a prediction).

1. **Lumpy ex-situ accretion.** The ex-situ channel's deposition is a marked point
   process along the history: events arrive at a rate proportional to
   $\mathrm{d}M/\mathrm{d}t'$ with an effective number $N_{\rm eff}$ of contributing
   events per e-fold of halo growth (one parameter; the merger-rate literature says a
   handful, because a few major mergers carry most ex-situ mass), each event
   depositing its share of $\varepsilon_{\rm ex}\,\Delta M$. The mean is the smooth
   integral; the variance and the covariance between radii follow from Campbell's
   theorem as integrals of $F_{\rm ex}^2$ — so the outskirts' scatter at fixed inner
   mass (truth 0.17 dex, incumbent mean 0.044 dex on the $M(30\text{–}50)|M(<30)$
   plane) is a *computed* number before any fit.
2. **A correlated efficiency history.** $\log\varepsilon_{\rm is}\to\log\varepsilon_{\rm is}+\eta(\ln t)$,
   $\eta$ a stationary Gaussian process with amplitude $\sigma_\eta$ and correlation
   length $\ell$ in $\ln t$ (two parameters). Short $\ell$ decorrelates centre from
   outskirts, long $\ell$ is a per-galaxy amplitude — the exp60 amplitude component
   as the $\ell\to\infty$ limit.
3. **Per-galaxy size offsets.** $\log f_{\rm is}$ and $\log f_{\rm ex}$ each carry a
   per-galaxy normal offset ($\sigma_{\rm is},\sigma_{\rm ex}$; two parameters).
   These are the exp60 "size/shape axis", now inside the model.

Five noise parameters. Fitted to *population* statistics at $z=0.4$ with a proper
scoring rule on draws (the energy score over the tier-2b/2d/2e quantities, which
rewards width exactly as much as location), calibrated on one random half and scored
on the other, with the exp60 permutation control. The mean model is fitted first with
the noise off; the noise is then fitted with the mean frozen; then both are re-fitted
jointly once, and the joint fit is accepted only if it does not move the mean's
Stage 2 gates (the trap exp60 Option B fell into — C15 — was hitting the decline
distribution by shifting the mean).

## 3. The experiment

Method throughout is the standing one: gates first, loss last; sweep leverage by
evaluation before spending a fit; a bound must match what it reports; every gate at
the null first; `--smoke` never writes production filenames; long fits in the main
shell; `import fit` before `halo`/`model`. Every stage produces figures in the
project style and a README section that states each number's measurement, meaning,
reference and figure.

### Stage 0 — the analytic engine, and the incumbent re-measured (gated)

Build `engine.py`: DiffMAH curve and growth rate in closed form; $z(t)$ and
$\rho_c(z)$ in closed form; $R_{200c}$ analytic (D1); Gauss–Legendre in $\ln t$ with
the panel count chosen by a convergence test; kernels with $R_{50}$ parametrisation
and closed-form Mellin transforms; a batched evaluation over all 2397 galaxies.

- **G0a, same-convention reproduction.** Evaluated on the 72-step grid with the
  incumbent's end-of-step convention and catalog $R_{200c}$, the engine reproduces
  `exp54/model.forward` to $10^{-6}$ dex on every galaxy without a mid-history gap
  (the probe reached $2\times10^{-6}$).
- **G0b, convergence.** Halving the panels moves no profile by more than $10^{-6}$ dex.
- **G0c, closed forms.** The power-law/EdS case reproduces tech note 04 Eq. (5) to
  $10^{-8}$ dex for all three kernels.
- **G0d, cost.** Full-sample $z=0.4$ evaluation timed; the plan's fits are scheduled
  from that number, not guessed.
- **Result 0.** The incumbent's $\theta$ evaluated by the exact integral, through the
  standard `qa.evaluate` battery: how much of the class's central deficit and of its
  epoch drift was the Riemann sum. Reported on both samples. This is the new
  baseline every later comparison is made against.

### Stage 1 — read the deposit-size distribution and the size law from the data

For every galaxy at $z=0.4$, with the kernel fixed (done twice: cored
`gompertz_log` $c=0.8$, cuspy Sérsic $n=4$), fit a non-negative $W(v)$ on a fixed
grid of 8 log-spaced deposit sizes from 0.5 to 500 kpc (non-negative least squares
on the 24-point curve of growth with the total free; a smoothness penalty only if the
NNLS solution is spiky — chosen on the smoke sample and frozen).

- **Measurement 1a — the representation floor of the class at one epoch.** Median
  CoG RMS of the NNLS fit against the 0.0044 dex of exp55's four-parameter analytic
  form and the 0.0034 dex of PCA-3. This is the exact single-epoch ceiling of "any
  deposition-only profile with this kernel", with per-object freedom; its control is
  the same fit on a permuted (galaxy-swapped) target ([[ceiling-needs-a-permuted-control]]).
- **Measurement 1b — the required size history.** With the incumbent's efficiency
  law giving the cumulative deposited stellar mass in time $C_t(t)$, and the fitted
  $W$ giving the cumulative in size $C_W(v)$, quantile matching yields
  $s^\ast(t)=C_W^{-1}(C_t(t))$ per galaxy. Plot $s^\ast(t)/R_{200c}(t)$ against $z$
  for the population (median and 16–84 per cent band) and against the incumbent's
  $0.14\,(1+z)^{-0.9}$.
- **G1 (decides the mean model's form, before any fit).** Pre-registered prediction:
  $s^\ast/R_{200c}$ is **not** one curve — it sits near 0.02 at early times and
  climbs to 0.1–0.3 late, and the per-galaxy $W$ has two modes in more than half the
  sample. If instead one curve within a 0.15 dex band fits 80 per cent of galaxies,
  the two-channel model is *not* warranted and Stage 2 fits the one-scale law on
  the exact engine (a cheaper experiment; still a result).
- **Measurement 1c — leverage of the halo record on $W$.** Partial correlations at
  fixed halo mass of the $W$ modes' positions and weights with $\alpha_e,\alpha_l,
  t_{50},c_{200c},f_{\rm form}$: which halo variables the split and the sizes may
  depend on. A variable with $|\rho|<0.1$ is not offered to the fit (exp59's rule).
- **Caveat stated up front.** One epoch fixes $W$, not the time labels;
  $(\varepsilon(t),s(t))$ are degenerate here. Measurement 1b uses the incumbent's
  $\varepsilon$ to place the labels. Stage 2's earlier-epoch predictions are what test
  the labels.

### Stage 2 — the mean model, fitted at $z=0.4$ only, predicted elsewhere

Fit §2.1 on the standard $z=0.4$ sample (2397, `sane_history_mask`) under a
single-epoch objective: the production loss's shape term plus the amplitude term at
$z=0.4$ only, and a **shells/log** term (`Objective(quantity="shells", residual="log")`,
the one objective that can see the innermost aperture — [[density-objective-cannot-see-the-centre]])
at equal weight; the weighting fixed on the smoke sample and frozen. Multi-start with a
named nested start (the incumbent embedded, $w_{\rm is}\equiv1$) and an assertion
that the nested start reproduces Stage 0's loss.

Gates, at the null (incumbent on the exact engine) first, then the candidate:

- **G2a inner slope**: the median $\mathrm{d}\log M_*/\mathrm{d}\log R$ over 2–4.9 kpc
  (grid points 0 and 3 of `fit.R_GRID`) within 0.05 of the data's 1.00, and its
  16–84 per cent width at least 0.7 of the data's 0.38 (the incumbent: 0.88 and
  0.17 — the width is the larger failure).
- **G2b tercile profiles**: median (model − data)/data within ±5 per cent at every
  radius in each halo-mass tercile, amplitude-pinned and raw (the incumbent: −10 per
  cent at 3 kpc, +7 per cent at 20 kpc in the top tercile).
- **G2c plane slope**: the $M(30\text{–}50)|M(<30)$ conditional slope within 0.1 of
  the truth's 1.42 (incumbent 1.25); the model-mean scatter is *not* gated here —
  width is the layer's job.
- **G2d no transport**: the fraction of $z=0.7\to0.4$ added stellar mass beyond
  50 kpc equals the data's within the exp57 G1b tolerance, and mass is conserved to
  the truncation boundary (exp54 contract check 3).
- **G2e sizes physical**: $f_{\rm is}<f_{\rm ex}$ at every epoch, $f_{\rm is}$
  within a factor 3 of 0.02.
- **G2f identifiability**: profiled ranges, not slices; a parameter whose profile is
  flat across its box is removed and the fit repeated.
- **Prediction, not fit — Result 2.** The fitted $\theta$ truncated at $t(z_k)$ for
  $z_k=0.7,1.0,1.5,2.0$, through the standard battery on both samples, with B1/B3 at
  each epoch and the decliner/non-decliner split (C8 table) alongside the incumbent
  and the exp61 per-epoch ceiling. Expected and acceptable: the compact in-situ
  channel places central mass *early*, so decliners will be over-predicted at $z=2$
  — that is the class limit being stated by a prediction rather than hidden by a fit.
  If the non-decliners' central span (exp54 C8: 0.045 dex) is not within 0.03 dex,
  the optional bounded mass-loss expansion is tried, once, under exp57's G2.

### Stage 3 — the stochastic deposition process

- **Sweep first (exp59's rule).** For each of the three noise sources alone, at
  three amplitudes, the *computed* (not drawn) plane widths and the mass-size scatter:
  which sources can move $M(30\text{–}50)|M(<30)$ from 0.044 toward 0.170 dex, and
  $R_{50}|M_*$ toward the truth's scatter, without moving the mean. A source with no
  leverage is dropped before the fit.
- **Fit** the surviving parameters to the held-out population statistics (energy
  score on tier-2b kpc planes, tier-2d mass-size, tier-2e CDFs at $z=0.4$), mean
  frozen; permutation control; then one joint re-fit, accepted only if G2a–G2e hold.
- **G3a planes**: energy-distance ratio to the split-half floor $\le1.5$ on the
  $z=0.4$ kpc planes (exp60 Option A: 1.69/2.17; exp41's 1.0–1.4 was oracle-pinned
  and is not the bar), and the mass-size plane ratio reported against exp60's.
- **G3b tier 2e from draws** (closes C16): the mass CDFs scored on drawn
  populations, width ratios within 0.15 of 1 at $z=0.4$.
- **Prediction — Result 3.** The same draws propagated to $z=0.7$–$2.0$: tier 2c
  coherence (truth size-rank persistence 0.33 between $z=0.4$ and 2; exp60 gave
  0.97) and the declining fraction and depth (targets 41.8 per cent, −0.066 dex;
  C14's exchangeable-draw ceiling was −0.120) — reported, not fitted.

### Stage 4 — the battery, the record, the handover

Standard `qa.evaluate` on three models (incumbent on the exact engine; the exp63
mean; the exp63 mean plus process draws), both samples, every epoch; the README with
one section per stage; PROBLEMS.md; memory; `doc/todo.md` review; a technical note
if the mean model is adopted.

### Cost, measured before scheduling

Stage 0's G0d sets it. The incumbent's evaluation costs 112 ms per full-sample call
at five epochs; a batched single-epoch quadrature over 2397 galaxies is expected to be
faster, and a smooth objective allows gradient-based optimisation, but the schedule
is written only after G0d. Stage 1's per-galaxy NNLS fits are seconds in total.

## 4. Decisions taken by the user (interview, 2026-08-28)

- **D1 — $R_{200c}$: analytic from $M(t)$ and $\rho_c(z)$.** The model is a
  function of the four DiffMAH numbers and nothing else. The catalog value is kept
  as a comparison column in Stage 0 only.
- **D2 — concentration: only if Stage 1c shows leverage.** $c_{200c}$ (its $z=0.4$
  value, acting on the in-situ size as $R_{200c}/c$) is offered to the fit only if
  its partial correlation with the required deposit-size distribution is
  $|\rho|\ge0.1$ at fixed halo mass; otherwise it never enters.
- **D3 — truncation: measure first.** Stage 0 reports the stellar mass the
  untruncated kernels place beyond $3R_{200c}$ for the incumbent's parameters; the
  $3R_{200c}$ truncation is kept only if that exceeds 1 per cent of the total, and
  the choice is frozen before Stage 1.
- **D4 — objective: the production loss plus the shells/log term at equal
  weight**, the weight fixed on the smoke sample and frozen; gate G2a reported under
  both objectives.
- **D5 — earlier epochs: report both samples at every epoch**, the full progenitor
  sample and the halo-mass-complete subset side by side; no pass/fail gate at
  $z\ge0.7$; C13 stays open as a scope note, nothing at $z>0.4$ is fitted.

## 5. What would count as failure, stated now

- G1 finds one universal size curve: the two-channel mean is not built; the exact
  engine and the corrected incumbent are the result.
- The two-channel mean passes G2a–G2c at $z=0.4$ but its $z=2$ prediction on the
  non-decliners is worse than the incumbent's by more than 0.05 dex in B3: the
  in-situ channel's time labels are wrong, and the plan's "one epoch fitted, four
  predicted" claim fails for this class — recorded as such.
- No noise source has leverage on the outskirts' width at fixed inner mass: the
  diversity is not in the halo *history* either, and the exp60 output layer stands.
- The joint fit only reaches G3a by moving the mean: the same trap as exp60 Option B;
  the mean-frozen result is kept and the trap recorded.
