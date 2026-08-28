# exp63 — growing a massive galaxy along its halo's growth curve

Branch `exp63-analytic-growth`. Plan: `doc/plans/2026-08-28-exp63-analytic-growth-model.md`
(the science plan, with the user's decisions D1–D5 in §4); derivation:
`doc/tech_note/04_analytic_deposition_integral.md`. One implementation plan per
stage in `doc/plans/2026-08-28-exp63-stage*-implementation.md`.

**The question.** exp54's deposition-only model (the "incumbent", hongshao v1's
mean) walks the halo's DiffMAH growth curve in 72 merger-tree steps, deposits stars
at each step, and adds up. Three things the user asked for at once: write that
recipe as one exact integral (done, tech note 04); a new $z=0.4$ model that grows
the galaxy along its history with two size scales and no transport; and a model
that is statistical from the start, so the diversity of profiles and the
stellar-mass and size planes are predictions. Everything is fitted at $z=0.4$ only;
the earlier epochs are predicted by truncating the integral and reported on both
samples (D5).

| script | what it does | writes |
|---|---|---|
| `engine.py` | the exact-integral engine: four DiffMAH numbers + closed-form cosmology → M*(<R) at five epochs, batched; the bridge `discrete_sum` to `exp54/model.forward`; self-check | — |
| `stage0_engine.py` | gates G0a–G0d, the D3 measurement, the bookkeeping the integral removes | `outputs/stage0_engine.{npz,log}`, `figures/exp63_stage0_engine` |
| `stage0_rebaseline.py` | Result 0: the incumbent's seven numbers through the standard battery on three integrations | `outputs/stage0_rebaseline.{npz,log}`, `figures/qa/qa_*_exp63_*`, `figures/exp63_stage0_compare` |

`outputs/` and `figures/` are gitignored and regenerable. Run everything with
`HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u <script> [--smoke]`.

## Stage 0 — the exact-integral engine, and the incumbent re-measured (2026-08-28)

### What the engine is

`engine.predict(spec, theta, curves, R)` evaluates

$$M_*(<R,t)=\int_{t_{\rm start}}^{t}\varepsilon\big(M(t'),z(t')\big)\frac{\mathrm{d}M}{\mathrm{d}t'}\,F\!\Big(\frac{R}{s(t')}\Big)\mathrm{d}t'$$

by Gauss–Legendre quadrature in $\ln t$ (8 panels × 24 nodes from $t=0.27$ Gyr to
$z=2$, then one panel per observation interval; the same nodes serve all five
epochs and all galaxies), with $M(t)$ the official DiffMAH curve, $z(t)$ and
$\rho_c(z)$ in closed form, $R_{200c}$ from $M(t)$ and $\rho_c(z)$ (**D1**), and the
incumbent's efficiency law, size law and truncated `gompertz_log` kernel imported
from `exp54/model.py` unchanged. A halo is four numbers; nothing stellar exists on
a `HaloCurve` (asserted).

### The gates (`stage0_engine.py`, all 2397 galaxies; log in `outputs/stage0_engine.log`)

| gate | what it checks | measured | threshold | verdict |
|---|---|---|---|---|
| G0a bridge | on the incumbent's own step grid with the catalog $R_{200c}$, `discrete_sum(sub=1)` reproduces `exp54/model.forward` (2073 galaxies without a mid-history gap) | max 2.4e-6 dex | 1e-5 dex (the stored curve is float32: 2.7e-6 dex) | pass |
| G0b limit | splitting each step 2/4/8/16× halves the distance to the integral per doubling; the Richardson extrapolation $2S_{16}-S_8$ lands on `predict` | ratios 1.96–2.06; median 1.4e-5 dex | ratios in [1.8, 2.2]; 5e-4 dex | pass |
| G0c closed forms | tech note 04 Eq. 5 at four MAH slopes; the Mellin constants of the three kernels | 2.4e-10 | 1e-8 | pass |
| G0d cost | one full `predict`, 2397 galaxies × 5 epochs × 24 radii | 0.32 s | reported | pass |

So **the integral is exactly the limit of the incumbent's sum** (G0b), the engine's
ingredients are exactly the incumbent's (G0a), and a full-sample evaluation costs a
third of a second — a fit of a few thousand evaluations is minutes, not hours.

**D3, measured.** With the incumbent's parameters and the `gompertz_log` kernel
untruncated, the stellar mass that would lie beyond $3R_{200c}$ of its own deposit
is **3.16 per cent** of the $z=0.4$ total (16–84 per cent: 2.90–3.38; max 3.80;
`figures/exp63_stage0_engine.png` panel c). The user's rule was "keep the truncation
iff the median exceeds 1 per cent": **the $3R_{200c}$ truncation is kept**, and the
exp54 mass-conservation contract stands unchanged.

**Bookkeeping the integral removes.** 324 of 2397 galaxies (13.5 per cent) have a
snapshot in the middle of their history with no catalog $R_{200c}$; the incumbent
drops that step's stars entirely (90th percentile 1.0 per cent of the galaxy's kept
$z=0.4$ stellar mass, maximum 8.2 per cent). Every galaxy also loses its first one
to five steps ($z\gtrsim10$; median 0.008 per cent, max 5.0 per cent). The 72-step
sum's discretisation error is $-0.027$ dex at 2 kpc falling to $-0.007$ dex at 148 kpc
(2073 galaxies, all five epochs within 0.003 dex of each other; panel a).

### Result 0 — the same seven numbers, three integrations (`stage0_rebaseline.py`)

Three products of the incumbent's frozen $\theta$, the same 2397 galaxies, the
standard `qa.evaluate` battery (`figures/qa/qa_*_exp63_{72step,exact-catalog,exact-analytic}`):
`72step` is hongshao v1 exactly as exp61 built it; `exact-catalog` is the integral
with the catalog $R_{200c}$ (v1 minus its numerics); `exact-analytic` is the integral
with $R_{200c}$ from $M(t)$ — **the exp63 baseline**.

Median relative bias, per cent (all 2397 / the mh-complete fit mask):

| quantity | product | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|---|
| $M_*(<10\,{\rm kpc})$ | 72step | −2.4 / −2.4 | −0.7 / +0.3 | +0.1 / +2.2 | −5.6 / −1.6 | −12.5 / −12.8 |
| | exact-catalog | +1.6 / +1.6 | +3.6 / +4.8 | +3.8 / +6.5 | −2.4 / +2.5 | −9.8 / −9.2 |
| | exact-analytic | +3.6 / +3.6 | +5.3 / +6.7 | +5.8 / +8.9 | −0.8 / +4.3 | −8.9 / −7.8 |
| $M_*(<100\,{\rm kpc})$ | 72step | −1.9 / −1.9 | +2.6 / +1.6 | +3.6 / +2.7 | +0.8 / +0.6 | −6.4 / −10.0 |
| | exact-catalog | 0.0 / 0.0 | +4.8 / +4.2 | +5.9 / +5.0 | +2.8 / +2.8 | −4.9 / −8.1 |
| | exact-analytic | +0.4 / +0.4 | +5.1 / +4.6 | +6.4 / +5.7 | +3.0 / +3.0 | −4.7 / −7.8 |

Tier 3 (median max|rel| beyond 5 kpc): 72step 0.2765 / 0.2805 / 0.2868 / 0.2965 /
0.3233; exact-analytic 0.2800 / 0.2818 / 0.2914 / 0.2910 / 0.3140. Inner curve-of-growth
slope over 2–6.4 kpc at $z=0.4$: data median 0.94 (16–84 per cent 0.74–1.13); 72step
0.85 (0.74–0.95); exact-analytic 0.83 (0.72–0.92).

**What it means, and a correction to tech note 04's first reading.** The integral
adds stellar mass everywhere (0.01–0.03 dex, most at the centre), so at the frozen
$\theta$ the incumbent's $z=0.4$ centre goes from 2 per cent low to 4 per cent high
and the $z=2$ deficit shrinks from −12.5 to −8.9 per cent. But the amplitude-pinned
residual pattern — a dip of −8 to −12 per cent at 3–5 kpc and an excess of +5 to
+8 per cent at 10–30 kpc in the top halo-mass tercile — is **unchanged** by the
integration (`figures/exp63_stage0_compare.png`, bottom row: the three line styles
run parallel). The discretisation error was a *level* the fit absorbed into $\theta$;
the *shape* defect is the deposit-size distribution's, exactly as tech note 04 §2
argues, and the tech note's opening sentence ("about half of the central deficit is
numerics") overstated the numerical share — it is half of the *level* at 2 kpc, none
of the shape. The inner slope is if anything flatter on the exact integral (0.83
against the data's 0.94), and its distribution is half as wide as the data's.

The $R_{200c}$ source (D1) moves the centre by a further +2 per cent at $z=0.4$ and
the tier-3 shape by 0.001; the two integrals are the same model to within the
catalog's own scatter. The $z\ge0.7$ $M_*(50\text{–}100)$ biases on the full sample
(+5 to +32 per cent) against the fit mask (−4 to −10 per cent) are the
progenitor-selection effect exp58 P-C measured, unchanged here.

**Decision.** The exp63 baseline is `exact-analytic` at the incumbent's $\theta$; every
later comparison is against it. No refit of the seven parameters on the exact engine
is made here — Stage 2's nested start ($w_{\rm is}\equiv1$, the incumbent's class on
the exact engine, $z=0.4$ only) is that refit, by construction.

## Stage 1 — the deposit-size distribution and the size law, read from the data (2026-08-28)

Tech note 04 Eq. 3 says a deposition-only profile is the cumulative of $W(v)$ — the
stellar mass per logarithm of deposit size — smoothed by the deposit kernel. So, with
a kernel fixed, $W$ can be **recovered** from each measured $z=0.4$ curve of growth
by non-negative least squares (`stage1_deconvolve.py`: 20 size bins from 0.3 to
600 kpc, fractional residuals, no smoothing penalty — the penalty scan showed even
$\lambda=0.003$ costs 12 per cent in median RMS, so $\lambda=0$ was frozen). Three
kernels: the incumbent's cored `gompertz_log` ($c=0.80$), a Sérsic $n=4$ (cuspy),
a Sérsic $n=1$ (exp55's compact component). 2397 galaxies. Log in
`outputs/stage1_deconvolve.log`, arrays in `outputs/stage1_deconvolve.npz`.

### 1a — how well can ANY deposition-only profile represent a massive galaxy's curve of growth?

| kernel | median CoG RMS [dex] (16–84 %) | permuted control | galaxies with two separated size modes |
|---|---|---|---|
| gompertz_log $c=0.80$ (the incumbent's) | **0.0036** (0.0019–0.0070) | 0.0900 | 92.2 % |
| Sérsic $n=1$ | **0.0020** (0.0008–0.0042) | 0.0909 | 99.6 % |
| Sérsic $n=4$ | 0.0133 (0.0074–0.0239) | 0.0902 | 56.7 % |

Reference: exp55's four-parameter analytic form reaches 0.0044 dex, PCA-3 0.0034
(`figures/exp63_stage1_representation.png`, panel a; panels b–d show the best,
typical and worst galaxy under the incumbent's kernel with their $W$ as bars). What
it means: with per-object freedom in *where the stars were deposited*, the
incumbent's own kernel represents the curves as well as the best analytic form the
project has, and an exponential ($n=1$) deposit does better than PCA-3 — so the class
is not limited by its kernel at one epoch. A cuspy $n=4$ deposit **cannot** represent
the curves: the data do not want central cusps in each deposit. That settles the
Stage 2 kernel question from the plan (§2.1): the in-situ channel's Sérsic index
should be near 1, not $\ge2$. The permuted control (galaxy $i$'s curve against galaxy
$\pi(i)$'s $W$, amplitude-matched at 100 kpc) is 0.090 dex under every kernel: the
per-object $W$ carries 25× more information than the family's generic shape.

### The two modes, by halo mass (`figures/exp63_stage1_w_population.png`)

Under the incumbent's kernel $W$ is **bimodal in 92 per cent of galaxies**: a compact
mode whose position does not move with halo mass, and an extended mode that grows
and moves outward with halo mass:

| $\log M_h(z=0.4)$ tercile | share of stellar mass at $s\ge15$ kpc (16–84 %) | compact mode | extended mode |
|---|---|---|---|
| 13.00–13.21 | 0.35 (0.12–0.53) | 6.2 kpc | 43 kpc |
| 13.21–13.40 | 0.45 (0.27–0.60) | 6.0 kpc | 53 kpc |
| 13.40–15.14 | 0.57 (0.40–0.70) | 6.3 kpc | 74 kpc |

(Sérsic $n=1$: compact mode 4.3 kpc at every mass, extended 46 → 62 kpc, share
0.49 → 0.69.) Correlation of the extended share with $\log M_h$: $+0.53$. Between
the modes, at $s\approx10$–30 kpc, the data want almost no deposit mass. What it
means for the model: the plan's two-channel form is what the data ask for — a
compact channel at a fixed few-kpc scale and an extended channel at
$\sim0.1\,R_{200c}$ whose share rises with halo mass (the ex-situ picture) — and the
incumbent's single size law, which must put deposit mass at every intermediate
scale as the halo grows, is what makes its 3–5 kpc dip and 10–30 kpc excess
(Stage 0). The compact mode's size is a *lower edge*: deposits smaller than the
first measured radius (2 kpc) are indistinguishable from the curve alone, so
"6 kpc" means "the compact stars sit at ≤ 6 kpc".

### 1b — the size-versus-time law the data demand (`figures/exp63_stage1_size_law.png`)

Matching the cumulative of $W$ against the cumulative stellar mass the incumbent's
efficiency law deposits in time (quantile by quantile, engine nodes, analytic
$R_{200c}$) gives each galaxy's required $s^\ast(t)$. Population medians of
$\log_{10}[s^\ast/R_{200c}(t^\ast)]$:

| kernel | deposits at $z^\ast>3$ | deposits at $z^\ast<0.7$ | trend |
|---|---|---|---|
| gompertz_log $c=0.80$ | −1.16 (0.069 $R_{200c}$; 16–84 % −1.36..−0.98) | −0.85 (0.14; −1.16..−0.42) | +0.31 dex |
| Sérsic $n=1$ | −1.42 (0.038; −1.60..−1.23) | −0.56 (0.28; −0.88..+0.18) | +0.86 dex |
| incumbent's law $10^{-0.84}(1+z)^{-0.90}$ | −1.47 at $z=4$ | −1.00 at $z=0.5$ | +0.47 dex |

What it means: the late deposits must be **larger** than the incumbent's law makes
them (0.14–0.28 against 0.10 $R_{200c}$), and the early ones about as the law says
(0.04–0.07 against 0.034). The pre-registered prediction (0.02 early, 0.1–0.3 late)
holds for the late end and is a factor 2–3 off at the early end — with the caveat
stated in the plan: the time labels are the incumbent's efficiency law's, which puts
most stellar mass early; a law that deposits more stars late would move the early
labels to smaller sizes. Stage 2's earlier-epoch predictions test the labels.

### Gate G1 and 1c — verdicts

**G1: two scales warranted** (rule: trend > 0.5 dex OR two modes in > 50 per cent,
under the incumbent's kernel; measured: trend +0.31 — below the rule — and two modes
in 92 per cent — far above it). Stage 2 builds the two-channel mean.

**1c leverage** (partial Spearman at fixed $\log M_h$, `figures/exp63_stage1_leverage.png`):
the compact share correlates with the DiffMAH late index at $+0.25$, with
$f_{\rm form}$ at $-0.26$, with $\log t_c$ at $-0.18$ (incumbent kernel); the
extended share with $t_{50}$ at $-0.11$. **$c_{200c}$: $|\rho|\le0.09$ under every
kernel and every statistic — NOT offered to Stage 2 (decision D2 resolved: no).**
`early`, `late`, `logtc`, `t50`, `f_form` pass the 0.1 rule; all of them are
functions of the DiffMAH curve the forward model already integrates, so they are
not added as extra inputs — the test is whether the forward model *reproduces* these
correlations (a Stage 2 report line), not whether they are fitted. The scatter of
the extended share at fixed halo mass (16–84 per cent range 0.12–0.53 in the lowest
tercile) is the diversity the stochastic layer must carry (Stage 3).

## Stage 2 — the two-channel mean, fitted at z=0.4 only, predicted at z=0.7–2.0 (2026-08-28)

`model2.py` is Stage 1's finding as a forward model on the exact engine: the
incumbent's efficiency law, and a deposit split into a **compact channel** (a
Sérsic deposit of index $n_c$ at $s_c=f_c(1+z)^{b_c}R_{200c}$) and an **extended
channel** (the incumbent's `gompertz_log` deposit with shape $c_e$ at
$s_e=f_e(1+z)^{b_e}R_{200c}$), the compact share of each deposit a logistic in the
halo mass at the deposit's own time, $w_c=\sigma((m_{1/2}-\log M)/d)$. Twelve
parameters; `nested_theta` reproduces the exact-engine incumbent bit for bit
(asserted). `stage2_fit.py` fits at $z=0.4$ only, on the 2397 sane + mh-complete
galaxies, the objective of decision D4: $L=\text{score}_A^2+\text{score}_F^2+\text{score}_S^2$
with the first two exactly exp54's and the third the shells/log objective on the
amplitude-pinned profiles, normalised by its value for the nested incumbent
(0.1318 dex) so every term is 1 at the null (null loss 3.1665). L-BFGS-B from
seven starts (the nested incumbent, Stage 1's default, four seeded jitters, the
default at $n_c=2$).

### The fit

**All seven starts converge to the same loss, 2.76396** (−12.7 per cent against
the null), the nested one to 2.76403; the start at $n_c=2$ also returns to
$n_c=0.75$. No parameter at a bound. The solution:

| | $a_0$ | $a_M$ | $a_z$ | $a_{Mz}$ | $m_{1/2}$ | $d$ | $\log f_c$ | $b_c$ | $\log f_e$ | $b_e$ | $n_c$ | $c_e$ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| incumbent (nested) | −2.553 | −0.562 | +0.505 | +0.318 | — | — | — | — | −0.843 | −0.896 | — | 0.796 |
| Stage 2 | −2.889 | −0.938 | +0.934 | +0.755 | 12.93 | 0.81 | −1.372 | −0.377 | −0.832 | +0.230 | 0.751 | 1.308 |

In words: compact deposits at $0.043\,R_{200c}$ (0.015 was the prior expectation,
0.04 the Stage 1 starting value), slightly more compact toward high redshift, with
an exponential-like profile ($n_c=0.75$ — the data reject cuspy deposits, as Stage 1
found); extended deposits at $0.15\,R_{200c}$ with a steeper cored profile than the
incumbent's ($c_e$ 1.31 against 0.80); the compact share falls through one half at
$M_h=10^{12.9}$ over 0.8 dex, giving a compact share of the $z=0.4$ stellar mass of
0.56 (16–84 per cent 0.47–0.61), 0.7 for the lightest haloes and 0.3 for the
heaviest — the halo-mass trend Stage 1 required (0.65 → 0.43 at $s<15$ kpc), with no
scatter at fixed mass (a mean model; `figures/exp63_stage2_channels.png`).

### The gates at z=0.4 (`figures/exp63_stage2_gates.png`)

| gate | baseline (incumbent, exact engine) | Stage 2 | verdict |
|---|---|---|---|
| G2a inner slope 2–4.9 kpc: median (data 1.015) / 16–84 width (data 0.42) | 0.858 / 0.20 | **1.018** / 0.15 | median PASS, width FAIL |
| G2b worst tercile median residual, raw or pinned | 19.3 % | 14.0 % | FAIL (top tercile, 2–3 kpc) |
| G2c plane slope M(30–50)\|M(<30) (truth 1.42) | 1.26 | 1.59 | FAIL (over-steep) |
| G2d mass conservation | 3e-16 | 4e-16 | PASS |
| G2e sizes physical | pass | pass | PASS |
| G2f identifiable | m_half railed | none railed; share 0.33 at 10^13.5 | PASS |

$M_*(<10\,{\rm kpc})$ at $z=0.4$: −1.9 per cent (baseline +3.6); $M_*(<100)$: −1.1 (+0.4);
tier 3: 0.2679 (0.2800). The median inner slope is now the data's; its width is not
— a mean model has one slope per halo history, and the data's spread (0.42) is what
Stage 3's process must supply. The remaining shape failure is in the top halo-mass
tercile, where the model is 11–14 per cent low at 2–3 kpc (`figures/exp63_stage2_residuals.png`):
those haloes get the smallest compact share (0.3) from the mass-only split, and the
data want more. The plane slope overshoots (1.59 vs 1.42) because the mean's
outskirts are now a steep function of inner mass with no scatter about it
(model scatter 0.047 dex against the truth's 0.170; `figures/qa/qa_planes_exp63_stage2_stage2.png`).

### Result 2 — the predictions (never fitted; both samples; `figures/exp63_stage2_predictions.png`)

Median relative bias, per cent, all 2397 / mh-complete fit mask:

| quantity | product | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|---|
| $M_*(<10)$ | baseline | +3.6 / +3.6 | +5.3 / +6.7 | +5.8 / +8.9 | −0.8 / +4.3 | −8.9 / −7.8 |
| | Stage 2 | −1.9 / −1.9 | +0.3 / −0.2 | +0.4 / +1.4 | −8.3 / −3.3 | −21.0 / −16.6 |
| $M_*(<100)$ | baseline | +0.4 / +0.4 | +5.1 / +4.6 | +6.4 / +5.7 | +3.0 / +3.0 | −4.7 / −7.8 |
| | Stage 2 | −1.1 / −1.1 | +5.1 / +5.0 | +7.2 / +8.5 | −0.1 / +7.1 | −14.7 / −7.8 |
| $M_*(50\text{–}100)$ | baseline | −3.5 / −3.5 | +4.6 / −5.4 | +8.7 / −7.4 | +19.7 / −8.4 | +32.4 / −9.5 |
| | Stage 2 | +4.5 / +4.5 | +19.6 / +16.4 | +31.2 / +24.7 | +35.5 / +28.0 | +35.7 / +21.5 |

Tier 3 per epoch: Stage 2 0.268 / 0.276 / 0.287 / 0.307 / 0.357 against the baseline's
0.280 / 0.282 / 0.291 / 0.291 / 0.314.

**The C8 decliner split** — median $\log_{10}$(model/truth) of $M_*(<4.92$ kpc):

| | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 | span |
|---|---|---|---|---|---|---|
| baseline, centre declined (42 %) | +0.052 | +0.029 | +0.005 | −0.052 | −0.153 | 0.205 |
| baseline, did not decline (58 %) | −0.027 | −0.018 | −0.004 | −0.003 | +0.010 | 0.037 |
| Stage 2, centre declined (42 %) | +0.010 | −0.018 | −0.039 | −0.106 | −0.218 | 0.228 |
| Stage 2, did not decline (58 %) | −0.047 | −0.037 | −0.032 | −0.044 | −0.048 | **0.016** |

What it means. Two things, in opposite directions:

1. **For the 58 per cent of galaxies whose centres never declined, the model fitted
   at one epoch predicts the other four to within 0.016 dex of a constant** — the
   "one epoch fitted, four predicted" claim holds for the population the class can
   describe (the plan's failure criterion was 0.03 dex). The declining 42 per cent
   are the class limit, as before (span 0.228; the model has their $z=2$ centres
   0.22 dex too light because it cannot lose mass afterwards — exp54 C8).
2. **The extended channel arrives too early.** At every earlier epoch the model is
   too extended: $M_*(50\text{–}100)$ over-predicted by 16–28 per cent on the fit mask
   (baseline −5 to −10), the amplitude-pinned residual −10 to −25 per cent at
   2–5 kpc, and the $z=2$ amplitude 15 per cent low on the full sample. The model
   credits a satellite's stars to the central when the halo accretes the satellite;
   in TNG they join the central only when the merger completes. The fitted extended
   size law ($b_e=+0.23$, $s_e$ rising toward high redshift, above Stage 1's required
   band at $z>2$) is the fit compensating at $z=0.4$ for that early arrival.

**Does the model reproduce Stage 1's leverage?** Partial Spearman at fixed
$\log M_h$ of the model's compact share against the halo variables, with Stage 1's
measured inner-share correlation in brackets: `late` −0.34 [+0.25], `f_form` +0.34
[−0.26], `logtc` +0.29 [−0.18], `t50` +0.76 [−0.06], `c200c` −0.49 [+0.07]. **Every
sign is wrong.** A split by halo mass at the deposit's time makes early-forming
haloes (massive early) extended and late-forming ones compact; the data say the
reverse, and in particular that haloes still growing fast today (`late` index high)
hold *more* of their stars compactly — which is what an arrival delay produces:
recently accreted halo mass has not yet delivered its extended stars.

### Stage 2b — the accretion-to-deposition delay (in progress)

A one-parameter extension, `tau_d`: the extended deposit of halo mass accreted at
$t'$ is credited at $t'+\tau_d/H(z(t'))$, with $\tau_d$ in Hubble times at accretion;
nothing moves after deposition (not a radial transport), $\tau_d=0$ is the Stage 2
model exactly (asserted), and mass accreted but not yet arrived is "in transit" —
exactly what a central's measured stellar mass excludes. **Fitted free at $z=0.4$ it
is not identifiable**: the nested start returns to the Stage 2 optimum with
$\tau_d$ railed at 0 (loss 2.763963), and the start at $\tau_d=0.5$ lands in a worse
basin (2.7848, three parameters railed) — the single-epoch degeneracy of the time
labels that tech note 04 §4 anticipated; the free fit was stopped after those two
starts. The delay is therefore **set from physics**: dynamical-friction merger times
for the 1:3–1:10 mass ratios that carry most ex-situ mass are 0.2–0.5 Hubble times
at accretion (Boylan-Kolchin et al. 2008), so $\tau_d=0.3$ is held fixed and the
other twelve parameters are refitted at $z=0.4$ (`stage2_fit.py --tau-fixed 0.3`);
the earlier epochs remain predictions and decide whether the delay is kept.

**Stage 2b verdict (2026-08-28): the delay is REJECTED by the predicted epochs.**
With $\tau_d$ fixed at 0.3, two starts (the nested incumbent and Stage 1's default)
converge to the same $z=0.4$ loss, 2.6752 — *better* than Stage 2's 2.7640 — but
only by pushing the extended deposit size to its upper bound ($\log f_e=0$: deposits
as large as the halo, shrinking as $(1+z)^{-2.6}$) and the compact index to its lower
bound ($n_c=0.5$), with the compact share confined below $10^{12.3}$: the fit uses the
delay to keep late-accreted stars *outside* the 148 kpc aperture at $z=0.4$. The
predictions then collapse: $M_*(50\text{–}100)$ −40 per cent at $z=0.7$ and −97 per cent
at $z=2$, $M_*(<100)$ −46 per cent at $z=2$ (fit mask −53), the non-decliners' central
span 0.052 (Stage 2: 0.016), and the assembly-correlation signs unchanged
(`late` −0.37 against the data's +0.25). $\tau_d=0.6$ from the nested start gave 2.892
with the same railing. Both fits were stopped after their verdict was clear
(`outputs/stage2_fit_tau0.3.PARTIAL.npz`, `stage2_fit_tau0.6.PARTIAL.npz`,
`outputs/stage2_eval_tau0.3.log`). What it means: a delay that is a fixed multiple
of the Hubble time removes the *early* extended deposits too (at $z=2$ almost
nothing has arrived), and the $z=0.4$ loss cannot see that. The arrival of ex-situ
stars is not a single lag; it is not a one-parameter repair of this class, and the
Stage 2 twelve-parameter mean is the mean carried into Stage 3. The wrong-signed
assembly correlations remain the open question of the mean model (PROBLEMS P9).

The cleaner test, done afterwards: the delay applied to the **Stage 2 solution
without any refit** (so no compensating size law can enter). Median relative bias
of $M_*(<10)$ on all galaxies and of $M_*(50\text{–}100)$ on the fit mask, $z=0.4\ldots2.0$,
and the partial correlations of the model's compact share with `late`, `f_form`,
`logtc` at fixed halo mass (data: +0.25, −0.26, −0.18):

| $\tau_d$ | $M_*(<10)$ | $M_*(50\text{–}100)$, fit mask | late / f_form / logtc |
|---|---|---|---|
| 0 | −1.9 / +0.3 / +0.4 / −8.3 / −21.0 | +4.5 / +16.4 / +24.6 / +27.9 / +21.3 | −0.34 / +0.34 / +0.29 |
| 0.1 | −1.9 / +0.3 / +0.3 / −8.6 / −21.6 | −5.1 / +0.5 / −0.6 / −10.2 / −24.7 | −0.35 / +0.56 / +0.48 |
| 0.3 | −1.9 / +0.2 / +0.1 / −9.3 / −23.0 | −25.8 / −31.0 / −40.2 / −59.6 / −73.3 | −0.39 / +0.81 / +0.68 |

Even a tenth of a Hubble time swings the outskirts from +16–28 per cent to −0.6 to
−25 per cent — most extended mass is deposited "recently" in Hubble-time units — and
every correlation sign moves further from the data's. The delay is the wrong
mechanism for both symptoms.

### Stage 2c — the split on the halo's growth rate (2026-08-28)

The P9 leverage probe (PROBLEMS.md) found that a compact share
$w_c=\sigma((m_{1/2}-\log M+g\,(\alpha-1))/d)$, with $\alpha={\rm d}\ln M/{\rm d}\ln t$
at the deposit and $g\approx-0.5$ to $-1$, reproduces every measured assembly
correlation by evaluation alone. Fitted (`stage2_fit.py --growth`, thirteen
parameters, nests at $g=0$; five of seven starts at the time of writing), the
$z=0.4$ optimum is 2.7003 (three starts; two others in a basin at 2.829 where the
"compact" channel has become a second extended one at $0.15\,R_{200c}$ — not a
candidate): $g=-0.66$, compact deposits at $0.030\,R_{200c}$ with $n_c=0.79$,
extended at $0.083\,(1+z)^{0.90}R_{200c}$, $m_{1/2}=13.5$ over a broad $d=1.6$
dex. The $z=0.4$ gates are Stage 2's (inner-slope median 1.027, tier 3 0.266,
compact share 0.50 at $10^{13.5}$) and **the assembly correlations now carry the
data's signs and sizes**: `late` +0.29 [data +0.25], `logtc` −0.16 [−0.18],
`f_form` −0.39 [−0.26], `c200c` +0.07 [+0.07] (`early` −0.49 [−0.08] overshoots).

**But the predicted epochs are far worse than Stage 2's**: $M_*(50\text{–}100)$ on the
fit mask +24 / +43 / +73 / +93 per cent at $z=0.7$–$2$ (Stage 2: +16 to +28),
$M_*(<10)$ −35 per cent at $z=2$ (Stage 2: −17), tier 3 at $z=2$ 0.454 (0.357), and
the non-decliners' central span 0.115 dex (0.016). Why: at high redshift *every*
halo grows fast ($\alpha\approx2$–3), so an absolute growth-rate split makes all
early deposits extended and the $z\ge1.5$ galaxies too diffuse and too light in
the centre. The data's correlation is with growth *relative to other haloes at the
same time* — mergers are growth in excess of the typical. **Stage 2d** replaces
$\alpha-1$ by $\alpha-\bar\alpha(t)$, $\bar\alpha$ the population-median rate at that
time (a fixed function tabulated from the sample and stored with the fit, not a
parameter); `stage2_fit.py --growth-rel`, running.

**Stage 2d verdict (2026-08-28): the same trade, slightly softer.** With the split on
$\alpha-\bar\alpha(t)$ ($\bar\alpha$ 0.98–2.73 across the nodes), the nested start
reaches 2.7047 with $g_{\rm rel}=-0.49$, compact deposits at $0.028\,R_{200c}$
($n_c=0.76$), extended at $0.072\,(1+z)^{0.99}R_{200c}$, and the assembly correlations
again carry the data's signs (`late` +0.35 [+0.25], `logtc` −0.33 [−0.18], `f_form`
−0.57 [−0.26], `c200c` +0.17 [+0.07]). The predictions are better than 2c's and
worse than Stage 2's: $M_*(50\text{–}100)$ on the fit mask +21 / +38 / +63 / +79 per
cent, $M_*(<10)$ −29 per cent at $z=2$, tier 3 at $z=2$ 0.417, the non-decliners'
central span 0.080 dex (2c 0.115; Stage 2 0.016). The fit was stopped after the
nested start (`outputs/stage2_fit_growthrel.PARTIAL.npz`, `stage2_eval_growthrel.log`);
the other starts could only find a different $z=0.4$ optimum, and the verdict is
about the predicted epochs, which no $z=0.4$ start can see.

**The decision, and what the three fits say together.** Every split that reproduces
the $z=0.4$ assembly correlations does so by making the extended channel's early
deposits large ($b_e\approx+0.9$ to $+1.0$: extended sizes near $R_{200c}(1+z)$, a
roughly constant physical size), and that is exactly what breaks the earlier
epochs. The mass-only split (Stage 2) keeps the evolution — the describable 58 per
cent of galaxies evolve to within 0.016 dex of a constant central error — and gets
the correlations backwards. **A split on $(M,\ \dot M)$ at the deposit cannot satisfy
both**; the $z=0.4$ correlation must come from something the extended channel's
*arrival* or *size* knows about the halo's later history, which a single-epoch fit
cannot see (P9 stays open with this as its measured content). **The Stage 2
twelve-parameter mean is the exp63 mean**: it is the one that predicts the other
four epochs, which is what the plan asked a single-epoch model to do.

## Stage 3 — noise inside the deposition process (2026-08-28)

`process3.py` puts three physically named random modifications INSIDE the history,
drawn once per galaxy and shared by every epoch (so every draw is a coherent
multi-epoch galaxy) on top of the Stage 2 mean (frozen): **lumpy accretion** — the
extended channel's deposition is a compound-Poisson process with `n_eff` events per
e-fold of halo growth (mean preserved; variance by Campbell's theorem, checked
against 300 draws to 7.5 per cent); an **efficiency history** — `log10 eps` gets a
zero-mean Gaussian process in ln t of amplitude `sigma_eta` and correlation length
`ell` (mean-preserving); **size offsets** — per-galaxy normal offsets `sigma_sc`,
`sigma_se` on the two size laws. Five parameters. `stage3_process.py`: targets,
leverage sweep, held-out fit, gates, predictions (log `outputs/stage3_process.log`).

**Targets and the null** ($z=0.4$, 2397 galaxies): the truth's scatter about the plane
regression is 0.170 dex for $M(30\text{–}50)|M(<30)$, 0.204 for $M(50\text{–}100)|M(<30)$,
0.173 for $R_{50}|M_*$; the mean model's is 0.047 / 0.061 / 0.053, its energy
distance to the truth 3.9 / 3.7 / 5.5 × the split-half floor, and its tier-2e widths
0.82–0.87 of the truth's — the conditional-mean under-dispersion exp61 Stage 4
measured, now on a model whose mean shape is right.

**Leverage sweep** (each source alone, four draws, `figures/exp63_stage3_sweep.png`):
all four sources can widen a kpc plane by ≥ 0.03 dex while moving no median by
more than 2 per cent — lumpy accretion at $n_{\rm eff}\le5$ (a handful of events per
e-fold is what the outskirts' width needs; 20 is too smooth), the efficiency history
at 0.2 dex, the compact size at 0.2 dex, the extended size at 0.1–0.2 dex. Nothing
dropped.

**Held-out fit** (Nelder–Mead on the sum of plane E/floor and tier-2e W1 ratios,
common random numbers, eight draws; calibrate on one random half, score on the
other):

| | $n_{\rm eff}$ | $\sigma_\eta$ [dex] | $\ell$ [ln t] | $\sigma_{sc}$ | $\sigma_{se}$ | held-out E/floor 30–50 / 50–100 / R50 (mean alone) | tier-2e widths |
|---|---|---|---|---|---|---|---|
| A→B | 6.16 | 0.143 | 0.29 | 0.150 | 0.153 | 1.27 / 1.20 / 1.27 (2.70 / 2.85 / 3.98) | 1.00, 0.94, 0.97, 1.02 |
| B→A | 6.25 | 0.139 | 0.30 | 0.165 | 0.140 | 1.25 / 1.28 / 1.19 (2.64 / 2.86 / 3.76) | 1.00, 0.94, 0.96, 0.96 |
| all | 5.21 | 0.148 | 0.33 | 0.170 | 0.132 | objective 11.94 (mean alone 25.42) | — |

The two calibrations agree to a few per cent on every parameter. **G3a passes on
both halves** (kpc-plane E/floor 1.20–1.28 against the 1.5 rule; exp60's output
layer reached 1.69–2.17) and **G3b passes on both** (tier-2e widths within 0.06 of 1,
from draws — the first time this programme has scored tier 2e on a generative
layer, closing open question C16). In words: about five significant accretion
events per e-fold of halo growth, a 0.15 dex efficiency history correlated over a
third of an e-fold of time, and 0.13–0.17 dex per-galaxy size offsets reproduce the
$z=0.4$ population planes to within 1.2–1.3× the noise floor of the simulation's
own sample (`figures/exp63_stage3_planes.png`).

**Predictions — the same draws at every epoch** (`figures/exp63_stage3_predictions.png`):

| tier-2e width, draws/truth [mean alone] | $M(<10)$ | $M(<30)$ | $M(30\text{–}50)$ | $M(50\text{–}100)$ |
|---|---|---|---|---|
| z=0.4 | 1.01 [0.82] | 0.95 [0.83] | 0.97 [0.86] | 1.00 [0.87] |
| z=0.7 | 1.00 [0.83] | 0.97 [0.87] | 0.98 [0.85] | 0.98 [0.83] |
| z=1.0 | 1.01 [0.85] | 1.01 [0.92] | 0.96 [0.84] | 0.98 [0.84] |
| z=1.5 | 1.05 [0.93] | 1.09 [0.99] | 1.01 [0.87] | 1.02 [0.87] |
| z=2.0 | 1.00 [0.93] | 1.03 [0.96] | 1.05 [0.90] | 1.00 [0.85] |

The widths fitted at $z=0.4$ hold at every earlier epoch to within 9 per cent —
because the noise is in the history, not added per epoch. Tier 2c: the size-rank
persistence of the same galaxy between $z=0.4$ and $z=2$ is **+0.68 ± 0.01** from the
draws against the truth's +0.33 and exp60's +0.97 — half the excess coherence of a
per-epoch layer removed, by construction rather than by a coherence prior. The
declining-centre fraction of the draws is **0** (truth 41.8 per cent): every draw is
a deposition-only galaxy, monotone in time — the class limit stated by a prediction
(PROBLEMS P8), and the reason the drawn $z=2$ sizes sit above the truth's in the
tier-2c plane.
