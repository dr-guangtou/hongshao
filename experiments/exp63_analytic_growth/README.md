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
   at one epoch holds a *constant* central error across the other four, to 0.016
   dex** (`figures/exp63_stage2_c8.png`) — constant at −0.03 to −0.05 dex, i.e. the
   inner 5 kpc is 7–11 per cent light at every epoch alike, not right. That is the
   sense in which the "one epoch fitted, four predicted" claim holds for the
   population the class can describe (the plan's failure criterion was 0.03 dex on
   the span); it concerns the innermost aperture only — the whole population's
   amplitude at z ≥ 1.5 is worse than the incumbent's (point 2). The declining 42 per cent
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
cent of galaxies keep a constant central error (to 0.016 dex, at a −0.03 to −0.05
dex offset) — and gets
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

## Stage 4 — the standard battery on three products (2026-08-28)

`stage4_battery.py`: `hongshao.qa.evaluate` on the same 2397 galaxies for
**baseline** (the incumbent's seven numbers on the exact integral), **stage2** (the
two-channel mean) and **stage2+p** (the mean with the Stage 3 process; the battery's
paired tiers score the FIRST draw — a realisation, not a prediction — and eight
draws feed the plane figure and the tier-2e-from-draws table). Figures
`figures/qa/qa_*_exp63_stage4_{baseline,stage2,stage2+p}.*`; log
`outputs/stage4_battery.log`.

Median relative bias, per cent, all / fit mask ($M_*(<10)$ and $M_*(<100)$):

| product | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| $M_*(<10)$ baseline | +3.6 / +3.6 | +5.3 / +6.7 | +5.8 / +8.9 | −0.8 / +4.3 | −8.9 / −7.8 |
| $M_*(<10)$ stage2 | −1.9 / −1.9 | +0.3 / −0.2 | +0.4 / +1.4 | −8.3 / −3.3 | −21.0 / −16.6 |
| $M_*(<10)$ stage2+p (one draw) | −6.2 / −6.2 | −4.7 / −5.3 | −5.1 / −4.3 | −12.3 / −8.2 | −25.6 / −21.5 |
| $M_*(<100)$ baseline | +0.4 / +0.4 | +5.1 / +4.6 | +6.4 / +5.7 | +3.0 / +3.0 | −4.7 / −7.8 |
| $M_*(<100)$ stage2 | −1.1 / −1.1 | +5.1 / +5.0 | +7.2 / +8.6 | −0.1 / +7.1 | −14.7 / −7.8 |
| $M_*(<100)$ stage2+p (one draw) | −5.0 / −5.0 | +0.7 / +0.5 | +2.6 / +2.8 | −3.9 / +3.0 | −18.0 / −11.5 |

Tier 3 (median max|rel| beyond 5 kpc): baseline 0.280 / 0.282 / 0.291 / 0.291 / 0.314;
stage2 0.268 / 0.276 / 0.287 / 0.306 / 0.357; one draw 0.387–0.432 (a realisation
carries the population's own scatter into a paired statistic; this number is not a
model error, exp41's lesson).

**Tier 2e from draws (C16 closed) and the planes, per epoch** — 16–84 width of the
drawn population over the truth's, and the energy-distance ratio to the split-half
floor on the full sample (mean alone in brackets):

| z | width $M(<10)$ / $M(<30)$ / $M(30\text{–}50)$ / $M(50\text{–}100)$ | E/floor $M(30\text{–}50)|M(<30)$ / $M(50\text{–}100)|M(<30)$ / $R_{50}|M_*$ |
|---|---|---|
| 0.4 | 1.01 / 0.96 / 0.97 / 1.00 [0.82 / 0.83 / 0.86 / 0.87] | 1.68 / 1.53 / 1.68 [3.88 / 3.73 / 5.50] |
| 0.7 | 1.00 / 0.98 / 0.98 / 0.97 [0.83 / 0.87 / 0.85 / 0.83] | 2.08 / 1.63 / 3.44 [4.65 / 4.56 / 7.08] |
| 1.0 | 1.01 / 1.02 / 0.96 / 0.98 [0.85 / 0.92 / 0.84 / 0.84] | 2.61 / 1.98 / 4.44 [4.86 / 4.65 / 7.56] |
| 1.5 | 1.06 / 1.09 / 1.01 / 1.02 [0.93 / 0.99 / 0.87 / 0.87] | 5.41 / 4.10 / 7.31 [7.92 / 6.47 / 10.12] |
| 2.0 | 1.00 / 1.04 / 1.04 / 0.99 [0.93 / 0.96 / 0.90 / 0.85] | 7.48 / 7.54 / 10.02 [9.18 / 9.10 / 12.79] |

(The full-sample E/floor is higher than Stage 3's held-out 1.2–1.3 because the
split-half floor shrinks with sample size while the model's residual distance does
not; the two are the same comparison at two sample sizes. exp60's Option A reached
1.69 / 2.17 on its own sample.) The widths hold at every epoch; the plane distances
grow with redshift because the *mean* is wrong there — the extended channel's
early excess and the missing decliners (`figures/qa/qa_planes_exp63_stage4_stage2+p.png`:
at $z\ge1$ the draws sit above the truth in $M(30\text{–}50)|M(<30)$ exactly where
Result 2 said the mean does).

**One caveat found here (PROBLEMS P10).** The draws' medians sit 4–5 per cent below
the mean model's at every epoch ($M_*(<10)$ at $z=0.4$: −6.2 against −1.9 per cent).
The process is mean-preserving in linear mass (the Poisson thinning and the
efficiency history both have unit expectation), so the median of the skewed drawn
distribution falls below the mean — a property of any multiplicative noise, not a
fit defect; a median-preserving convention is a one-line change and is left as a
reporting decision.

## Stage 5 — the single-epoch round: z=0.4 only, judged by the binned average CoGs (2026-08-28)

**The user's direction after Stage 2.** Stage 2 is not a clear $z=0.4$ improvement
by the binned average curves of growth (`qa_bins` / `qa_bins_ms`): the top
halo-mass tercile is $-12$ per cent at 2–4 kpc, and its extrapolation to earlier
epochs tilts. So this round looks at **$z=0.4$ only**, judges every candidate by
the average CoGs in halo- and stellar-mass terciles and the planes (the *visual
gate*, `single_epoch.py --gate`), and does not quote any $z>0.4$ number until
$z=0.4$ is right. Every fit in this round **freezes the four time exponents** —
$a_z$, $a_{Mz}$, $b_e$ at the incumbent's values ($+0.505$, $+0.318$, $-0.896$) and
$b_c$ at 0 (the incumbent has no compact channel; 0 means "a fixed fraction of
$R_{200c}$ at the deposit") — because a single epoch cannot constrain a time
exponent, and Stage 2's free exponents were what tilted its extrapolation
(`stage2_fit.py --freeze`). The incumbent it is compared with was fitted at five
epochs; this is said wherever it is compared.

**What the gate reads.** For each product at $z=0.4$: the median
(model$-$data)/data per halo-mass tercile at 2, 3.7, 4.9, 10, 33, 103 and 148 kpc,
raw and amplitude-pinned (model scaled to the data's $M_*(<100)$); the worst
central ($\le5$ kpc) and outskirt ($\ge50$ kpc) tercile residual; the rms of the
tercile residual over all radii ("rms(h)"); the two kpc planes' slope and scatter;
the same at 148 kpc by stellar-mass tercile; the loss components. One overlay
figure per comparison, `figures/exp63_single_epoch_gate*.png` (median CoGs and
residuals by halo-mass tercile; residuals by stellar-mass tercile).

### 5a — the lever sweep by evaluation (no fit; `single_epoch.py --sweep`, log `outputs/single_epoch_sweep.log`)

Standing method: sweep a candidate's gate leverage by evaluation before spending
a fit; a slice may disqualify, never promote. Every lever moved one at a time from
the Stage 2 solution (so the loss can only rise); the size exponents `g_c`/`g_e`
have their `log_f` compensated so only the *trend* with halo mass changes. Top
halo-mass tercile mean raw residual at 2–5 kpc, worst central and outskirt
tercile residuals, rms over terciles and radii, and the (unrefitted) loss:

| lever | top-tercile centre | worst centre | worst outskirt | rms(h) | loss |
|---|---|---|---|---|---|
| Stage 2 (base) | −12.5 % | 14.0 % | 2.4 % | 3.89 | 2.764 |
| `w_min` = 0.1 (compact share floored) | **−6.6** | **8.4** | 2.6 | **3.26** | 2.868 |
| `w_min` = 0.2 | −0.8 | 7.2 | 6.1 | 6.82 | 3.197 |
| `g_c` = −1/3 (compact size fixed in kpc) | −20.3 | 47.4 | 2.4 | 13.30 | 3.604 |
| `g_c` = +0.15 | −9.2 | 18.8 | 2.6 | 4.95 | 2.934 |
| `g_e` = ±0.15 | −12.8 / −11.9 | 14.1 / 13.5 | 3.6 | 4.66 / 3.84 | 2.837 / 2.827 |
| `c_e` = 0.5 (extended core steeper) | +0.8 | 10.3 | 7.9 | 8.49 | 4.402 |
| `n_c` = 1.0 | −9.9 | 11.6 | 2.4 | 3.62 | 2.787 |
| `m_half` = 13.3 | +1.2 | 10.0 | 5.4 | 7.67 | 3.299 |
| `d_split` = 0.4 | −4.1 | 14.4 | 6.9 | 6.42 | 3.094 |
| extended kernel Sérsic $n_e$ = 1 | −9.8 | 11.7 | 3.9 | **3.17** | 2.918 |
| extended kernel Sérsic $n_e$ = 2 | **−6.0** | **8.0** | **1.7** | 3.85 | 3.010 |
| extended kernel Sérsic $n_e$ = 4 | +3.6 | 11.5 | 5.4 | 7.53 | 3.764 |

What it means. Two levers reach the top tercile's centre without breaking the
other two terciles or the outskirts, and are promoted to fits: **a floor on the
compact share** (`w_min`; the mass-only logistic sends the heaviest haloes' share
toward zero while Stage 1 wants 0.43 in the top tercile), and **the extended
kernel's family** (a Sérsic deposit of index 1–2 in place of the cored
`gompertz_log`: more central mass from the extended stars themselves, and the
best outskirts of any row). Two are disqualified: **a compact size fixed in
physical kpc** (`g_c` = −1/3, the hypothesis that the model's $s_c\propto R_{200c}(t')$
makes the top tercile's compact deposits too large) makes the top tercile −20 per
cent and the low tercile −47 per cent — because the size is set by the halo mass
*at the deposit*, which is $10^{12}$–$10^{12.5}$ for most compact stars whatever
the final mass, the mass exponent acts on the early deposits, not on the final
tercile; and the extended size's mass exponent `g_e` moves nothing. `c_e`,
`m_half`, `d_split` reach the centre only by breaking the outskirts. Note that the
gate and the loss disagree at `n_c`: the gate prefers 1.0 (rms 3.62 against 3.89),
the loss 0.75 — the loss is a mean over galaxies of per-galaxy residuals, the gate
a median over a tercile; recorded, not acted on.

### 5b — the fits with frozen exponents, at the gate (`single_epoch.py --gate`, logs `outputs/single_epoch_gate_*.log`)

Every fit: $z=0.4$ only, the 2397 sane + mh-complete galaxies, $a_z$, $a_{Mz}$,
$b_c$, $b_e$ frozen as above, L-BFGS-B from three starts (seven for `_frozen`),
each converged (no start hit the evaluation cap). "loss" is always the
production D4 loss, whatever objective the fit used, so the column compares.
Median (model$-$data)/data by halo-mass tercile, raw, at 2 / 3.7 / 4.9 kpc and
the worst outskirt ($\ge50$ kpc) residual; rms over terciles and radii; the
slope of $M_*(30\text{–}50)|M_*(<30)$ (truth 1.42):

| product (fit file tag) | starts agree | low tercile 2/3.7/4.9 kpc | mid | **top** | worst outskirt | rms(h) | plane slope | D4 loss |
|---|---|---|---|---|---|---|---|---|
| incumbent, exact engine (five-epoch fit) | — | +17.8 / −0.6 / −2.5 | +19.3 / +2.9 / +1.9 | +9.5 / −2.1 / −1.2 | 4.0 | 4.98 | 1.26 | 3.166 |
| Stage 2 (free exponents) | 7/7 | −2.4 / −6.5 / −5.3 | −2.6 / −6.1 / −4.0 | **−12.2 / −13.4 / −10.5** | 2.4 | 3.89 | 1.59 | 2.764 |
| `_frozen` | 7/7 at 2.86082 | +4.4 / −6.0 / −5.6 | +3.0 / −6.0 / −5.6 | **−11.1 / −17.0 / −14.1** | 2.6 | 4.52 | 1.51 | 2.861 |
| `_frozen_sersic` (extended kernel Sérsic, $n_e$ = 2.4) | 3/3 | −3.3 / −6.3 / −4.1 | −3.3 / −6.1 / −4.1 | **−10.8 / −13.1 / −10.9** | 2.5 | 3.76 | — | 2.815 |
| `_frozen_wmin` (compact-share floor) | `w_min` railed at 0 | = `_frozen` | | | | | | 2.861 |
| `_frozen_outer` (+ outskirt term $M_*(50\text{–}148)$) | 3/3 | +2.3 / −8.8 / −8.4 | +3.1 / −6.5 / −6.1 | **−8.0 / −15.2 / −12.0** | 3.2 | 4.37 | 1.48 | 2.869 |
| `_frozen_binned` (+ the tercile-median term) | 3.29 / 3.32 / 3.48 | +10.6 / −2.9 / −3.2 | +10.1 / −1.4 / −1.7 | **−2.8 / −10.7 / −8.0** | **1.8** | **3.16** | **1.43** | 2.891 |

What it means, in order:

1. **Freezing the exponents costs the top tercile.** With the incumbent's time
   exponents the two-channel mean is *worse* at the gate than Stage 2 (rms 4.52
   against 3.89; top tercile −17 per cent at 3.7 kpc). Stage 2's free exponents
   were doing real work at $z=0.4$ — the price of not tilting the extrapolation.
2. **The production loss is nearly blind to what the gate reads.** In the top
   tercile the per-galaxy rms of the $2$–$5$ kpc log residual is 0.18 dex (the
   intrinsic diversity of the centre, memory `halo-history-knows-a-little-about-the-core`)
   against a median offset of 0.065 dex; both the galaxies whose centres
   declined and those that did not are low there (Stage 2: −7 and −15 per cent).
   A loss that is a mean over galaxies of per-galaxy residuals is dominated by
   the scatter it cannot reduce, so it trades the binned offset away for gains
   elsewhere: the compact-share floor `w_min`, which improved the gate by
   evaluation, is railed at zero by the fit; the Sérsic extended kernel wins the
   loss (2.815) and leaves the top tercile where it was.
3. **Pricing the binned average CoG directly (`--objective binned`) moves the
   gate most**: rms 3.16, the best outskirts (1.8 per cent), the plane slope on
   the truth (1.43 against 1.42), for 1 per cent of the production loss. Its
   three starts do not agree (3.29–3.48; a median is piecewise-smooth, so
   L-BFGS-B on finite differences stalls in different places — the best start
   is reported and the spread is stated). What it cannot do: hold 2 kpc and
   4 kpc at once — the low and mid terciles rise to +10 per cent at 2 kpc as the
   top tercile's 4 kpc improves. That is the incumbent's 3–5 kpc dip (Stage 0,
   Result 0) now inside the compact channel: one Sérsic index and one
   $R_{200c}$-scaled size for every compact deposit give a fixed curvature over
   2–5 kpc, and the data's median curvature differs between terciles.
4. The outskirt term does nothing at the gate: by halo-mass tercile the
   outskirts were already within 2–3 per cent, and the $\pm10$–17 per cent at
   148 kpc by *stellar-mass* tercile is the regression-to-the-mean of a mean
   with no scatter about it (the `qa_bins_ms` caveat) — a property of every mean
   model here, incumbent included (+20 / −10), and not something an objective
   term on the mean can remove.

### 5c — the compact channel in physical kpc, and the size levers (2026-08-28)

Every product of 5b shares one residual: 2 kpc high relative to 3.7–4.9 kpc, in
every tercile, whatever the objective or the extended kernel — the compact
channel's *curvature* over 2–5 kpc is wrong. In Stage 2's form the compact size
is $s_c = f_c\,R_{200c}(t')$: tied to the halo's radius at the deposit, it spreads
one galaxy's compact deposits over a factor ~10 in size along its history, while
Stage 1 read a *narrow* compact mode at a fixed few kpc at every halo mass. Two
ways to test that, both fitted with the frozen exponents:

- `--compact-kpc`: $s_c = 10^{\log f_c}$ kpc, a physical scale (`model2.Spec2(compact_in_kpc=True)`;
  the nested incumbent is unchanged);
- the size levers on the $R_{200c}$ form: $s = f\,(1+z)^b R_{200c}(t')\,(M(t')/10^{13})^{g}$
  for both channels (`g_c`, `g_e`; $g=-1/3$ cancels $R_{200c}\propto M^{1/3}$), plus the
  compact-share floor `w_min` (`--levers g_c,g_e,w_min`).

| product (tag) | starts (binned-objective loss) | low 2/3.7/4.9 kpc | mid | top | worst outskirt | rms(h) | plane slope | D4 loss |
|---|---|---|---|---|---|---|---|---|
| `_frozen_binned` (5b, for reference) | 3.29 / 3.32 / 3.48 | +10.6 / −2.9 / −3.2 | +10.1 / −1.4 / −1.7 | −2.8 / −10.7 / −8.0 | 1.8 | 3.16 | 1.43 | 2.891 |
| `_frozen_kpc` (production loss) | 3/3 agree | −6.5 / −5.9 / −4.0 | −4.6 / −3.6 / −1.8 | −13.1 / −11.1 / −8.1 | 2.7 | 3.65 | 1.40 | **2.817** |
| `_frozen_binned_kpc` | 2.95 / 2.98 / 2.99 | −0.2 / −1.7 / −1.1 | +2.6 / +1.8 / +2.5 | **−2.8 / −2.2 / −1.0** | 2.9 | **1.50** | 1.34 | 2.857 |
| `_frozen_binned_levers` ($R_{200c}$ form + `g_c,g_e,w_min`) | 2.85 (cap) / 3.47 / fail | +1.1 / −2.0 / −0.4 | +2.5 / −0.8 / +0.5 | **+0.4 / −3.7 / −1.4** | **1.7** | **1.16** | **1.44** | **2.796** |
| `_frozen_sersic_binned_levers` | 2.95 / 2.96 / 2.96 (cap) | +1.0 / −4.0 / −2.6 | — | — | 1.8 | 1.50 | 1.44 | 2.858 |
| `_frozen_binned_kpc_levers` (kpc form + `g_c,g_e,w_min`) | 2.89 / fail / 2.98 (cap) | −0.5 / −3.2 / −1.6 | +3.2 / +0.9 / +2.4 | **−1.1 / −1.3 / −0.7** | 1.8 | 1.41 | **1.42** | 2.811 |

("cap": the start used all 3000 evaluations; "fail": jitter0 started in a region
where the model does not evaluate and stopped at once.)

**What it means.**

1. **The compact size must not scale with the halo's mass at the deposit.** Sized
   in kpc, the two-channel mean under the binned objective puts **every halo-mass
   tercile within ±3 per cent at every radius** (rms 1.50; Stage 2: 3.89; the
   five-epoch incumbent: 4.98) — the top tercile's 2–4 kpc goes from −12 to −2
   per cent, the outskirts stay within 3 per cent, and the production D4 loss
   (2.857) is *lower* than the frozen $R_{200c}$-form fit's (2.861), i.e. this is
   not a trade against the per-galaxy fit. The levers fit reaches the same
   place from the other side: it chooses $g_c=-0.45$ (a compact size
   $\propto M^{-0.45}R_{200c}\approx M^{-0.12}$, nearly mass-free), $g_e=+0.08$
   (the extended size stays $\propto R_{200c}$ — the halo-mass-dependent extended
   size the user asked about is *not* needed beyond $R_{200c}$'s own scaling), and
   `w_min` = 0.12; rms 1.16, plane slope 1.44 against the truth's 1.42, and the
   best production loss of any frozen fit (2.796; Stage 2 with four free time
   exponents: 2.764). The 5a slice that "disqualified" $g_c=-1/3$ was wrong for
   the reason recorded in `doc/lessons.md`: a slice from another optimum cannot
   judge a reparameterisation whose other parameters all move. The third
   direction agrees: with the compact size already in kpc, the levers fit
   chooses $g_c=-0.07$ and `w_min` = 0.02 — no residual mass dependence and no
   floor are wanted once the size is physical — and puts the plane slope on the
   truth (1.42) with the top tercile within 1.3 per cent at 2–5 kpc (rms 1.41,
   D4 loss 2.811). The three products are one result seen three ways.
2. **What the channels are, physically** (mass-weighted medians of the $z=0.4$
   stars; `single_epoch` numbers): a compact channel of 2.6–4 kpc with an
   exponential-like profile ($n_c$ 0.9–1.9), deposited at a median $z\approx2$–2.6,
   holding 17 per cent (`_frozen_binned_kpc`) to 44 per cent
   (`_frozen_sersic_binned_levers`) of the stars — the share falling with halo
   mass in every product (e.g. 0.42 / 0.33 / 0.23 across the terciles for
   `_frozen_binned_levers`; Stage 1's $s<15$ kpc share: 0.66 / 0.55 / 0.42) — and an
   extended channel of 24–39 kpc deposited at a median $z\approx1.2$–1.5, with a
   core shape 1.1–1.5. The levers fits switch sharply: $d$ = 0.12–0.14 dex at
   $m_{1/2}=10^{12.4-12.6}$ — deposits laid down while the halo was below
   $\sim10^{12.5}$ are compact, everything after is extended (with a 12 per cent
   floor), which is the in-situ-core-then-accretion picture in one number.
3. **At one epoch the compact share is degenerate with the extended kernel's
   inner shape.** Three products within 0.35 of each other in rms(h) hold 17, 33
   and 44 per cent of the stars in the compact channel, compensating with the
   extended kernel's core. The $z=0.4$ curves cannot separate them; the earlier
   epochs can (the compact stars are in place by $z\approx2$, the extended ones
   arrive later) — that is the first thing the extrapolation must be asked, *after*
   the user has read these $z=0.4$ figures.
4. The stellar-mass-binned view (`exp63_single_epoch_gate_levers.png`, bottom
   row) is unchanged in shape for every product — +20 / −5 / −10 per cent at
   148 kpc — because it is the regression-to-the-mean of a conditional mean
   with 0.09 dex of plane scatter against the truth's 0.17; the amplitude-pinned
   curves are within ±5–8 per cent. That is the stochastic layer's job, not the
   mean's.

Figures: `figures/exp63_single_epoch_gate_{frozen,binned,sersic_binned,kpc,levers}.png`;
tables in `outputs/single_epoch_gate_*.log`; the sweep in `outputs/single_epoch_sweep.log`.

### 5d — the extrapolation of the two candidates, opened after the user read the z=0.4 gate (2026-08-29)

The user's reading of the z=0.4 QA figures: greatly improved by the average CoGs
and the planes; Option 2 (`_frozen_binned_levers`) for performance, Option 1
(`_frozen_binned_kpc`) as the simple back-up (it is 3–5 per cent low in total
stellar mass above $10^{13.7}$ — the incumbent's tilt, slightly worse; Option 2
halves it). With that, `stage2_fit.py --eval-only` was run on both: fitted at
$z=0.4$ only, the frozen exponents the incumbent's, every other epoch a
prediction; the incumbent it is compared with was fitted at all five epochs.
Median relative bias, whole sample / fit mask (`outputs/stage2_eval_frozen_binned_{levers,kpc}.log`,
`figures/exp63_stage2_predictions_frozen_binned_{levers,kpc}.png`):

| quantity | product | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|---|
| $M_*(<10)$ | incumbent | +3.6 / +3.6 | +5.3 / +6.7 | +5.8 / +8.9 | −0.8 / +4.3 | −8.9 / −7.8 |
| | Option 2 | +0.9 / +0.9 | +2.6 / +1.6 | +4.5 / +3.3 | +2.2 / +1.5 | **−2.4** / −4.4 |
| | Option 1 | 0.0 / 0.0 | +0.9 / +1.2 | +2.0 / +2.8 | −2.3 / +0.5 | −7.4 / −7.9 |
| $M_*(<100)$ | incumbent | +0.4 / +0.4 | +5.1 / +4.6 | +6.4 / +5.7 | +3.0 / +3.0 | −4.7 / −7.8 |
| | Option 2 | 0.0 / 0.0 | +4.8 / +4.4 | +6.1 / +5.5 | +1.1 / +2.8 | −8.3 / −9.0 |
| | Option 1 | −0.1 / −0.1 | +4.8 / +4.2 | +6.1 / +5.4 | +2.5 / +2.7 | −5.6 / −8.4 |
| $M_*(50\text{–}100)$ | incumbent | −3.5 / −3.5 | +4.6 / −5.4 | +8.7 / −7.4 | +19.7 / −8.4 | +32.4 / −9.5 |
| | Option 2 | +4.8 / +4.8 | +4.3 / +1.4 | −5.7 / −8.3 | **−31.8** / −27.2 | **−57.7** / −46.9 |
| | Option 1 | +0.8 / +0.8 | +4.1 / −3.7 | +2.3 / −10.5 | −3.7 / −20.4 | −7.1 / −32.5 |

Tier 3 (median max |rel| beyond 5 kpc): incumbent 0.280 / 0.282 / 0.291 / 0.291 / 0.314;
Option 2 0.267 / 0.272 / 0.286 / 0.299 / 0.314; Option 1 0.269 / 0.275 / 0.287 / 0.296 / 0.315.
C8 (median log model/truth of $M_*(<4.92)$, whole population span over the five
epochs): incumbent 0.058, Option 2 0.058, Option 1 0.087; the non-decliners' span
0.037 / 0.032 / 0.028. Plane $M(30\text{–}50)|M(<30)$ slope at $z=2$ (truth 1.43):
incumbent 1.26, Option 2 **2.70**, Option 1 **1.44** (Option 1 is on the truth's
slope at every epoch, 1.34–1.44 against 1.42–1.67).

**What it means.** Both beat the five-epoch incumbent at the centre at every
epoch — Option 2's $M_*(<10)$ is within ±4.5 per cent at all five, against the
incumbent's −9 at $z=2$ — and both fail differently in the outskirts. Option 2's
extended channel arrives far too late/compact at $z\ge1.5$ ($M_*(50\text{–}100)$
−32 and −58 per cent; the $z=2$ plane slope 2.7): the sharp switch at
$10^{12.4}$ with the incumbent's $b_e=-0.9$ leaves the early extended deposits
tiny. Option 1 keeps the outskirts and the plane slopes at every epoch on the
whole sample but tilts at the centre ($M_*(<10)$ −7 per cent at $z=2$, C8 span
0.087). Neither is a five-epoch model as fitted; both have the structure to be
one. The $z=2$ gate on the whole sample (`single_epoch.py --gate --epoch 4`,
`figures/exp63_single_epoch_gate_z2_from_z04.png`) shows the shared limit: the
top tercile's 2 kpc is −40 to −45 per cent at $z=2$ for both (incumbent −17), the
class-limit decline problem (P8) seen from $z=2$'s side.

**Decision (the user's rule: if either extrapolates well or has the potential,
fit the other epochs independently).** Both qualify. Stage 5e fits each option at
$z=0.7$, 1.0, 1.5 and 2.0 alone (`stage2_fit.py --epoch k`: that epoch's data on
that epoch's sane + mh-complete mask, its $\sigma_A$ and halo-mass terciles; the
integral truncated at its anchor, so each fit sees only the MAH before its
epoch), judged by the gate at that epoch (`single_epoch.py --gate --epoch k`) and
by how the parameters move with epoch — a parameter that must change across
epochs is a time law the frozen exponents do not carry.

### 5e — each epoch fitted alone (2026-08-29)

`stage2_fit.py --epoch k` fits the same model at one epoch: that epoch's data on
that epoch's sane + mh-complete mask (1780 / 1435 / 1144 / 839 galaxies at
$z=0.7$ / 1.0 / 1.5 / 2.0), its $\sigma_A$ and its halo-mass terciles for the
binned term; the integral truncated at that epoch's anchor, so the fit sees only
the MAH before its epoch. Time exponents frozen as in every Stage 5 fit; binned
objective; three starts. Judged by the gate at that epoch, on both samples
(`single_epoch.py --gate --epoch k`; `figures/exp63_single_epoch_gate_e{1,2,3,4}.png`;
logs `outputs/single_epoch_gate_e*.log`). Gate rms over terciles × radii on the
fit mask, with the worst central / outskirt tercile residual:

| judged at | incumbent (5-epoch fit) | Option 1 from its z=0.4 fit | Option 1 fitted at this epoch | Option 2 fitted at this epoch |
|---|---|---|---|---|
| z=0.4 (2397) | 4.98 (19.3 / 4.0) | **1.50** (3.9 / 2.9) | — | 1.16 (3.7 / 1.7) |
| z=0.7 (1780) | 6.54 (16.6 / 7.4) | 5.07 (14.6 / 6.3) | **1.05** (3.2 / 1.9) | 0.95 (3.4 / 1.9) |
| z=1.0 (1435) | 7.67 (13.6 / 9.7) | 8.21 (26.3 / 8.9) | **1.48** (5.5 / 2.3) | 1.37 (4.7 / 2.4) |
| z=1.5 (1144) | 5.60 (14.9 / 9.6) | 11.34 (39.2 / 9.7) | **1.17** (3.2 / 1.9) | 1.27 (4.2 / 1.7) |
| z=2.0 (839) | 10.48 (22.8 / 12.5) | 16.47 (49.9 / 13.7) | **1.36** (4.2 / 1.8) | 1.32 (3.6 / 1.7) |

**Every epoch can be fitted to the z=0.4 precision by the same form** — every
halo-mass tercile within ±4–5 per cent at every radius at each of the five
epochs, where the five-epoch incumbent is 5–10 and the z=0.4 fit extrapolated is
5–16. The plane slope $M(30\text{–}50)|M(<30)$ is reproduced at $z\le1.0$ (1.39 vs
1.37; 1.37 vs 1.31) and is too steep at $z\ge1.5$ even when fitted there (1.35 vs
1.21; 1.40 vs 1.00 at $z=2$): the mean's outskirt-against-centre relation at
$z=2$ is steeper than the data's — a residual the class shows only at $z\ge1.5$
and only in the plane, not in the average CoGs. On the whole 2397-galaxy sample
at $z=2$ the per-epoch fit is +22–28 per cent in the lowest tercile — those are
galaxies outside the mask (incomplete halo histories at $z=2$) that the fit never
saw; D5 says both samples are reported, and this is the sample difference.

**How the parameters move with the epoch fitted** (Option 1, the kpc form;
`figures/exp63_single_epoch_params_vs_epoch.png`):

| fitted at | $a_0$ | $a_M$ | $m_{1/2}$ | $d$ | $s_c$ [kpc] | $\log(f_e)$ | $n_c$ | $c_e$ |
|---|---|---|---|---|---|---|---|---|
| z=0.4 | −2.564 | −0.563 | 10.65 | 1.24 | 2.64 | −0.717 | 0.88 | 1.09 |
| z=0.7 | −2.571 | −0.559 | 11.21 | 1.25 | 2.58 | −0.640 | 1.09 | 1.07 |
| z=1.0 | −2.585 | −0.575 | 10.91 | 1.78 | 2.37 | −0.573 | 1.14 | 1.04 |
| z=1.5 | −2.611 | −0.614 | 10.95 | 3.00 (bound) | 2.02 | −0.463 | 1.28 | 1.00 |
| z=2.0 | −2.621 | −0.658 | 12.09 | 3.00 (bound) | 1.75 | −0.393 | 1.25 | 1.05 |

Smooth and monotonic, which is what makes them readable as a time law the frozen
exponents do not carry:

1. **The compact size shrinks toward high redshift**, 2.64 → 1.75 kpc. With
   $b_c$ frozen at 0 every fit uses one size for its whole history, so the
   sequence says the stars in place by $z=2$ sit at ~1.8 kpc and by $z=0.4$ the
   compact component is at ~2.6 kpc. Read as a deposit-size law it is
   $s_c\propto(1+z')^{b_c}$ with $b_c\approx-0.5$ (from the end points, −0.54) —
   later compact deposits larger; a single-epoch fit cannot tell that from the
   early compact stars expanding, but the sequence fixes the exponent either way.
2. **The extended size is a constant fraction of $R_{200c}$ at the deposit.**
   $\log f_e$ rises by 0.32 over $\Delta\log(1+z)=0.33$, i.e. the frozen
   $b_e=-0.90$ is wrong by +1.0: $b_e\approx+0.1$. The incumbent's strongly
   negative $b$ was a single size law compensating for having no compact
   channel (Stage 1's reading); with the compact channel in place the extended
   deposits simply scale with the halo. This is also why the z=0.4 fits'
   outskirts collapsed at $z\ge1.5$ in 5d: with $b_e=-0.9$ the early extended
   deposits were made far too small.
3. **The split flattens at high redshift** ($d$ 1.2 → 3.0, at the bound): at
   $z\ge1.5$ the compact share hardly depends on the halo mass at deposit —
   every halo is below the switch mass early on, so the mass-only logistic has
   nothing to act on and the fit asks for a constant share instead.
4. The efficiency law drifts a little ($a_0$ −2.56 → −2.62, $a_M$ −0.56 → −0.66)
   — the frozen $a_z$, $a_{Mz}$ are close but not exact; $n_c$ 0.9 → 1.25 and
   $c_e\approx1.0$–1.1 are steady.

Option 2 fitted at each epoch converges to the same structure: $g_c$ = −0.39 /
−0.40 / −0.32 / −0.40 at $z$ = 0.7 / 1.0 / 1.5 / 2.0 (a compact size that does
not scale with the halo mass at deposit, at every epoch), $d$ at the bound at
$z=2$, and gate rms 0.95 / 1.37 / 1.27 / 1.32 — the third time the two forms
have agreed. (Its `w_min` is 0.13–0.17 at $z\le1$ and 0 at $z\ge1.5$; its
three starts spread more than the kpc form's, e.g. 2.68 / 2.83 at $z=1.0$.)

**What follows from this.** The frozen exponents were the right move for the
single-epoch question, and the per-epoch sequence now *measures* two of them:
$b_c\approx-0.5$, $b_e\approx+0.1$. The natural next fit is the kpc form with
$b_c$ and $b_e$ free (and $a_z$, $a_{Mz}$ either frozen or free — the drift in
$a_0$, $a_M$ is small) at all five epochs jointly, the binned term summed over
epochs — one $\theta$ tested against five gates. Whether one $\theta$ reaches
rms ≲ 1.5 at every epoch, or the per-epoch optimum is a ceiling the shared law
cannot reach (exp61's lesson for the incumbent), is the question that decides
whether this is a five-epoch model. Not started: a scope decision for the user.

### 5f — one θ at all five epochs: the joint fit against the per-epoch ceiling (2026-08-29)

`stage2_fit.py --joint`: one θ, the loss the sum over the five epochs of that
epoch's $A^2+F^2+S^2+B^2$, each on its own sane + mh-complete mask with its own
$\sigma_A$, halo-mass terciles and null references (every term 1 at the null;
null loss 19.27), one `predict2` call over all galaxies and epochs per
evaluation (0.32 s). The kpc form. Two versions: (i) $b_c$, $b_e$ free, $a_z$,
$a_{Mz}$ frozen (three starts: nested, from the $z=0.4$ solution, from it with
the 5e exponents re-anchored — 16.18 / 16.15 / 16.22; two named starts that
were re-anchored crudely began at loss 48–182 and the first line search ran
them into the failing region: a start must begin near a valid solution);
(ii) all four time exponents free (two starts: 15.630 / 15.633 — the same loss
from different parameters, $a_z$ 0.30 vs 0.41, $b_e$ −1.35 vs −0.49: a flat
valley, the exponents partly degenerate; $n_c$ at its lower bound 0.5 in every
joint start). The sum of the five per-epoch optima (5e) is 13.37.

Gate rms on each epoch's fit mask (`figures/exp63_single_epoch_joint_summary.png`;
tables `outputs/single_epoch_gate_joint_e*.log`):

| judged at | incumbent (5-epoch fit) | Option 1 from z=0.4 | joint, $b_c$ $b_e$ free | joint, four exponents free | fitted at that epoch alone |
|---|---|---|---|---|---|
| z=0.4 | 4.98 | 1.50 | 3.44 | 3.33 | 1.50 |
| z=0.7 | 6.54 | 5.07 | 2.66 | 2.04 | 1.05 |
| z=1.0 | 7.67 | 8.21 | 3.90 | 3.77 | 1.48 |
| z=1.5 | 5.60 | 11.34 | 4.03 | 3.72 | 1.17 |
| z=2.0 | 10.48 | 16.47 | 10.42 | 7.59 | 1.36 |

The standard battery on the four-exponent joint fit (`--eval-only --tag _joint_kpc_free`;
nothing is a prediction here — every epoch was fitted), whole sample / fit mask,
incumbent in brackets:

| quantity | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| $M_*(<10)$ | −1.5 / −1.5 [+3.6] | +1.6 / +2.1 [+5.3] | +4.6 / +5.6 [+5.8] | +3.2 / +4.1 [−0.8] | +1.4 / −3.1 [−8.9] |
| $M_*(<100)$ | −2.6 / −2.6 [+0.4] | +2.0 / +1.2 [+5.1] | +3.5 / +1.6 [+6.4] | +3.8 / +0.9 [+3.0] | +1.3 / −7.0 [−4.7] |
| $M_*(50\text{–}100)$ | +9.2 / +9.2 [−3.5] | +14.9 / +3.8 [+4.6 / −5.4] | +15.9 / −1.7 [+8.7 / −7.4] | +23.0 / −6.6 [+19.7 / −8.4] | +33.0 / −10.7 [+32.4 / −9.5] |
| tier 3 (beyond 5 kpc) | 0.276 [0.280] | 0.278 [0.282] | 0.286 [0.291] | 0.292 [0.291] | 0.305 [0.314] |

C8 (median log model/truth of $M_*(<4.92)$): whole population span **0.013**
[incumbent 0.058]; decliners 0.161 [0.205]; non-decliners 0.091 [0.037] — the
joint fit centres the whole population at every epoch by letting the
non-decliners drift from −0.038 to +0.053 dex. Plane slope $M(30\text{–}50)|M(<30)$
1.28 / 1.31 / 1.32 / 1.37 / 1.49 against the truth's 1.42 / 1.62 / 1.67 / 1.63 / 1.43:
shallow at $z=0.7$–1.5, as the per-epoch fits were.

**What it means.**

1. **One shared law is better than the five-epoch incumbent at every epoch**
   (gate rms 2.0–3.8 against 5.0–7.7 at $z\le1.5$; 7.6 against 10.5 at $z=2$;
   the battery better on $M_*(<10)$, tier 3 and the fit-mask outskirts at every
   epoch; the whole-population central error flat to 0.013 dex). By the
   programme's own bar it is the best five-epoch mean the deposition-only
   class has produced.
2. **It does not reach the per-epoch ceiling.** 3.3–3.8 against 1.0–1.5 at
   $z\le1.5$ and 7.6 against 1.4 at $z=2$ — the same lesson exp61 recorded for
   the incumbent (memory `per-epoch-ceiling-does-not-transfer`), now with the
   gap measured for this form: the four time exponents carry about half of the
   distance (16.15 → 15.63 → the per-epoch sum 13.37 is in loss units 40 per
   cent of the way). What the per-epoch sequence asked for and the shared law
   cannot write: a compact index that rises with redshift ($n_c$ 0.9 → 1.25;
   the joint fits rail $n_c$ at 0.5 instead), a split that flattens at $z\ge1.5$
   ($d$ at its bound in the per-epoch fits at $z\ge1.5$, at the *other* bound in
   the joint fit), and a compact size that shrinks toward high $z$ faster than
   any $(1+z)^{b_c}$ that also fits $z=0.4$.
3. **$z=2$ is where the shared law fails most, and it fails in amplitude**: on
   the fit mask the joint fit is −7 to −11 per cent at every radius at $z=2$
   (the incumbent −12). The per-epoch $z=2$ fit removed this with its own
   $a_0$, $a_M$; freeing $a_z$, $a_{Mz}$ recovered part of it (10.4 → 7.6) at the
   cost of a flat valley in the exponents. The efficiency law's time dependence
   is the next thing the class needs, and it is not a size law.

The joint fits are the closing product of the single-epoch round:
`outputs/stage2_fit_joint_kpc{,_free}.npz` (merged from the parallel starts),
their batteries in **`figures/qa_joint/`** (the two joint models' standard battery, kept apart from `figures/qa/`; the other Stage 5 batteries are in `figures/qa_stage5/`; `stage2_fit.py --eval-only --qadir NAME` writes a product's battery to its own folder)
and `figures/exp63_stage2_{predictions,c8,residuals,gates,channels}_joint_kpc{,_free}.png`.

### 5g — galaxies whose measured history is not physical (2026-08-30)

The user's reading of the joint fit's worst cases (`figures/qa_joint/qa_cases_*`):
objects with a dramatic, very likely unphysical CoG evolution, which the
existing filter does not catch. `selection.sane_history_mask` drops an epoch
only when $M_*(<100)$ is more than 3 dex below the galaxy's own $z=0.4$ value
(row 181, the broken cross-match); row 2372 grows 3 dex in the centre from
$z=2$ to 0.4 and passes it. `history_outliers.py` compares each epoch's
stellar mass with the halo instead — three galaxy-epoch flags: **A** more than
0.5 dex off the population's $M_*(<100)$–$M_h$ running median at that epoch
(the population's 16–84 half-width is 0.12–0.15 dex, so a 4σ outlier; a strict
0.75 dex set is kept too), **B** an adjacent-epoch jump of more than 1 dex in
$M_*(<100)$ (the sample's 99th percentile is 0.68), **C** an adjacent-epoch drop
of more than 0.3 dex in $M_*(<30)$ (1st percentile −0.06). Log
`outputs/history_outliers.log`; flags `outputs/history_outliers.npz`; the
candidates' CoGs, halo growth curves and $M_*$–$M_h$ tracks in
`figures/exp63_history_outliers.png`.

| rule | galaxy-epochs | galaxies | flagged galaxy-epochs *inside* the fit mask at that epoch (z=0.4 … 2) |
|---|---|---|---|
| A at 0.5 dex | 59 (47 below the relation) | 41 | 2 / 1 / 5 / 3 / 4 |
| A at 0.75 dex (strict) | 14 (all below) | 7 | 0 / 0 / 0 / 0 / 0 |
| B (jump > 1 dex) | 2 | 2 | 0 |
| C (drop > 0.3 dex) | 0 | 0 | 0 |

What the candidates look like (figure): haloes of $10^{12.7-13.4}$ whose DiffMAH
curve grows smoothly by ~1 dex while the measured stellar mass grows 1.5–2 dex
from 0.8–1.5 dex *below* the relation at $z\ge1$ — a minor or wrong progenitor
in the cross-match at the early epochs, not evolution; their CoG *shapes* at
those epochs are unremarkable, only the mass is wrong. Adjacent-epoch drops do
not occur at all: the "dramatic" cases are all under-massive early, never
over-massive.

**Impact.** The fits are already clean: the sane + mh-complete masks exclude
every strict candidate at every flagged epoch, and only 15 flagged
galaxy-epochs (of ~7,600) are inside any mask under the 0.5 dex rule; dropping
them changes the joint fit's loss from 15.630 to 15.537 with no per-epoch term
moving by more than 0.015. Where they do count is the **whole-sample QA**
(D5's "all" column and the worst-case panels), which still contains row 181
with a 771,137 per cent residual. `single_epoch.py --gate/--qa --physical` and
`stage2_fit.py --eval-only --physical` now drop the 41 candidates from the QA
sample; the fit masks are unchanged. Recommendation: make the 0.5 dex
$M_*$–$M_h$ rule part of `selection` alongside the 3 dex rule (a user
decision — it touches every experiment's sample).

### 5h — what exp62's profile families offer this model (2026-08-30, reading only)

exp62 (merged to `master`, worktree `hongshao_exp62_cog_fit_atlas`) fitted
every galaxy at every epoch with one- and two-component CoG families: the
unrestricted double Sérsic (6 parameters, median full-CoG rms 0.00263 dex) is
the accuracy ceiling with degenerate parameters; the **cubic-logit profile**
(5 parameters, `hongshao/cubic_logit.py`, `doc/cubic_logit_profile.md`) —
the enclosed fraction $F=\sigma(ax+bx^2+cx^3)$, $x=\ln R/R_{50}$, monotone by
construction — reaches 0.00249 dex with twelve times better conditioning; its
one caveat is a formal central density hole ($\Sigma\to0$ as $R\to0$; exp64
is searching for a family without it). exp62's halo-information stages found
that the concurrent halo mass reproduces the *average* normalised CoG, DiffMAH
explains 2–7 per cent of the diversity about it, concentration nothing, and
more than 90 per cent of the individual diversity is unexplained — the same
picture as exp63's Stage 1 and memory `halo-history-knows-a-little-about-the-core`.

Three ways the families could enter here, assessed:

1. **As the deposit kernel.** The kernel is the profile of *one deposit*, not of
   the galaxy; the galaxy is the kernel convolved with the deposit-size
   distribution (tech note 04). A more flexible kernel is not what the model
   is short of — Stage 1 showed the incumbent's own kernel represents every
   $z=0.4$ curve to 0.0036 dex once the size distribution is free, and 5f
   showed the shared law's failure is in the *time dependence* of sizes,
   shares and the efficiency amplitude, not in the kernel's shape. A cubic-logit
   kernel would add two shape parameters and the central hole (a deposit with
   $\Sigma(0)=0$ is unphysical for in-situ stars, and every deposit's hole
   would sit inside 2 kpc at every epoch). Not recommended.
2. **As the extended kernel only.** The extended deposit is an ex-situ envelope,
   where a cored, sharply bounded shape is plausible; 5b showed the gate does
   not care about the extended family (gompertz vs Sérsic identical under the
   binned objective). No leverage to gain.
3. **As the target representation of the data**, replacing the 24-radius CoG in
   the objective by each galaxy's five cubic-logit coordinates (or the model's
   decoded coordinates against the data's). This is where exp62's result is
   usable: the coordinates are well conditioned and $R_{50}$, $R_{20}$, $R_{80}$
   are exact, so a *size-and-shape* objective — the model's and the data's
   $R_{20}/R_{50}/R_{80}$ and warp coordinates compared directly — would price
   the 2–5 kpc curvature and the outskirts in a low-dimensional, interpretable
   way, and could replace the per-galaxy shells term that 5b showed is blind
   to the binned offset. Worth a single-epoch test under the standing method
   (evaluate the leverage on the gate before fitting); not started.

So: the families do not change the deposition model's design; exp62's
conditioning result makes the cubic-logit coordinates a candidate *objective*,
and its halo-information result independently confirms that the mean is
already at the limit of what the halo history determines.

### 5i — the fitting sample, corrected: the same fits on all sane galaxies (2026-08-30)

**The gap, and the rule (the user).** Every fit in this experiment — and in
exp57 and exp54's later stages — used exp54 Stage 3.7's per-epoch masks as the
*fit* sample: the progenitors above the halo-mass completeness cut (2397 / 1780
/ 1435 / 1144 / 839 galaxies). exp54 had designed that subset as an *after-fit*
re-score with θ frozen (`selection.py`, section 3), and `stage37_size_epoch.build`
handed it on as the mask. The user's rule, now repo-wide (`CLAUDE.md`,
`doc/lessons.md`, memory): **the fit uses every galaxy selected at z=0.4 whose
halo history and stellar history are sane, at every epoch; halo-mass
completeness is an after-fit check, never a fitting criterion.** Implemented
once, `selection.fitting_sample_mask`: finite CoGs and DiffMAH mass at all
five epochs, the 3 dex backward rule, and no stellar-history outlier at any
epoch (5g's rules, now in `selection.stellar_history_flags`: 0.5 dex off the
running-median $M_*$–$M_h$ relation, > 1 dex adjacent jump, > 0.3 dex adjacent
drop); one flagged epoch removes the galaxy everywhere. **2356 of 2397
galaxies, at every epoch.** `stage2_fit.py --sample sane` is the default
(`legacy` reproduces the old masks); every fit log prints the sample and what
each criterion removed; the mh-complete subset (2356 / 1758 / 1416 / 1129 / 831)
is printed after every fit as the check.

**The refits** (kpc form, binned objective, frozen exponents for the per-epoch
fits; the four-exponent joint fit from two starts, 15.70 / 15.35 against the
sane-sample null 19.25). Gate rms, judged on the fitting sample | on the
mh-complete check (`figures/exp63_single_epoch_sample_summary.png`; logs
`outputs/single_epoch_gate_sane_e*.log`, `..._jointsane_e*.log`):

| judged at | incumbent | joint, old masks | **joint, fitting sample** | each epoch alone, old masks | **each epoch alone, fitting sample** |
|---|---|---|---|---|---|
| z=0.4 | 5.04 \| — | 3.32 \| — | 3.33 \| — | 1.46 \| — | **1.42** \| — |
| z=0.7 | 6.25 \| 6.54 | 2.45 \| 2.11 | 2.36 \| 2.76 | 1.56 \| 1.08 | **1.17** \| 1.23 |
| z=1.0 | 6.86 \| 7.67 | 4.25 \| 3.78 | 3.57 \| 4.32 | 2.93 \| 1.44 | **1.37** \| 2.35 |
| z=1.5 | 4.99 \| 5.60 | 4.78 \| 3.75 | 3.28 \| 4.62 | 3.54 \| 1.19 | **1.23** \| 2.90 |
| z=2.0 | 8.45 \| 10.48 | 8.24 \| 7.48 | 6.88 \| 9.45 | 11.91 \| 1.61 | **2.86** \| 4.99 |

The joint refit's battery on the fitting sample (mh-complete check in
brackets), $M_*(<10)$: −1.7 / +1.2 / +4.2 / +1.7 / −1.6 per cent [−1.7 / +1.7 /
+5.1 / +4.1 / −4.7]; $M_*(<100)$: −2.5 / +1.5 / +2.1 / +0.2 / −4.6 [… −9.7 at
$z=2$]; $M_*(50\text{–}100)$: +1.1 / +2.7 / +1.2 / −1.5 / −1.6 [+1.1 / −4.6 /
−12.3 / −20.8 / −28.4]; tier 3 0.274 / 0.272 / 0.279 / 0.283 / 0.297 (incumbent
0.279 / 0.281 / 0.290 / 0.289 / 0.310); C8 whole-population span 0.013 dex.
Battery in `figures/qa_joint/` (`*_joint_kpc_free_sane`).

**What changed, and what it means.**

1. **On the sample the rule prescribes, the picture of 5e–5f holds and the
   numbers improve.** Fitted alone, every epoch reaches gate rms 1.2–1.4 at
   $z\le1.5$ and 2.9 at $z=2$; the joint θ beats the incumbent at every epoch
   (2.4–3.6 against 5.0–6.9 at $z\le1.5$; 6.9 against 8.5 at $z=2$) and its
   whole-sample outskirts are now within 3 per cent at every epoch (5f: +9 to
   +33). The old-mask fits, judged on the sample they never saw, were far
   worse than they looked (the per-epoch $z=2$ fit: 11.9).
2. **The massive subset is no longer a free lunch.** Every refit is worse on
   the mh-complete check than the old fit that had been tuned to it (per-epoch
   $z=2$: 1.6 → 5.0; joint $z=2$: 7.5 → 9.5; $M_*(50\text{–}100)$ on the subset
   −28 per cent at $z=2$). One θ cannot fit the massive third and the whole
   sample at $z=2$ to the same precision: the low-mass $z=2$ progenitors (two
   thirds of the sample, the progenitor-selected, fast-growing minority of
   their mass) want a different model than the massive third — a halo-mass
   dependence at high redshift that the old masks had hidden. This is the
   after-fit check doing its job, and it is the first concrete thing the
   corrected sample teaches.
3. **The per-epoch parameter sequence changes at $z=2$** (fitting sample):
   $s_c$ 2.9 → 2.2–2.5 kpc (a milder shrink, $b_c\approx-0.3$), $\log f_e$
   −0.68 → −0.16 (a steeper rise, $b_e\approx+0.7$), and the split now moves
   *up* in mass with redshift ($m_{1/2}$ 11.5 → 13.1, $d$ 1.0 → 0.5) instead of
   flattening; $n_c$ 1.25 → 2.1 and $c_e$ 1.2 → 1.6 at $z=2$. Read with point 2:
   the $z=2$ fit is describing a different population, and the "time law" of
   5e was partly a mass-selection law.
4. The joint refit's parameters: $a_z$ 0.35, $a_{Mz}$ 0.12, $b_c$ −0.80,
   $b_e$ −1.20, $n_c$ 0.53 (at the bound again), split width 0.24 dex at
   $10^{11.8}$ — the same structure as 5f's, so the n_c symptom is not a
   sample artefact.

Fit files: `stage2_fit_frozen_binned_kpc_sane_e{0..4}.npz`,
`stage2_fit_joint_kpc_free_sane{,_s0,_s1}.npz`. Everything before 5i in this
README was fitted on the old masks and is labelled as such where quoted.

## What exp63 delivered, and what it did not (2026-08-28)

Against the science plan's three requests:

1. **The analytic integral (request 1): done and proved.** The incumbent's 72-step
   sum is a first-order approximation of an exact integral along the DiffMAH curve;
   the engine evaluates 2397 galaxies × 5 epochs in 0.32 s; the model is a
   convolution in log radius of the deposit-size distribution with the kernel, with
   closed forms for power-law histories (tech note 04; Stage 0).
2. **A better single-epoch mean (request 2): done, with its limit measured.** Read
   from the data by deconvolution (Stage 1: two size modes, cuspy deposits rejected,
   $c_{200c}$ no leverage), the two-channel mean fitted at $z=0.4$ alone has the
   data's inner slope and a 4 per cent better shape in the two lower halo-mass
   terciles (the top tercile keeps a −12 per cent deficit at 2–4 kpc). Its
   predictions must be read precisely (`figures/exp63_stage2_c8.png`,
   `figures/qa/qa_bins_exp63_stage4_stage2.png`): for the non-declining 58 per
   cent the *central* error is **constant** across the five epochs to 0.016 dex —
   at an offset of −0.03 to −0.05 dex, not at zero — while for the whole
   population the amplitude at z ≥ 1.5 is *worse* than the incumbent's (M(<10 kpc)
   −8 and −21 per cent at z=1.5 and 2 against −1 and −9; the incumbent was fitted
   at all five epochs, this model at one). It over-predicts the
   $z\ge0.7$ outskirts (+16–28 per cent) and gets the assembly correlations
   backwards; the three variants that fix the correlations (delay, two growth-rate
   splits) all break the evolution — a measured tension of the class (P9), not a
   tuning problem.
3. **A model that is statistical from the start (request 3): done, and it
   transfers.** Five noise parameters inside the history reproduce the $z=0.4$
   population planes to 1.2–1.3× the sampling floor held out (the mean alone:
   2.6–3.0; exp60's output layer: 1.7–2.2), the tier-2e widths to within 6 per cent
   at $z=0.4$ and 9 per cent at every predicted epoch, and halve the excess
   cross-epoch size coherence of a per-epoch layer — with no coherence parameter.

What it does not do: declining centres (0 per cent against 42 — the deposition-only
class, P8), the top halo-mass tercile's inner 3 kpc (−11 per cent — **repaired at
$z=0.4$ by Stage 5**: the compact size in physical kpc, or a mass exponent that
cancels $R_{200c}$'s, puts every halo-mass tercile within ±3 per cent at every
radius; the extrapolation of that mean has not been looked at, by the user's
rule), the $z\ge1.5$ amplitude (−15 to −21 per cent at $z=2$ on the full sample),
and the $z=0.4$ assembly correlations (P9). The plan's §5 failure criteria: G1 did find two scales
(not the first failure); the two-channel mean's $z=2$ B3 on the non-decliners is
−0.048 dex against the baseline's +0.010 — worse by 0.058, past the 0.05 line the
plan set, so by the plan's own rule the "one epoch fitted, four predicted" claim
holds at $z\le1.0$ (span 0.015) and is degraded at $z=2$ by the extended channel's
early arrival; every noise source had leverage (not the third failure); the joint
mean–noise refit was not needed (the mean was frozen throughout; not the fourth).
