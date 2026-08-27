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
