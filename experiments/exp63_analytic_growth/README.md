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
