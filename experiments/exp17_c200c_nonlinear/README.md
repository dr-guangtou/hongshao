# exp17 — The nonlinear limit of `c_200c`

## Question
exp16 added halo concentration to the *linear* mean (+5% CRPS on DiffMAH(4)).
How much *more* can a flexible model extract — i.e. is there a nonlinear
interaction (concentration mattering more for some halos) worth a closed-form
cross-term, and what is the limit of `c_200c`'s improvement?

## Method
Target: the four CoG-derived annulus masses (primary observable), 5-fold CV
CRPS. On the portable set **DiffMAH(4)+c_200c** (and MAH-PCA(4)+c_200c for the
richest-MAH context) we compare: linear · poly-2 (all degree-2, analytic) · GBM
(flexible ceiling, exp09) · PySR (parsimonious polynomial symbolic regression on
the linear residuals, exp12 recipe — to surface any interpretable cross-term).

## Key result
**`c_200c`'s improvement saturates at ~+8% over DiffMAH-only; it enters mostly
linearly, with a small degree-2 bonus from `c_200c²` and a `late·c_200c`
cross-term. Nothing beyond degree-2.**

| model (features) | overall CRPS | vs DiffMAH | vs lin+c200c |
|---|---|---|---|
| linear (DiffMAH) | 0.0882 | — | |
| linear (DiffMAH + c200c) | 0.0839 | +4.9% | — |
| **poly-2 (DiffMAH + c200c)** | **0.0808** | **+8.4%** | +3.7% |
| PySR (DiffMAH + c200c) | 0.0832 | +5.7% | +0.9% |
| GBM (DiffMAH + c200c) | 0.0837 | +5.1% | +0.3% |
| linear (MAH-PCA(4) + c200c) | 0.0827 | +6.2% | |
| GBM (MAH-PCA(4) + c200c) | 0.0822 | +6.8% | |

- **`c_200c` enters essentially linearly.** The flexible GBM ceiling barely beats
  the linear DiffMAH+c200c model (+0.3%); a *more* flexible GBM only overfits
  (0.090–0.100, checked separately). So there is no deep nonlinear structure.
- **A modest degree-2 bonus exists.** poly-2 (analytic, all degree-2 terms) is
  the **best** model (0.0808, +3.7% beyond linear+c200c) — and it beats the GBM,
  because for this smooth, low-dimensional (~5-feature), modest-sample problem an
  explicit low-order polynomial is a better function approximator than trees.
  poly-2 ≈ the practical ceiling; trees find nothing below it.
- **The interpretable nonlinear terms** (PySR on the residuals): **`c_200c²`** for
  the 10–50 kpc annuli (concentration curvature), **`late·c_200c`** and
  `late·c_200c²` for the 50–100 kpc outskirts (concentration's effect on the
  outer mass is *modulated by late-time accretion*), and `logtc·late` in the core
  (the exp12 term). PySR's parsimonious pick captures +0.9% (1–3 terms) vs
  poly-2's +3.7% (20 terms) — the usual parsimony/accuracy trade-off.
- **Best portable closed form:** poly-2 (or the selected `c_200c²` + `late·c_200c`
  cross-terms) on DiffMAH+c_200c, 0.0808 — both portable and a closed form, and
  it edges out even the non-portable MAH-PCA(4)+c_200c (0.0822–0.0827).

## Decision
- The **limit** of adding `c_200c` is ~+8% CRPS over DiffMAH-only (0.0882 →
  0.0808). ~5/8 of that is linear; the rest is a smooth degree-2 effect
  (`c_200c²` + `late·c_200c`). There is no payoff beyond degree-2.
- For the working emulator: **linear `DiffMAH + c_200c`** is the simple, robust
  choice (+5%, clean). If the extra ~+3% is wanted, add the two interpretable
  cross-terms `c_200c²` and `late·c_200c` (the parsimonious slice of poly-2).
- The `late·c_200c` term echoes exp12/exp14: `late` (recent accretion) keeps
  surfacing as the axis that modulates the outskirts — now in interaction with
  concentration.

## Addendum — the poly-2 form, and is the gain justified? (`poly2_check.py`)

**The fitted form** (standardized; coef = dex per 1σ of the term). The dominant
degree-2 terms per aperture:

| aperture | dominant degree-2 terms |
|---|---|
| <10 | `logtc·late` +0.045, `logmp·early` −0.025, `logmp·late` −0.015 |
| 10–30 | `late²` +0.037, `logmp²` −0.029, `late·c200c` +0.021, `c200c²` −0.019 |
| 30–50 | `logmp²` −0.048, `late²` +0.042, `late·c200c` +0.036, `c200c²` −0.021 |
| 50–100 | `logmp²` −0.051, `late·c200c` +0.033, `late²` +0.031, `c200c²` −0.017 |

These are the same physical terms seen before: `logmp²` (SHMR-slope curvature —
the relation flattens at high halo mass), `late²` (exp12), `late·c200c` (exp17
PySR), `c200c²` (concentration curvature), `logtc·late` (exp12 core term).

**Is it justified?** Yes — by two independent criteria — but it is *diffuse*, not
one elegant term:
- **Cross-validation (primary).** 10-seed 5-fold CV: linear 0.0841±0.0001 →
  poly-2 0.0807±0.0002, improvement **+4.1% ± 0.2%, positive in 10/10 splits**.
  CV penalizes degrees of freedom that don't generalize, so an out-of-fold
  improvement *is* the justification — the extra terms are real, not overfit.
- **BIC (which penalizes parameters).** A parsimonious 7-extra-term model has the
  lowest summed BIC (−10029), below both linear (−9321) and full 15-term poly-2
  (−9951). So an information criterion favors adding ~7 degree-2 terms but **not**
  all 15 — the full poly-2 is mildly over-parameterized. (Caveat: per-aperture
  BIC treats the correlated annuli as independent; the CV result is the cleaner
  test.)
- **The downside — the gain is diffuse.** Forward selection shows the +3.7%
  builds up gradually over ~7 terms (~+0.2–0.5% each: `logmp²`, `logtc·late`,
  `early²`, `late·c200c`, `late²`, `logmp·early`, `logmp·late`), not from one
  dominant cross-term. So there is no single clean closed-form correction;
  capturing the full degree-2 gain costs ~7 extra fitted coefficients per
  aperture, which for a *portable, observation-constrainable* model means more
  sample-dependence and less transparency.

**Verdict.** The poly-2 improvement is real and statistically robust (not a DOF
artifact), and BIC even prefers a 7-term subset over the linear model. But
because the gain is diffuse and modest (+3.7% on top of the +5% that `c_200c`
already gives linearly), the **linear `DiffMAH + c_200c`** remains the right
working model for portability/interpretability; the degree-2 terms are optional
polish, best added as the BIC-preferred subset (led by `logmp²` and the
`late`/`c_200c` curvature) rather than a full poly-2.
