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
