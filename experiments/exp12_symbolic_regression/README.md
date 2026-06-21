# exp12 — Symbolic regression for a parsimonious nonlinear SHMR mean (PySR)

## Question
exp09 showed a flexible gradient-boosted-trees ceiling does **not** beat the
linear mean on the aperture/annulus masses, so the predictable
`M*(annulus | MAH)` relation is essentially linear and the closed-form linear
emulator (exp08/exp11) is at the ceiling. The remaining question is
*interpretive*, not about raw accuracy: **is there a single, parsimonious
nonlinear term** (e.g. a cross-term `logmp·early`) that belongs in the
closed-form equation, and does it buy any measurable skill *on top of* the full
linear model? We use symbolic regression (PySR) to search the four portable
DiffMAH params for such a term.

## Method
PySR restricted to `{+, -, *, square}`, so every equation is a **polynomial** —
i.e. PySR performs sparse, parsimonious selection over polynomial cross-terms
(a parsimonious cousin of exp09's dense poly-2). Two complementary searches:

1. **Discovery on the linear residuals** (the sharp test). Fit the full 4-term
   linear model per aperture, then run PySR on its residuals vs `(logmp, logtc,
   early, late)`. OLS residuals are orthogonal to the linear features, so PySR
   can only surface *nonlinear* structure the linear model misses — or nothing.
2. **Discovery on the full target** (the literal closed form). PySR on `M*`
   itself, per aperture, for the human-readable equation + Pareto front.

**Evaluation** — all 5-fold CV with the exact exp11 conditional-Gaussian
emulator (linear mean + full residual covariance), scored with the exp07 suite
(CRPS, interval calibration). Three nested models, each keeping the four linear
terms so the comparison isolates the value of the *added* term, not PySR's
pruning:
- `linear` (4 DiffMAH terms) — baseline;
- `linear + PySR correction` (linear + each aperture's discovered nonlinear
  term) — the parsimonious augmented model;
- `poly-2` (all 14 degree-2 terms) — dense reference (exp09).

PySR runs deterministically (`parallelism="serial"`, `random_state=0`) so the
result is reproducible; the whole run (8 fits, n=2539) takes ~100 s once the
Julia backend is installed.

## Inputs
- `data/processed/tng300_072_z0p4.fits` (`use` sample, n=2539): the four cached
  DiffMAH params `dmah_logmp/logtc/early/late` (exp10) and `logmstar_aper`.
- Library: `hongshao.metrics` (CRPS, calibration), the exp11 emulator (inlined).
- Dependency added: `pysr` (Julia `SymbolicRegression.jl` backend, auto-installed).

## Key result
**The relation is essentially linear, and PySR confirms it — the residual
nonlinearity is small (≲5% CRPS) and lives in the late-time accretion index.**

| model | overall CRPS | per-aperture (<10/10-30/30-50/50-100) | vs linear |
|---|---|---|---|
| linear (DiffMAH) | 0.0883 | 0.069 / 0.087 / 0.098 / 0.099 | — |
| linear + PySR correction | 0.0865 | 0.065 / 0.086 / 0.096 / 0.098 | **+2.1%** |
| poly-2 (dense, 14 terms) | 0.0843 | 0.064 / 0.084 / 0.094 / 0.095 | +4.6% |

- **Discovered terms.** From the linear residuals, PySR independently picks the
  outskirts' missed nonlinearity as **`late²`** (the 10–30, 30–50, 50–100 kpc
  annuli) and the core's as **`logtc·late`** (<10 kpc). Both involve `late`, the
  DiffMAH late-time accretion index. All correction coefficients are **positive
  and stable across all 5 folds** (e.g. `late²`: +0.019 ± 0.002 to +0.021 ±
  0.003), so the term is robust, not a fluke.
- **Parsimony.** One term per aperture recovers ~46% of the dense poly-2 gain
  (+2.1% of +4.6%) at 1 extra parameter instead of 10. The PySR Pareto front
  has a sharp knee at complexity ~3–5 nodes then a flat plateau to 18 — the
  fingerprint of a near-linear relation.
- **Visual confirmation** (`figures/exp12_residual_structure.png`, the AGENTS.md
  "don't trust metrics alone" check): the linear residual vs `late` is an
  unmistakable convex U-shape in every aperture, strongest in the outskirts —
  exactly the `late²` signature. A milder inverted-U in `logmp` (a `logmp²`
  curvature) is visible in the outer annuli; poly-2 captures it but the
  single-term pick does not, which is why poly-2 gains a bit more.
- **Calibration unchanged** (0.54/0.71/0.91/0.95) — adding the term does not
  break the honest uncertainties.

### Physical reading
`late` is the late-time power-law index of the halo MAH (recent accretion
vigor). The positive `late²` term means the effect of recent accretion on the
**outer** stellar mass *accelerates*: halos with vigorous late growth build their
30–100 kpc envelopes super-linearly — consistent with recent accretion
depositing ex-situ stars at large radii. In the core (<10 kpc) the relevant
interaction is `logtc·late` (transition-timing × late-accretion). The four
DiffMAH params enter `M*` non-separably because the integrated stellar growth
depends on the *shape* of the MAH, not its parameters individually; PySR
identifies `late`-driven curvature as the single dominant departure from linear.

## Decision
- **Keep the linear closed-form emulator as the working model.** The honest,
  cross-validated gain from the most parsimonious nonlinear term is only +2.1%
  CRPS (and the full degree-2 ceiling only +4.6%), with no calibration change —
  consistent with exp09. A nonlinear form is **not** required for accuracy.
- **If a closed-form nonlinear term is wanted for interpretation, add `late²`**
  to the outer-annulus means (and `logtc·late` to the core): it is the single
  most defensible, robust, physically-readable correction. It is *optional
  polish*, not a structural change.
- The genuine lever for further gains remains **scatter modeling** (the residual
  std grows outward 0.12→0.18 dex), not the mean shape — the mean is settled.
