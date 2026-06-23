# exp21 — An Re-based emulator, finer outskirt bins, and a richer-MAH test

## Question
The graduated emulator predicts stellar masses in *fixed physical* apertures.
A fixed kpc aperture mixes physically different regions across the size–mass
range, so we bin the profile in **effective-radius (Re) units** (Re = half-mass
radius within 120 kpc, from the 1-D curve of growth). Two outskirt-focused
questions:
1. **Finer Re outskirt bins** — six bins `<0.5 / 0.5–1 / 1–2 / 2–4 / 4–6 / 6–9 Re`
   (replacing the earlier coarse `4Re–120kpc`).
2. **Does a richer MAH help the Re outskirts?** The outskirts may track *recent*
   accretion, which the smooth DiffMAH rolling-power-law could blur. Compare
   `DiffMAH(4)+c_200c` vs `MAH-PCA(8)+c_200c` vs `raw-MAH(18)+c_200c`, per bin.

## Method
- Re from the CoG; six bins in Re. Same probabilistic emulator as exp19 (linear
  mean + heteroscedastic full covariance), 5-fold CV, via a **local n-bin copy**
  of the exp19 machinery — `hongshao/emulator.py` is untouched.
- **Selection cost of `6–9 Re`:** 9 Re reaches ~89 kpc for the median galaxy but
  exceeds the measured CoG (148 kpc) for larger galaxies. Requiring the 9 Re edge
  to be real data **drops 19% of the sample** (n=2050), preferentially the
  largest-Re (extended, massive) galaxies — a mass-correlated selection. The
  `6–9 Re` bin also runs past the 120 kpc observational comfort zone for ~⅓ of
  galaxies, so it is harder to measure in *real* data than the inner bins.
- Richer-MAH test: per-bin linear CV (homoscedastic, matching exp13). MAH-PCA is
  the M0-normalized-shape PCA + M0; raw-MAH is the 18-point `log M_halo(t)`.

## Key result
**(1) The six-bin Re emulator works and is well-calibrated. (2) A richer MAH does
NOT help the outskirts — DiffMAH(4)+c_200c is at the ceiling in the Re frame just
as in kpc (exp13).**

Re emulator (DiffMAH+c_200c, 6 bins, n=2050): **CRPS 0.0701**, joint NLL −7.110,
coverage 0.53/0.71/0.91/0.95 (nominal 0.50/0.68/0.90/0.95), conditional gap 0.023.

Richer-MAH test — mean CV CRPS [dex] and gain over DiffMAH:

| feature set | mean CRPS | gain | `6–9 Re` bin gain |
|---|---|---|---|
| DiffMAH(4)+c_200c | 0.0706 | — | — |
| MAH-PCA(8)+c_200c | 0.0699 | **+1.0%** | +0.0009 dex |
| raw-MAH(18)+c_200c | 0.0703 | **+0.6%** | +0.0003 dex |

- **The richer-MAH gain is tiny (~1%) and, crucially, does NOT rise in the
  outskirts** — it is flat-to-slightly-core-weighted (largest in `<0.5Re`,
  ~1.6%, smallest in the mid bins). The hypothesis that DiffMAH over-smooths
  *recent* assembly and the outer bins would benefit from a fuller MAH is **not
  supported**: the `6–9 Re` outskirt gains essentially nothing (+0.0003 dex from
  raw-MAH). The four DiffMAH parameters already carry all the MAH's
  profile-predictive power, at every radius.
- This **confirms and extends exp13** (which found +0.9% for the kpc 50–100 bin)
  to the Re frame and to finer outer bins. The residual outskirt scatter is
  **intrinsic** (projection / ICL / low-SB / stochastic accretion), not
  feature-limited — no MAH representation we tried recovers it.
- MAH-PCA(8) edges out raw-MAH(18) despite carrying less information: 9 vs 19
  features, so PCA's compression regularizes the CV fit (raw-MAH overfits
  slightly). Both sit at the ~1% ceiling.

## Value over a plain current-mass SHMR
**Yes — the emulator beats a `M_halo(z=0.4)`-only relation at every radius,
including the hard outskirts, and roughly halves the scatter a plain SHMR
leaves there.** Per-bin CV CRPS [dex] and R² (variance explained):

| outskirt bin | M0 only | DiffMAH(4) | DiffMAH+c_200c | CRPS gain | R² |
|---|---|---|---|---|---|
| kpc 50–100 | 0.128 | 0.098 | **0.093** | **+27%** | 0.52 → 0.74 |
| Re 6–9 | 0.106 | 0.079 | **0.074** | **+30%** | 0.56 → 0.78 |
| Re 4–6 | 0.106 | 0.078 | **0.073** | **+31%** | 0.51 → 0.75 |

- Across **all** bins the gain is **+22% to +36%** CRPS over M0-only (largest in
  the core, but the outskirts gain ~+27–31%). In variance terms, DiffMAH+c_200c
  **removes ~½ of the residual scatter** a current-mass SHMR leaves in the
  outskirts (50–100 kpc: 48% unexplained → 26%).
- **Assembly history does the bulk of it**: M0→DiffMAH is +23–25%, and `c_200c`
  adds a further ~4–5 points (+27–31% total). So "the outskirts are hard" is
  relative to a *good* model — they are far better predicted than the textbook
  current-mass SHMR allows.

## Decision
- **Keep DiffMAH(4) + c_200c as the feature set** — it is portable *and* at the
  MAH information ceiling for the profile, outskirts included. Do not adopt
  MAH-PCA / raw-MAH (non-portable, ≤1% gain, and they do not help where it was
  hoped).
- **Re-based binning is a viable alternative target** (well-calibrated, same
  assembly predictability), but the `6–9 Re` bin costs a 19% mass-correlated
  selection and probes beyond the 120 kpc observational limit — keep it as an
  exploration, not the default. The graduated kpc emulator stays primary.
- The outskirt residual being intrinsic redirects future gains away from "more
  halo features" and toward the data (projection/ICL modeling) or a genuinely
  independent halo axis (environment / initial conditions), not the MAH.
- Independent experiment; `hongshao/emulator.py` and `forward.py` unchanged.
