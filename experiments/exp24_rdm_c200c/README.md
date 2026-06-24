# exp24 — Does c_200c predict the radial-DiffMAH (RDM) shape parameters?

## Question
exp03/04/05 found the halo→RDM connection is **weak for the *shape* parameters**
(inner/outer slope `beta_in`/`beta_out`, transition radius `R_c`): the MAH
predicts the normalization `logMstar0` well but the shape parameters barely at
all. Those experiments predate `c_200c`. Hypothesis (this experiment): since
concentration is the natural predictor of profile concentration, **`c_200c`
should rescue the shape parameters** the MAH could not reach — making the old
"weak connection" verdict stale.

## Method
n=2539, the cached `rdm_*` params (median fit rms 0.001 dex; `R_c`→`log10 R_c`).
- **(A) Per-parameter predictability:** CV R² of each RDM parameter from `M0` /
  `DiffMAH(4)` / `DiffMAH(4)+c_200c`, and the partial correlation of `c_200c`
  with each parameter at fixed MAH (residual-on-residual; the exp16 test).
- **(B) Generative reconstruction:** predict the joint 5-vector with the
  (N-target) heteroscedastic emulator, sample θ~N(μ,Σ), reconstruct the CoG via
  `profiles.cog_from_physical`, and compare per-radius CRPS / recon RMS to the
  exp22 PCA route. (Judged on R≥5 kpc, the RDM fit range.)

## Key result — the hypothesis is REFUTED
**`c_200c` does NOT rescue the RDM shape parameters. It helps only the
normalization (`logMstar0`) — the same total-mass effect concentration has
everywhere. The exp03/04/05 "weak halo→shape-parameter connection" verdict
stands, now extended to `DiffMAH + c_200c`.**

CV R² (variance explained) and the `c_200c` increment:

| RDM param | M0 | DiffMAH | +c_200c | c_200c gain | partial r(c200c · MAH) |
|---|---|---|---|---|---|
| **logMstar0** (norm) | 0.50 | 0.61 | **0.65** | **+0.041** | **+0.33** |
| beta_in (inner slope) | 0.00 | 0.05 | 0.06 | +0.006 | −0.08 |
| beta_out (outer slope) | 0.17 | 0.19 | 0.19 | −0.001 | +0.01 |
| logR_c (transition R) | 0.10 | 0.13 | 0.14 | +0.005 | +0.08 |
| Delta (transition width) | 0.00 | 0.01 | 0.01 | +0.001 | −0.04 |

- **All of `c_200c`'s contribution is to the normalization** (R² 0.61→0.65,
  partial r +0.33) — i.e. *how much* stellar mass, not *how it is distributed*.
  This is exactly the total-mass effect `c_200c` has in the aperture emulator
  (exp16–19: it improves the mean, which is total-mass-dominated).
- **The shape parameters stay weakly predictable and `c_200c` adds ≈0** (gains
  ≤0.006 R², partial r ≤0.08, no sign stability). `beta_in` and `R_c` are *not*
  lit up by concentration. My prior ("`c_200c` ≈ profile concentration, so it
  should predict the inner slope / transition radius") is not supported.
- **Why the prior was wrong:** I over-read exp22, which compared CoG-PC1 vs
  density-PC1 predictability (*both* already using `DiffMAH+c_200c`) — it never
  isolated `c_200c`'s *increment* to the shape. This experiment isolates it
  cleanly, and the increment is small.

## Reconstruction ties the PCA route (despite the weak per-parameter R²)
Generative DiffMAH+c_200c → 5 params → CoG (R≥5 kpc): **mean per-radius CRPS
0.0605, recon RMS 0.108 dex** — comparable to the exp22 PCA route (~0.064 / 0.118;
the small edge is mostly the R≥5 kpc cut skipping the noisy inner 2–5 kpc, not a
real win). Two reasons the reconstruction is good even though the shape params
are weakly predicted:
1. the profile is **dominated by the normalization**, which *is* well predicted
   (R²=0.65);
2. the RDM shape parameters are **degenerate** (exp03/05): the modest shape
   signal that exists lives in a *combination*, so the per-parameter R²
   understates it — but it washes into the reconstruction.

## Decision / interpretation
- **The RDM route is a valid, interpretable, equally-accurate, guaranteed-
  monotonic profile predictor** — it ties the PCA route on reconstruction. As an
  emulator output it is a fine alternative.
- **But it does NOT deliver the hoped-for new physical statement.** The honest
  conclusion is the one exp03/04/05 reached, now stronger: halo properties
  (mass + assembly + concentration) predict the galaxy's **total mass /
  normalization** well and its **profile shape only weakly**, and `c_200c` adds
  to the *former*, not the latter. Concentration sets *how much* mass, not (in
  the RDM basis) *how it is distributed*. The user's memory was correct.
- Net: a clean negative result. No reason to graduate the RDM route over the PCA
  route on physical-insight grounds; keep it only if guaranteed monotonicity or
  the specific 5 physical numbers are wanted downstream. The profile-shape
  residual remains intrinsic (projection / ICL / triaxiality), consistent with
  exp13/exp21/exp22.
- Library untouched (reused `emulator.fit` + `profiles.cog_from_physical`).
