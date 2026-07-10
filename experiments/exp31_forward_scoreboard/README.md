# exp31 — standardized tiered QA + the honest forward scoreboard

> **RESULT (2026-07-11, n=45, all leave-galaxy-out, halo-only inputs).** The
> transport emulator (DiffMAH input, universal θ) TIES the statistical-regression
> ceiling on aperture masses, DECISIVELY WINS the outskirt tier at z≥1.5 and the
> observational-plane distribution fidelity (3–4×), and matches it on the profile
> metric. The recommended product configuration stands: **transport-diffmah**.

## Question
HongShao's basic goal is aperture/outskirt stellar masses per epoch; the ambitious
goal is full profile evolution. Scored identically across that ladder, does the
physical transport emulator earn its complexity over plain per-quantity regression?

## Method
- **Part A** — `hongshao/qa.py` (graduated from exp29 `mass_qa` + exp30
  `profile_qa`): one entry point, three tiers — (1) apertures kpc {<10,<30,<50,
  <100} + Re {<1,<2,<4}: per-epoch bias + dex scatter; (2) annuli [10,30],[30,50],
  [50,100],[100,150] kpc + Re analogs + envelopes M*(>R); (2b) observational
  planes (e.g. M*(<30 kpc) vs M*[50–100 kpc]): the truth-vs-model relation
  slope/scatter/rho — the predicted POPULATION must match, since this plane is
  what observations use; (3) profile max|rel| over ALL radii AND R>5 kpc + the two
  visual products (mass-tercile median CoGs, best/worst gallery). Synthetic `demo`
  self-check (identity → exact zeros; +10% scaling → +10% bias, 0 scatter).
- **Part B** — four models, each emitting full (n,5,24) CoGs, LOGO, n=45:
  `transport-real` / `transport-diffmah` (phase-4 universal θ × MAH-derived SHMR
  amplitude); `shmr-only` (per-radius/epoch LOGO regression on logMh(z_k));
  `direct` (exp08 pattern: + c200c, t50, fz2 — the statistical ceiling; per-radius
  regressions, no consistency, no monotonicity).

## Results (median dex scatter; epoch-avg where noted)

| tier | quantity | transport-diffmah | direct (ceiling) | shmr-only |
|---|---|---|---|---|
| 1 | apertures z≤1.0 | 0.10–0.11 | **0.08–0.11** | 0.11–0.12 |
| 1 | apertures z≥1.5 | 0.13–0.18 | 0.14–0.20 | 0.14–0.19 |
| 2 | annuli z≥1.5 | **0.30–0.39** | 0.70–0.87 | 0.88–0.97 |
| 2 | envelopes z≥1.5 | **0.38–0.87** | 1.7–2.0 | 1.6–2.0 |
| 2b | plane fidelity \|Δscatter\| | **0.137** | 0.398 | 0.515 |
| 3 | max\|rel\| all R / R>5 kpc | 39.8% / 31.6% | 39.6% / 32.0% | 40.2% / 32.7% |

(transport-real is uniformly worse than transport-diffmah — 44.9%/36.4% tier 3,
2× worse tier-2 at z=2 — confirming the phase-4 config choice.)

1. **Tier 1 is feature-limited, not model-limited**: four very different models
   land within ~0.02 dex on aperture masses. The basic goal is served by any of
   them; the transport machinery adds nothing per-aperture (echoes exp09's
   linear ≈ GBM ceiling).
2. **Physical consistency pays exactly where it should**: differential outskirt
   masses at high z. Per-radius regression on a tiny annulus mass is unstable
   (log of ~empty bins); the transport model's one consistent history keeps
   annuli/envelopes 2–5× tighter at z≥1.5.
3. **Regression to the mean is invisible in per-galaxy scatter but glaring in the
   planes**: the regression models' predicted distributions are too tight and too
   shallow (M(<30) vs M(50–100) at z=0.4: slope 1.88→1.0–1.1, scatter
   0.206→~0.05), because each quantity is shrunk toward the mean independently.
   The transport emulator propagates real MAH diversity and reproduces the
   relation 3–4× more faithfully — though not perfectly (slope 1.88→1.15,
   scatter 0.206→0.108): its predicted planes are still tighter than truth.
4. **Tier 3 medians tie** — but only the transport model gives monotonic,
   mass-conserving, continuous-radius, differentiable profiles.

**Verdict:** the transport-diffmah universal-θ emulator is the product
configuration: it ties the statistical ceiling on the basic goal and is the only
model that carries the outskirts, the observational planes, and the profile
structure. The remaining +20-point per-galaxy individuality gap (phase 4) and the
plane-scatter shortfall are the same open problem seen from two sides: the
predicted population is not yet diverse enough at fixed halo.

## Files
- `run.py` — scoreboard driver (`demo` self-check: exact LOGO regression;
  e2e totals = SHMR prediction).
- Library: `hongshao/qa.py` (the standard harness; own synthetic `demo`).

Run: `PYTHONPATH=. uv run python experiments/exp31_forward_scoreboard/run.py [--refit]`
