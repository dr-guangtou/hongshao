# exp30 — transport kernel: core-retaining redistribution (feasibility gate)

> **RESULT (2026-07-02, n=45): GATE PASSED.** Median profile max|rel| epoch-avg,
> real MAH, ALL radii: additive floor **18.5%** → transport **9.1%** (envelope 11.3%),
> vs loose-zdep reference 9.9% and per-epoch ceiling 0.2%. The global-clock transport
> (τ≈2.9 Gyr, q≈1.5) **halves the additive consistency floor and edges past the
> inconsistent 15-param loose-zdep fit with a genuine consistent history** (4 outer
> params + NNLS masses), flat 8–9% at every epoch. Mass QA: cumulative apertures
> ≤~1% everywhere; R_half envelopes good (M*(>2Re) within ~7%); the fixed-kpc far
> outskirt at high z remains unfixed (M*(>50 kpc) −91% at z=2) — the global-clock
> variant migrates too little by z=2, exactly what the envelope variant fixed on the
> BCG subsample (n=5 z=2: 29→10%). → Redistribution IS the missing freedom; next is
> an event-driven / combined form and then the population model.

## Why this exists
exp29 ended with a proven structural limit: **no consistent additive Gaussian-sum
history can reach the per-epoch ceiling.** The free-mass NNLS floor showed each epoch
fits ALONE to ~0.2% max-rel, but one shared mass vector (a single additive history)
caps the joint fit at ~12–18%. Four independent exp29 findings point at the same
missing ingredient:

1. **Additivity is the binding limit**, not the mass/width/efficiency parameterization
   (`nnls_floor.py`).
2. **The data demand de-concentration**: the pre-z=2 mass must occupy ~2x larger radius
   by z=0.4 (5-kpc fraction 0.64→0.44; `param_trends.py`).
3. **Puffing failed diagnostically**: widening whole deposits empties the core the data
   keep → redistribution must be **core-retaining** (`puff_fit.py`).
4. **The real MAH carries merger events and the missing high-z outskirt is ex-situ
   territory** (`real_mah_test.py`, `mass_qa.py`).

The physical process with all these signatures: mergers/accretion **redistribute
existing stars outward while retaining a core** — a transport process, not addition.

## The model (minimal, nests everything tried before)
Deposit `i` (mass `dM_i`, laid at `t_i` with width `σ_0,i = σ_0 (t_i/t_obs)^g`),
observed at `t_k`, is split into a retained core + a migrated envelope:

```
CoG_i(R, t_k) = f_core(Δt)·[1 − e^(−R²/2σ_0,i²)] + (1 − f_core(Δt))·[1 − e^(−R²/2σ_w,i²)]
f_core(Δt)   = exp(−Δt/τ),          Δt = t_k − t_i
σ_w,i(t_k)   = σ_0,i (t_k/t_i)^q
```

- `τ→∞` (f_core≡1): the exp29 **additive** model exactly.
- `τ→0` (f_core≡0): pure **puffing** (ratio law) exactly.
- In between: the **core-retaining split** — the one freedom neither had.
- Mass conserved per deposit; observation-time dependence only through Δt → a genuine
  consistent (non-additive) history, unlike loose-zdep's per-epoch parameter drift.
- For fixed (σ_0, g, τ, q) the CoG is **linear in the masses** → convex NNLS inner
  solve (the exp29 floor machinery); only 4 params in the outer loop.

## The gate
Same method, same metric, same galaxies (exp29 corrected standard: real de-dipped MAH,
ALL radii, n=45 from `integrated_check.npz`):

- `alone`     : per-epoch free-mass NNLS (the ceiling, ~0.2–2%)
- `additive`  : joint free-mass NNLS, f_core≡1 (the exp29 floor, ~12–18% max|rel|)
- `transport` : joint free-mass NNLS + (τ, q)

**Decision:** transport ≪ additive and competitive with the loose-zdep ~10% → core-
retaining transport is the missing freedom → build it out (event-driven kicks at the
real-MAH merger bursts, then the population/forward model). Transport ≈ additive →
the redistribution form is wrong or the limit is deeper → stop before over-investing.

Evaluation per repo standard: profile max|rel| over ALL radii + `mass_qa.evaluate()`
(kpc + R_half aperture/envelope masses), figures surfaced with paths.

## Fair model comparison (`ic_compare.py`)
Effective parameters: outer params + **active (nonzero) NNLS masses** (nnls returns
exact zeros; active-set size is the standard effective-df estimate) + 5 aperture pins
for the exp29 parametric models. Median k_eff / rel-RMS / ΔAIC/ΔAICc/ΔBIC (n=45):
transport **k_eff=14, 4.5%** dominates loose-quad (k_eff=20, 5.1%) on BOTH parsimony
and fit — every IC prefers it (ΔAICc 555 vs 600). Within the consistent-history class
the IC ranking is transport < envelope < puff-ratio < additive on every criterion.
Caveats: CoG points are correlated (n=120 overstated; n_eff=40 sensitivity keeps the
ordering) and ICs cannot see the consistency requirement — alone/independent "win"
only by abandoning a single history.

## Files
- `transport_floor.py` — the gate: three-mode comparison + figures; `demo` self-check.
- `ic_compare.py` — effective-parameter accounting + AIC/AICc/BIC table + figure.

Run: `PYTHONPATH=. uv run python experiments/exp30_transport_kernel/transport_floor.py [n] [--refit]`
