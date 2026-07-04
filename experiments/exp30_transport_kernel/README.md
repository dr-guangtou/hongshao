# exp30 — transport kernel: core-retaining redistribution (feasibility gate)

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

## Files
- `transport_floor.py` — the gate: three-mode comparison + figures; `demo` self-check.

Run: `PYTHONPATH=. uv run python experiments/exp30_transport_kernel/transport_floor.py [n] [--refit]`
