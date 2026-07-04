# exp30 — transport kernel: core-retaining redistribution (feasibility gate)

> **RESULT (2026-07-04, n=45): GATE PASSED — winner `dyntrans` at 7.5%.** The phase-2
> "combined clock" step completed the 2×2 of {clock} × {migrated-width form}, median
> profile max|rel| epoch-avg (real MAH, ALL radii):
>
> | | global clock e^(−Δt/τ) | dynamical clock e^(−Δt/(α·tᵢ)) |
> |---|---|---|
> | **multi-scale width** σ₀ᵢ(t_k/tᵢ)^q | transport **9.1%** | **dyntrans 7.5%** ← winner |
> | **shared width** w₀(t_k/t_obs)^{g_w} | (combined→τ₀=0) | envelope 11.3% |
>
> (additive floor 18.5%, loose-zdep 9.9%, ceiling 0.2%; the 7-param "combined"
> two-clock model collapsed onto envelope, 11.7% — the shared width was its binding
> flaw, not the clock.) **Both ingredients matter**: the dynamical clock (9.1→7.5 at
> fixed width form) and the per-deposit multi-scale migrated width (11.3→7.5 at fixed
> clock). `dyntrans` (4 outer params + NNLS masses) is best at EVERY epoch among
> consistent models and beats loose-zdep by every IC (k_eff=14, rel-RMS 3.4%, ΔAICc
> 495 vs 600). Its fitted clock is maximally simple: **α ≈ 1.04 — the migration
> timescale equals the cosmic time at deposition** (self-similar), q ≈ 1.6. Mass QA:
> apertures ≤1%; R_half envelopes now excellent (M*(>2Re) within 1.8% at every
> epoch); only the fixed-kpc far tail at z=2 remains (M*(>50 kpc) −86%). → Next:
> event-triggered kicks (phase 2.2) and the held-out-epoch test (phase 2.3).

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

## Phase 2.2 — event-triggered kicks (`event_kicks.py`): clean NEGATIVE
Replaced dyntrans's smooth clock with kicks at the real-MAH bursts,
`f_core,i(t_k) = prod (1 − ε₀ s_j)` over events between deposition and observation
(s_j = peak-history steps; dip-recovery discounted by construction). Three consistent
results (n=45): **(1)** pre-test NULL — fitted dyntrans migration speed does not
correlate with MAH burstiness (Spearman ρ=+0.01/−0.13, p≫0.05); **(2)** events
underperform at every threshold (s>0.03/0.05/0.10 → 10.3/10.7/12.1% vs dyntrans 7.5%),
monotonically worse with fewer events (the event product only approaches the smooth
clock as s_min→0); **(3)** no per-galaxy scatter reduction (p75/p90 slightly worse).
Reading: halo-MAH-step timing is a poor proxy for stellar redistribution timing
(dynamical-friction delays ~Gyr; relaxation continues between events) — the
self-similar τ≈tᵢ clock stands. **Keep dyntrans.**

## Phase 2.3 — held-out-epoch generalization (`holdout.py`): the in-sample ranking INVERTS
LOEO (fit 4 epochs, predict the 5th, symmetric aperture pin, no leakage; n=45,
held-out median max|rel| averaged over h ∈ {z=0.7,1.0,1.5,2.0}):

| model | in-sample | held-out | gap |
|---|---|---|---|
| additive | 19.8% | **30.9%** | +11.1 |
| loose-quad | 9.2% | 35.3% | +26.1 |
| dyntrans | **7.5%** | 53.7% | **+46.2** |

**The best in-sample model is the worst predictor.** The discriminator is the mass
parameterization: dyntrans's free NNLS masses (+ migration flexibility) absorb
epoch-specific information and cannot predict an unseen epoch; loose-quad's masses are
parametric (efficiency), so it degrades less despite its z-drifting params; rigid
additive has the smallest gap. NO current model predicts acceptably (all ≥ 30%).
**Conclusion: the free-mass gate and the predictive emulator are different regimes —
phase 3 (parametric masses INSIDE the dyntrans transport structure, ~7 params total,
zero free masses) is required, not optional.** Native total-mass prediction is fine
(dyntrans |dlog M*| 0.06–0.16) — it is the SHAPE that overfits.

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
