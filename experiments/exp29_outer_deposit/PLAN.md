# exp29 multi-epoch emulator — exploration plan

## Goal (phenomenological, not physical)
A fast, flexible **forward-model emulator**: halo MAH (dip-free DiffMAH curve, or
its 4 params) → the stellar-mass **CoG** (preferred) or **Σ(R)** at z =
0.4/0.7/1.0/1.5/2.0, for massive galaxies. No physical recipe — the per-step
"fraction" `f` (deposited stellar mass per unit halo-mass increment) and the
deposit "width" are free phenomenological knobs. We optimize math capacity + speed,
**not** physical plausibility. Terminology: *deposit fraction* `f`, *deposit width*
`σ`, *deposit kernel* — never "SFH", "efficiency", "quenching".

## The model (unchanged structure)
Galaxy = cumulative sum of mass-normalized deposits, one per MAH step `i`:
```
dM*_i = f_i · dMh_i                                   # phenomenological fraction
CoG(<R, z_k) = Σ_{i: t_i ≤ t_k} dM*_i · κ(R; σ_i)     # cumulative to epoch z_k
```
with a centred-Gaussian kernel `κ = 1 − exp(−R²/2σ²)` (reproduces single-epoch CoG
to 0.005 dex; the wing only matters for Σ, handled later). Two free functions:
the **fraction** `f` (sets how much mass by each epoch) and the **width** `σ` (sets
where it lands). The single-epoch fit pins a *combination*; multi-epoch must pin
both — that is the open question.

## Key technical lever — the inner solve is convex (NNLS)
CoG is **linear** in the deposit masses `{dM*_i}`. So for any fixed set of widths
`{σ_i}`, the masses that best match all 5 epochs jointly are a **non-negative
least-squares** solve (`scipy.optimize.nnls`) on the stacked 5×24 = 120-vector,
with the cumulative/nested structure baked into the design matrix (deposit `i`
contributes to epoch `k` iff `t_i ≤ t_k`). Log-space is matched by weighting rows
by `1/(ln10·CoG)`. This makes the whole exploration fast: the masses are exact +
convex; only the low-D width law is a nonlinear outer loop.

## Phase 1 — Capacity ceiling: is multi-epoch fundamentally possible? (THE gate)
Per galaxy, joint 5-epoch CoG fit, increasing flexibility:
- **1a** free masses (NNLS) + power-law width `σ(t)=σ₀(t/t_obs)^g` (2 width params).
- **1b** free masses + flexible width (piecewise / spline `σ(t)`, ~4–6 knots).
- **1c** coarse deposit grid (~12 epochs) with free masses **and** free widths
  (still overdetermined vs 120 points) — the near-absolute structural ceiling.

Run on ~15–30 galaxies spanning M\* and assembly time.
**Decision gate 1:**
- ceiling ≪ 0.02 dex → the cumulative-Gaussian structure *can* represent the data;
  the challenge is parametrization/generalization → **Phase 2**.
- ceiling stays high (≳0.05–0.1 dex) even at 1c → the structure is fundamentally
  limited; diagnose which constraint binds (monotone-growth? the inside-out
  increment shape from exp26?) → **Phase 3**.

## Phase 2 — The low-D emulator (if Phase 1 passes)
Find the fewest parameters that approach the ceiling.
- **Fraction `f`:** ladder — constant → two-epoch → 3–4-knot spline in z (or in
  `Mh`, or in halo growth rate). Free per-galaxy first.
- **Width `σ`:** power-law `σ(t)` → 2–3-param `σ(t)` → `σ` tied to a halo scale.
- Each rung: report in-sample 5-epoch RMS **and** the parameter count.

## Phase 3 — Structural alternatives (if the ceiling is high, or to enrich)
- **Break the strict time→radius ordering** (the exp29 cog-extrapolate failure:
  model over-predicts high-z mass because late-arriving mass is forced wide). Let
  `σ_i` depend on the *increment* not just `t`, or add a 2-channel deposit
  (compact + extended per step) so late mass can also land centrally.
- **Alternative kernel** for Σ targets: exponential deposit `κ=1−(1+R/σ)e^{−R/σ}`
  (shallower wing, closed-form CoG, good math) — the clean upgrade from Gaussian,
  **not** free Sérsic (bad math, steepens the core).
- **DiffMAH-parameter regression:** learn a map (4 DiffMAH params + a few shape
  params) → the 5-epoch profile directly; a different emulator flavour.

## Phase 4 — Generalization + population + Σ
- **Epoch hold-out** is the real emulator test (not in-sample): fit a subset of
  epochs, predict the rest. Splits: fit z=0.4 only → predict 0.7–2.0 (done: 0.34
  dex, the hard baseline); fit endpoints {0.4, 2.0} → predict the middle
  (interpolation); leave-one-epoch-out.
- **Population / shared kernel:** scale the winner to ~2400 galaxies, few global
  params + per-galaxy halo input. Does the per-galaxy capacity survive sharing?
- **Σ(R) check:** does a model that matches all CoGs also match the Σ profiles, or
  is the Gaussian wing fatal (→ exponential kernel)?

## Validation discipline (cross-cutting)
- Metric: per-epoch CoG RMS (dex), decomposed into **total-mass** error and
  **shape** error (offset removed) — as in cog_extrapolate.
- Always small-scale (10–30 galaxies) before population.
- Report parameter counts and hold-out, not just in-sample, so we never mistake
  flexibility for a working emulator.
- A `ponytail:` note on any deliberate cap (deposit grid coarseness, kernel choice).

## Status / decision log
- **Phase 1 — GATE 1 PASSED (n=20).** Free per-step fraction (NNLS) + power-law
  width fits all 5 CoGs jointly to **0.032 dex** median (spline width 0.028 — width
  law is NOT the bottleneck). Hold-out (predict one fully held-out epoch, z=0.4
  observed) = **0.038 dex** ≈ in-sample → free-fraction *generalizes*, not
  overfitting. Multi-epoch is representable AND predictable. `capacity_test.py`.
- **Phase 2 — compact emulator exists (n=20).** Smooth K-knot fraction f(t) +
  power-law width: 3-knot (5 par) 0.041, 4-knot (6 par) 0.040, → NNLS ceiling
  0.032. **~5–6 params/galaxy reach ~0.04 dex.** The 2-knot (old two-epoch) reaches
  0.046 *when fit jointly* — vs 0.34 frozen-from-z=0.4: the fraction was never too
  rigid, a single epoch just can't constrain it. `emulator_param.py`.
- **Phase 4 — a universal forward emulator works, at ~0.11 dex (n=150, 75/75
  train/test; `population.py`).** One shared K=4-knot fraction f(t) + power-law width
  maps each galaxy's dip-free MAH → its 5-epoch CoG (total pinned to z=0.4 M*),
  held-out **test 0.112 dex** mean (per-epoch 0.089/0.100/0.105/0.138/0.127; high-z
  worst). Comparable to the single-epoch data-driven emulator (0.116) and exp25
  Phase-3 universal kernel (0.080).
  - **But the fraction is NOT universal and NOT halo-predictable.** Per-galaxy log-f
    knots scatter ±0.5–0.75 dex with **r(logMh)≈r(logM*)≈0**; logMh-conditioning
    helps +0.007 dex (nil). Width ties weakly to logMh (σ₀ r=0.22, g r=0.15). So the
    cost of universality (per-galaxy 0.048 → shared 0.11) is **genuine
    galaxy-to-galaxy timing scatter orthogonal to halo/stellar mass** — the scatter
    in *when* stars accrue relative to halo growth.
  - **Decision:** a forward emulator exists (~0.11 dex). To beat it we must predict
    the per-galaxy fraction from richer inputs (R50 for width; MAH-shape / formation
    time for f), else ~0.11 is the floor (intrinsic SHMR-timing scatter). → Phase 4d.
- **Completion mode** (observe ≥2–3 epochs, fit f) reaches the ~0.04 ceiling
  regardless — that path is already validated (Phase 1 hold-out).
- **Phase 4d — conditioning helps: forward floor 0.11 → 0.092 dex (n=200; `phase4d.py`).**
  Conditioning the shared model on **R50** (→ width) and **halo formation time t50**
  (→ fraction): U 0.111 → W(R50) 0.105 → F(t50) 0.100 → **WF(both) 0.092** test;
  per-epoch the gain is largest at low z (z=0.4: 0.088→0.062). Remaining gap to the
  per-galaxy ceiling (0.048) is scatter not captured by these observables.
  - **Methodological note:** per-galaxy param regression R²≤0.05 (params are
    degenerate, so param-space correlations understate the signal), yet the *direct
    forward fit* still improves — fit the forward RMS, not the parameters.
  - Figures: `exp29_p4_forward_qa` (universal model vs TNG, test galaxies),
    `exp29_p4_rms_summary` (RMS by epoch across all models + ceilings),
    `exp29_p4_param_predictors` (weak param↔observable ties). Results cached in
    `outputs/p4d_cache.npz` (rerun figures with no `--refit`).
- **Phase 4e — LEGAL halo-only forward (n=200; `phase4e.py`). CRITICAL re-scope.**
  Phase 4d's R50 (galaxy size) and M*(z=0.4) pin are stellar/galaxy quantities, NOT
  available on a fresh N-body run — illegal emulator inputs. Re-done with halo-only
  inputs: width ← `c_200c` (concentration, replaces R50), fraction ← `t50`. Test:
  U 0.111 → Lc 0.106 → Lf 0.100 → **L both legal 0.097** (vs illegal WF 0.092 —
  R50 added almost nothing over halo concentration). The amplitude M*(z=0.4) is a
  **separable SHMR** (logM*=5.71+0.44·logMh, test scatter **0.086 dex** from logMh
  alone). So the honest end-to-end halo→profile forward emulator ≈ **shape 0.097
  (MAH+c200c+t50) ⊕ amplitude 0.086 (SHMR) ≈ ~0.13 dex**.
- **QA (`qa_diag.py`):** metric = per-galaxy RMS over 5 epochs × 24 CoG radii of
  log10(model/data) of M*(<R), median over galaxies. Shared-WF **train 0.090 ≈ test
  0.095** (not overfitting); per-galaxy ceiling 0.048. Figures `exp29_qa_rms_hist`
  (train≈test vs per-galaxy ceiling) and `exp29_qa_train_vs_pergal` (per-galaxy fits
  track the data; the universal model misses the high-z amplitude for massive BCGs).
- **Known minor issue:** the Gaussian deposits leak ~8% mass beyond the 148 kpc
  aperture → model CoG at 148 kpc is ~0.035 dex low (pin normalization to the
  aperture to fix).
- **Metric caveat (important; `best5_qa.py`).** The averaged log-RMS understates
  per-galaxy error: even the 5 *best* per-galaxy fits (log-RMS 0.021–0.026) carry
  **max relative errors of 21–38%** locally. It's the *averaging over 120
  radius×epoch points* that hides it (log error ≈ relative error, so not a
  log-vs-linear issue). The stacked relative-residual map shows the errors are
  **structured, not noise**: z=0.4 good (≲5%); **z≥1.5 an S-shape — model
  under-predicts the centre (−6 to −14%), over-predicts mid R (+10 to +13%),
  under-predicts the outskirts** (the centred-Gaussian can't hold the compact
  high-z shape — same lesson as the Σ-profile cliff); inner ≤3 kpc large (softening
  + flat core). Figures `exp29_best5_linear`, `exp29_residual_map`.
  - **Evaluation upgrade needed:** report radius×epoch-resolved relative residuals +
    percentiles (max / 90th), exclude or downweight inner ≤3 kpc — not a single mean.
- **Next candidates:** (a) the high-z profile *shape* is the real target (the
  centred Gaussian over-spreads high-z mass into mid R) — revisit the kernel for the
  compact high-z regime; (b) the aperture-leak normalization fix; (c) full ~2400 +
  end-to-end SHMR.
- **Deferred:** Σ-profile check (does CoG-matching give acceptable Σ, or need the
  exponential kernel); DiffMAH-param regression flavour.
