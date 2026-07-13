# exp36 — beyond the single-width transport kernel (two-channel + new DOF)

> **RESULT (2026-07-14, z=0.4, n=2397, 10-fold CV, held-out 148-pinned shape
> max|rel| R>5 kpc).** The two-channel deposit closes almost the entire
> physicality tax INSIDE the (3.0, 4.0) physical box, and the phase-0
> conditioning earns its slots:
>
> | model | params | in-sample loss | held-out shape | massive dlog f148 |
> |---|---|---|---|---|
> | exp35 z04-slope (single width) | 10 | 0.1756 | 19.6% | +0.0149 |
> | 2ch-global | 8 | 0.1590 | 16.9% | +0.0145 |
> | 2ch-slope | 13 | 0.1566 | 16.8% | +0.0053 |
> | **2ch-cond (B)** | 23 | **0.1543** | **16.4%** | **+0.0024** |
> | marks | | | 16.1 (unconstr.) / 15.6 (statistical) | 0 = data |
>
> For the first time in the family's history the width law sits OFF the
> rails (log_s0 ~ 2.05 ~ 110 kpc, g ~ 3.1 interior): the wide role moved to
> the ex channel (log_s0_ex at the 3.0 box edge, carrying f_ex of the mass),
> which is what the bounds-stress test said the loose bounds were faking —
> and unlike the stress fit, the loss gain comes WITH the observables: the
> massive-end aperture fraction now tracks the data (dlog f148 +0.015 ->
> +0.002, figure), and the in-sample loss (0.1543) beats the stress fit
> (0.1676) without any horizon escape. The fitted split rises f_ex = 0.45 ->
> 0.75 over logMh 13 -> 15 — a free parameter landing on a plausible
> ex-situ fraction trend. Remaining gap to the statistical wall: 0.8 points.
> NEXT: the multi-epoch fit + differential-deposition test, then (C) the
> statistical dressing.

Status: (A)+(B) built and validated at z=0.4 (2026-07-14). Goal reframed
by the user: no physicality mandate — the primary goal is the best
parameterized model of the [MAH, halo properties] -> [stellar profile / CoG]
statistical relation. New degrees of freedom that improve on the transport
kernel should be explored; physical interpretability is a bonus, not a
constraint.

## 1. The old model, reviewed (exp29 -> 30 -> 32/33 -> 35)

The transport kernel deposits each MAH step `dM_i` at time `t_i` as a
centred Gaussian of width `sigma_0 (t_i/t_obs)^g`, then lets it relax into a
retained core + migrated envelope:

    CoG_i(R, t_k) = f_core [1 - e^(-R^2/2 sigma_0i^2)]
                  + (1-f_core) [1 - e^(-R^2/2 sigma_wi^2)]
    f_core = exp(-dt / (alpha t_i))   (dynamical clock, fitted alpha ~ 1)
    sigma_wi = sigma_0i (t_k/t_i)^q   (multi-scale migrated width)

Population form (exp33/35): 5 physical params (alpha==1, lognormal
efficiency f(z) with peak mu and width sig) + optional linear logMh slopes
(10 params); exp35 normalizes to M(<500 kpc) so the beyond-aperture budget
is data, not a hiding place.

**What it does well (keep):**
- Single-epoch form efficiency: fitted to z=0.4 alone it TIES the ~90-param
  statistical PCA-Gaussian (16.1% vs 15.6% held-out pinned shape R>5) with
  14 params.
- The ONLY model with a passed out-of-model physics test: the fitted width
  law reproduces the measured differential-deposition curve (massive
  tercile z0.7->0.4: 0.37/0.11 measured vs 0.40/0.12) and its mass trend,
  and predicts the aperture fraction to 0.004-0.016 dex.
- Mass conservation + a consistent history across epochs (no per-epoch
  re-fitting), linear-in-masses CoG (convex inner solve).

**The measured problems:**
- P1 — the consistency tax: one consistent history costs ~3-5 points vs
  independent per-epoch fits (multi-epoch 19.1% unconstrained / 20.5%
  exp35-physical vs the 16.1% single-epoch form; statistical mark 15.6%).
  exp29 proved this is structural (free-mass alone 0.2% -> joint 12-18%).
- P2 — the width rail is a horizon escape: log_s0/g rail at ANY box; the
  4.3% loss headroom is reachable by a single global width law only by
  pushing deposits past whatever normalization radius exists (stress test:
  z[1,2) visibility 0.95 -> 0.62, observables unimproved).
- P3 — the massive end wants more outward transport: model f148 0.875-0.920
  vs data 0.83-0.883; one width exponent must serve the whole population.
- P4 — the efficiency form is too rigid: the lognormal f(z) peak drifts
  between z04-only and multi fits (peak z 2.7 vs 4.0) — the only ingredient
  that does not land in one basin.
- P5 — halo information unused: the population theta depends on logMh only;
  c200c is measured to carry independent profile information (+5% CRPS on
  DiffMAH features, only ~25% MAH-determined; exp16/18) and the transport
  family never sees it.
- P6 — no scatter layer: the kernel is deterministic; its plane fidelity is
  the mean's (E/floor 2.4-4.6), and it cannot draw populations. As a model
  of the STATISTICAL relation it is only half a model — exp37 showed the
  other half (heteroscedastic cores + AR(1) latent) is what passes the 2-D
  planes.
- (P7, known caveat, not chased: the z=2 fixed-kpc far-outskirt tail is an
  absolute-radius artifact of compact progenitors; Re-relative views are
  fine. exp29 lesson.)

## 2. The current plan (from the exp35 verdict), reviewed

Two-channel deposit: split each deposit into a compact in-situ channel and a
wide ex-situ channel with a mass-dependent split, +2-3 params on the exp35
machinery; judged by f148(logM*) and the differential-deposition curve; box
stays (3.0, 4.0). — Still the right FIRST move: it is the minimal structural
DOF that decouples P2/P3 (the massive end widens through the split, not by
dragging the global width law past the horizon). But it does not touch P1,
P4, P5, or P6.

## 3. Proposed design space (this experiment)

- **(A) Two-channel deposit** (the plan; targets P2+P3). Channel widths:
  sigma_in = compact (own small log_s0_in, g_in ~ 0), sigma_ex = wide with
  the multi-scale migration; split f_ex(logMh, t_i) with 2-3 params
  (amplitude + logMh slope [+ time tilt]).
- **(B) theta(halo) generalization** (targets P5): add c200c slopes to the
  population theta alongside logMh (evidence-backed, +5 params, composable
  with A). Optionally MAH-shape terms (late), gated by a phase-0 signal.
- **(C) Transport spine + statistical dressing** (targets P1+P6; the
  emulator-goal move): use the (A+B) kernel CoG as the MEAN, then fit an
  exp37-style compressed residual layer (pooled-basis scores from
  [DiffMAH(4), c200c], heteroscedastic, AR(1) in epoch) ON TOP. The spine
  carries the physics (differential deposition, mass conservation of the
  mean); the dressing pays the consistency tax and makes the model
  generative (planes, draws). Ablation guard: the dressing on a FLAT spine
  (no kernel) must do WORSE, else the spine earns nothing.
- **(D) Flexible efficiency** (targets P4): replace the lognormal f(z) with
  a monotone low-order spline (3-4 knots). Total-norm prices the deletion
  channel, but watch the rail lesson: run the bounds-stress test on any new
  freedom before reading it.
- **(E) Step-conditioned split** (parked pending phase 0): let f_ex depend
  on the MAH step size (merger-ness). exp30's event-kick negative and the
  burstiness nulls tested migration TIMING, not deposit destination — but
  the cheap correlation pre-test must fire before this is built.

## 4. Phase 0 — residual anatomy (run FIRST, the exp30 pre-test lesson)

Before adding any DOF, measure what is left to win:
1. Correlate exp35 multi-slope held-out residual profiles (per galaxy, per
   radius) with the single-epoch statistical emulator's residuals: the
   SHARED component is the information limit — no kernel DOF can win it.
2. Regress the exp35 residual (compressed to a few PCs) on the candidate
   levers — c200c, burstiness, MAH step stats, f148 residual — with shuffle
   controls: each proposed DOF (A/B/E) must show a pre-signal before it is
   built.
3. Refit the massive tercile alone with the exp35 form (does a
   mass-restricted single-width fit reach the massive-end f148? If yes the
   split is about population coupling, not form).

## Phase 0 RESULTS (2026-07-14, `phase0.py`, n=2397, z=0.4)

1. **The wall is lower for the exp35 variant — ~half the residual variance
   is winnable.** Per-radius correlation of the 148-pinned shape residuals
   (transport vs statistical mode-3 OOF, R>5 kpc): multi-slope 0.67/0.68/0.71
   (min/med/max), z04-slope 0.69/0.70/0.75 — below the exp33 record
   (0.82-0.89, measured for the exp32 variant). Shared variance ~45-56%:
   the total normalization changed the residual structure, and the dressing
   (C) has real unshared material to work with.
2. **The unshared residual is organized by formation time and concentration,
   not by merger burstiness.** PC1 carries 97% of the unshared variance;
   shuffle-controlled R^2 on PC1: fz2 0.244*, t50 0.165*, c200c 0.148*,
   burst 0.069*, dmah_late 0.035*, logmh 0.001 (already conditioned on).
   -> (B) should condition theta on c200c AND an epoch-matched formation-time
   summary (fz2 or t50), not just logMh. Note the kernel consumes the full
   MAH as deposits, yet its population theta ignores formation time — that
   is the measured omission. (E) stays parked: the burst signal is weak.
3. **Massive-tercile probe: decoupling closes 60% of the f148 gap; the form
   still rails.** The single-width exp35 form refit to the massive logM*
   tercile alone: dlog f148 +0.0149 (population multi-slope) -> +0.0059 dex,
   with log_s0 railed at 3.0 and g ~ 3.97 even massive-only, and no shape
   improvement (20.4 -> 21.7% in-sample). -> the split (A) is justified
   twice over: as population decoupling (most of the f148 gap) AND as form
   freedom (the residual rail).

Build plan refined by phase 0: (A) two-channel deposit + (B) theta
conditioned on [logMh, c200c, fz2-or-t50] + (C) the statistical dressing,
in that order, each against the fixed criteria below.

## The build (`run.py`, 2026-07-14)

    CoG_i = (1 - f_ex) B(log_s0, g, q; t_i) + f_ex B(log_s0_ex, g, q; t_i)
    f_ex  = expit(fa + fb mh_std)

- The ex channel shares the deposition-time scaling (g) and migration (q,
  alpha == 1) and differs only in its width scale `log_s0_ex` — bounded by
  the SAME physical box (<= 3.0): the point is that the wide role moves to a
  channel that carries only f_ex of the mass, so the in-situ law can go
  narrow instead of railing. Everything else (lognormal f(z), M(<500)
  normalization, loss, folds) is exp35's, reused via the spawn-safe
  worker pattern from phase 0.
- Nesting contracts (demo-asserted, exact): f_ex = 0 reproduces
  `exp35.model_cogs_total` to 1e-12; equal channel widths reduce to the base
  model at any f_ex; the theta layer nests global -> slope -> cond.
- Variants: `2ch-global` (8 params), `2ch-slope` (+logMh slopes on base5,
  13), `2ch-cond` = (B): slopes on the standardized phase-0 conditioning
  vector [logMh, c200c, fz2] (23). 10-fold CV, warm-started from the full
  fit, exp32/33 fold convention.

## 5. Judged by (fixed before any fitting)

- Held-out (exp32/33 folds) 148-pinned shape max|rel| R>5 at z=0.4 and
  epoch-avg, against the marks: 15.6% (statistical), 16.1% (transport
  z04-slope), 19.6/20.5% (exp35 physical). Quote all-R too.
- f148(logM*): massive-end amplitude (data 0.83); differential-deposition
  curve (0.37/0.11 massive tercile) — the physics tests stay.
- If (C) is built: generative planes per epoch (energy/floor) and the
  growth plane, against exp37's records (draws ~1x floor).
- Bounds-stress test on every new width/efficiency freedom.
