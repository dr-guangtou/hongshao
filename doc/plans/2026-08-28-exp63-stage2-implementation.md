# exp63 Stage 2 Implementation Plan — the two-channel mean, fitted at z=0.4 only, predicted elsewhere

> **For agentic workers:** execute inline in the primary session; the multi-start fit runs in the main shell in the background (the standing rule). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build and fit, at $z=0.4$ only, a two-channel deposition model on the exact-integral engine — a compact channel (Sérsic deposit, index near 1, at a few per cent of $R_{200c}$) and an extended channel (cored deposit at $\sim0.1$–$0.3\,R_{200c}$) whose share rises with halo mass — under an objective that can see the innermost aperture; gate it (G2a–G2f); then PREDICT $z=0.7$–$2.0$ by truncating the same integral and report on both samples with the decliner split.

**Architecture:** `model2.py` holds `Spec2` (the parameter vector and its bounds, nested to the incumbent's class) and `predict2(spec2, theta, curves, R, include_epochs)` built on `engine.nodes` / `engine._deposits`-style arrays; `stage2_fit.py` holds the objective (`Problem2`), the multi-start L-BFGS-B fit with a named nested start, the gates, the earlier-epoch predictions and the figures. Nothing from Stage 1 is a fit input; Stage 1 numbers are the starting values and the prior expectations only.

**Tech Stack:** `hongshao.fitting.minimize_loss` (L-BFGS-B, finite differences, per the module's own measurements), `hongshao.objective.Objective` (production and `shells/log`), `exp54/fit.py` constants (`R_GRID`, `I100`, `SIGMA_A`, `L_F_REF`), `engine.py`, `hongshao.qa`.

**Spec:** science plan §2.1, §3 Stage 2, §4 D1–D5 as decided; Stage 1 README (kernel verdict, the modes, the leverage).

## Global Constraints

As before, plus: the compact channel's Sérsic index is bounded to $[0.5, 2.5]$ (Stage 1 rejected $n=4$; the bound is a *measurement*, not a prior); every parameter has a box; a parameter at a bound is reported by name; the nested start reproduces Stage 0's `exact-analytic` product bit for bit before any fit; the fit's objective is frozen after the smoke run; the earlier epochs are never in the loss.

---

### Task 1: `model2.py` — the two-channel model on the engine

**Files:**
- Create: `experiments/exp63_analytic_growth/model2.py`

**Interfaces:**
- Consumes: `engine.nodes`, `engine.log_mah`, `engine.dm_dlnt`, `engine.r200c_of`, `engine.z_of_t`, `engine.log_eps_e2`, `engine.r50_s2`, `exp54/model.cog_truncated`, `exp53/families`.
- Produces:
  - `Spec2` with `theta_names = ("a0", "a_M", "a_z", "a_Mz", "m_half", "d_split", "log_f_c", "b_c", "log_f_e", "b_e", "n_c", "c_e")`, `bounds()`, `unpack(theta) -> dict`, `nested_theta(theta_incumbent) -> theta` (the incumbent embedded: `m_half=-10, d_split=1` so the compact share is 0 everywhere; `log_f_e, b_e, c_e` = the incumbent's; `log_f_c, b_c, n_c` inert), `default_theta()` (Stage 1 starting values: `m_half=13.2, d_split=0.5, log_f_c=log10(0.04), b_c=0, log_f_e=log10(0.2), b_e=-0.5, n_c=1.0, c_e=0.80`, efficiency = the incumbent's).
  - `compact_share(m, theta) = expit((m_half - m)/d_split)` with `m = log10 Mh(t') - 13.5` at the deposit's own time.
  - `predict2(spec2, theta, curves, R, epochs=(0,1,2,3,4), truncate=True) -> (n, len(epochs), len(R))`.
  - `channel_masses(spec2, theta, curves) -> (n, 2)`: total stellar mass deposited by each channel by $z=0.4$ (for the report of the compact/extended share against Stage 1's modes).

- [ ] **Step 1: self-check (the failing test).** `nested_theta` + `predict2` equals `engine.predict(spec_incumbent, theta_incumbent)` to 1e-12 relative at all five epochs on 60 galaxies; `default_theta` changes the profile (max |dlog| > 0.02 dex); mass conservation: $M_*(<10^9\,{\rm kpc})$ equals the sum of both channels' deposited mass to 1e-6; monotone in radius and time; no stellar field.
- [ ] **Step 2: run, expect failure; implement; run, expect pass; commit** (`exp63 Stage 2: the two-channel model, nested to the incumbent`).

### Task 2: `stage2_fit.py` — objective, fit, gates, predictions, figures

**Files:**
- Create: `experiments/exp63_analytic_growth/stage2_fit.py`
- Output: `outputs/stage2_fit.{npz,log}` (+ `.PARTIAL.npz` per start), `figures/exp63_stage2_{residuals,gates,predictions,channels}.{png,pdf}`, `figures/qa/qa_*_exp63_stage2.*`
- Modify: `README.md` (Stage 2 section), `doc/todo.md`

**Interfaces:**
- Consumes: `S0.build`, `engine.build_curves`, `model2`, `fit.{R_GRID, I100, SIGMA_A, L_F_REF}`, `Objective()`, `Objective(quantity="shells", residual="log")`, `hongshao.fitting.minimize_loss`, `hongshao.qa.evaluate`, `stage37_size_epoch.central_error`-style decliner flag (measured $M_*(<4.92)$ fell from $z=2$ to $0.4$).
- Produces: `stage2_fit.npz` with `theta_best`, `theta_names`, `loss_best`, `start_losses`, `theta_nested`, `loss_nested`, `railed`, gate table, per-epoch bias tables on both samples, the decliner split table, `inner_slope_model`.

- [ ] **Step 1: the objective, `Problem2`.** At $z=0.4$ only, on the sane-history + mh-complete mask: $L=\text{score}_A^2+\text{score}_F^2+(\text{score}_S)^2$ with $\text{score}_A$ and $\text{score}_F$ exactly as `exp54/fit.Problem.scores` (amplitude at 100 kpc over `SIGMA_A[0]`; production shape loss over `L_F_REF`) and $\text{score}_S=L_{\rm shells}/L_{\rm shells,ref}$ where $L_{\rm shells}$ is `Objective(quantity="shells", residual="log")` on the amplitude-pinned profiles and $L_{\rm shells,ref}$ is its value for the nested incumbent on the exact engine (frozen after the smoke run and printed) — so every term is ~1 at the null (D4: equal weight). Failures priced as in exp54 (`FAIL` per failing galaxy, never dropped).
- [ ] **Step 2: the fit.** L-BFGS-B via `minimize_loss` with `fd_step=1e-5`, bounds from `Spec2.bounds()`, starts: (i) the nested incumbent (named `nested`; its loss is asserted equal to the null's), (ii) `default_theta`, (iii)–(vi) four jittered copies of `default_theta` (seeded), (vii) `default_theta` with `n_c=2.0`. A `.PARTIAL.npz` after every start. Report per start: loss, iterations, railed parameters. Smoke: every 24th galaxy, `maxiter=30`.
- [ ] **Step 3: the gates** (thresholds frozen here; the null = nested incumbent on the exact engine, evaluated first):
  - G2a inner slope over 2–4.92 kpc (grid points 0, 3): |median − 1.00| ≤ 0.05 and 16–84 % width ≥ 0.7 × 0.38 = 0.27.
  - G2b tercile profiles: median (model−data)/data within ±5 % at every grid radius in each $\log M_h$ tercile, raw and amplitude-pinned.
  - G2c plane slope: `kpc:M(30-50)` vs `kpc:M(<30)` conditional slope within 0.1 of the truth's (1.42 in exp61).
  - G2d no transport / conservation: mass conservation to the truncation boundary (1e-6); the share of $z=0.7\to0.4$ added mass landing beyond 50 kpc within the exp57 G1b tolerance of the data's (reported; a prediction, since $z=0.7$ is not fitted).
  - G2e sizes physical: `log_f_c < log_f_e` and $|{\log_{10} f_c-\log_{10}0.02}|\le0.5$.
  - G2f identifiability: `railed` empty; the compact share at $10^{13.5}$ between 0.05 and 0.95.
- [ ] **Step 4: predictions.** `predict2` at all five epochs; `qa.evaluate` on the QA sample (name `exp63_stage2`); the both-sample bias tables for `kpc:M(<10)`, `kpc:M(<100)`, `kpc:M(50-100)`; the C8 decliner split of the central error at every epoch against the exact-analytic baseline; the Stage 1 leverage correlations reproduced by the model (compact share vs `late`, `f_form`, `logtc` at fixed $M_h$).
- [ ] **Step 5: figures.** `residuals`: the exp63 Stage 0 compare layout, baseline vs Stage 2; `gates`: inner-slope histograms + the plane; `predictions`: B3/B1 vs epoch on both samples, three lines; `channels`: the fitted compact share vs halo mass and the two size laws vs $z$ against Stage 1's required $s^\ast/R_{200c}$.
- [ ] **Step 6: README, todo, commit.**

## Self-review

- Spec coverage: §2.1 model (Task 1), D4 objective (Step 1), gates G2a–G2f (Step 3), Result 2 predictions on both samples with the decliner split (Step 4), the Stage 1 tie-back (Step 4, Step 5 `channels`).
- Placeholders: none — every threshold is a number; the reference for the shells term is defined as the null's value.
- Types: `predict2` returns `(n, len(epochs), len(R))`; `theta` order fixed by `Spec2.theta_names`.
