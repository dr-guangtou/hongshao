# exp63 Stage 0 Implementation Plan — the exact-integral engine and the incumbent re-baselined

> **For agentic workers:** execute inline in the primary session (long evaluations in the main shell — see memory: subagent background processes die with their host). Steps use checkbox (`- [ ]`) syntax for tracking. One stage per plan: Stage 1's plan is written after Stage 0's measurements exist, because the science plan makes Stage 2's form contingent on Stage 1's gate G1.

**Goal:** Replace the incumbent's 72-step deposition sum by the exact integral along the analytic DiffMAH curve, prove the replacement reproduces the old sum on the old grid, and re-measure the incumbent's parameters on the exact integral through the standard QA battery (Result 0), including the D3 truncation measurement.

**Architecture:** One experiment-local module `engine.py` holds the halo curve (four DiffMAH numbers), the closed-form cosmology, the quadrature nodes shared by all five epochs, and a batched `predict` that evaluates every galaxy at once. The incumbent's `Spec`, efficiency law, size law and truncated kernel are REUSED from `exp54/model.py` — Stage 0 changes the integration, nothing else. Two stage scripts: `stage0_engine.py` (gates G0a–G0d + the D3 measurement, refuses to write unless every gate passes) and `stage0_rebaseline.py` (the three products through `hongshao.qa.evaluate`, both samples, the compare figure).

**Tech Stack:** numpy, scipy (`gammaincc`, `expit`), existing `hongshao.qa`, `exp54/model.py`, `exp53/families.py`, `exp57/stage0_cost.build` for the sample.

**Spec:** `doc/plans/2026-08-28-exp63-analytic-growth-model.md` (§3 Stage 0, §4 decisions D1, D3) and `doc/tech_note/04_analytic_deposition_integral.md` (Eqs. 2–5).

## Global Constraints

- `export HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 PYTHONPATH=.`; run with `uv run python -u`.
- `import fit` before `halo`/`model` (exp54's `fit.py` must win the module name).
- `--smoke` runs every 24th galaxy and never writes production filenames (`*_smoke.*`).
- No stellar quantity may enter the engine (Contract B); the engine reads `HaloCurve` only.
- Every gate refuses to write outputs on failure ("REFUSING TO WRITE", not a warning).
- Figures via `hongshao.plotting.set_style/save_fig`, PNG + PDF, usetex-safe text, Okabe-Ito epoch colours as in `hongshao/qa.py`.
- Snake_case everywhere; every number printed with what it measures and its reference.

---

### Task 1: `engine.py` — the halo curve, the cosmology, the quadrature, the batched predictor

**Files:**
- Create: `experiments/exp63_analytic_growth/engine.py`
- Create: `experiments/exp63_analytic_growth/README.md` (header + Stage 0 section stub filled at Task 3)
- Test: the module's `__main__` self-check (project convention: gated self-checks, no pytest tree)

**Interfaces:**
- Consumes: `exp54/model.py` (`Spec`, `log_eps`-equivalent formulas, `cog_truncated`, `M_PIV`), `exp53/families.py` (`cog_unit`), `hongshao.diffmah.MAH_K`, `hongshao.tng_data.load_cosmic_time`.
- Produces:
  - `HaloCurve(row, index, logmp, logtc, early, late, t_cat, r200c_cat, t_first_good)` — a dataclass with NO stellar field; `logmp` anchored at `LOGT0 = log10(13.8027 Gyr)` (official DiffMAH, z=0).
  - `build_curves(recs) -> list[HaloCurve]` from exp54 `Halo` records + `diffmah_combined.fits` official columns; asserts `diffmah_source == "official"` for every galaxy and that the curve reproduces `rec.logmh` to 1e-5 dex.
  - `log_mah(lt, hc)`, `dm_dlnt(lt, hc)` (arrays over `lt = log10 t`).
  - `z_of_t(t)`, `rho_c(z)`, `r200c_analytic(logm, z)` (physical kpc), `r200c_of(hc, lt, mode)` with `mode in {"analytic", "catalog"}` (catalog = log-log interpolation of `r200c_cat`, clamped).
  - `nodes(t_start, t_anchor, n_early=8, n_node=24) -> (lt, w, epoch_of_panel)`: Gauss–Legendre nodes in ln t on `n_early` panels from `t_start` to `t_anchor[-1]` (z=2) and one panel per anchor interval after; `panel_upper_epoch[k]` = index of the last panel included in epoch k's integral.
  - `predict(spec, theta, curves, R, mode="analytic", truncate=True, t_start=T_START, n_early=8, n_node=24) -> (n, 5, len(R))` cumulative stellar mass in Msun at the five anchor epochs, batched over galaxies.
  - `discrete_sum(spec, theta, hc, R, sub=1) -> (5, len(R))`: the incumbent's end-of-step Riemann sum on the record's own grid (catalog R200c, record's kept steps), each step split `sub` times on the analytic curve — the bridge to `exp54/model.forward`.
  - `mass_beyond_truncation(spec, theta, curves, t_start) -> (n,)`: for untruncated kernels, the fraction of z=0.4 deposited stellar mass beyond 3 R200c(t') of its own deposit (the D3 number).
  - `T_START = t(snap 1) = 0.2708 Gyr`; `ANCHOR_SNAP = (72, 59, 50, 40, 33)`.

- [ ] **Step 1: Write the self-check first (the failing test).** In `engine.py::__main__`: (a) `build_curves` on every 40th galaxy reproduces `rec.logmh` to 1e-5 dex; (b) `z_of_t` vs `hongshao.tng_data.TNG_COSMO.age` inverse to 1e-5; (c) `discrete_sum(sub=1)` equals `exp54/model.forward` to 1e-6 dex on galaxies without a mid-history R200c gap; (d) `predict(mode="catalog")` vs `discrete_sum(sub=16)` within 2e-3 dex and closer than `sub=8`; (e) `predict` with `n_node=24,n_early=8` vs `n_node=32,n_early=16` within 1e-6 dex; (f) a Contract-B check: `HaloCurve` has no field named in `("m500","data","logms","cog","mstar")`; (g) the untruncated power-law closed form of tech note 04 Eq. 5 for gompertz_log, and the Mellin constants for moffat and sersic, to 1e-8.

- [ ] **Step 2: Run it; expect failure** (`ModuleNotFoundError`/`AttributeError` before the implementation exists).

- [ ] **Step 3: Implement** `HaloCurve`, `build_curves`, the cosmology, `nodes`, `predict`, `discrete_sum`, `mass_beyond_truncation`. Formulas, verbatim from tech note 04:
  - `alpha = early + (late-early)*expit(MAH_K*(lt-logtc))`; `dalpha = (late-early)*MAH_K*s*(1-s)`; `log10 M = logmp + alpha*(lt-LOGT0)`; `dM/dlnt = M*(alpha + dalpha*(lt-LOGT0))`.
  - `a(t) = (Om/OL)^(1/3) sinh^(2/3)(1.5 H0 sqrt(OL) t)`, `H0 = 67.74/977.792 Gyr^-1`, `Om = 0.3089`.
  - `rho_c(z) = 2.775e11 (H0/100)^2 [Om(1+z)^3 + OL]` Msun/Mpc^3; `R200c = (3M/(4 pi 200 rho_c))^(1/3) * 1e3` kpc.
  - efficiency and size law copied from `exp54/model.log_eps` (E2 branch) and `r50_of` (S2 branch) so that `Spec.unpack` order is honoured; kernel via `M.cog_truncated(spec.family, shape, r50, spec.trunc_C*r200, R)` when `truncate`, else `families.cog_unit(spec.family, r50, shape, R)`.
  - batched evaluation: arrays `(n, N)` for `lt, lm, z, r200, dm, r50`; kernel matrix `(len(R), n*N)`; per-epoch sums via `panel_upper_epoch`.

- [ ] **Step 4: Run the self-check; expect every assertion to pass** and print the seven numbers.

- [ ] **Step 5: Commit** `engine.py` + README stub: `exp63 Stage 0: the exact-integral deposition engine (self-checked against exp54/model.forward)`.

### Task 2: `stage0_engine.py` — the gates and the D3 measurement

**Files:**
- Create: `experiments/exp63_analytic_growth/stage0_engine.py`
- Output: `outputs/stage0_engine.{npz,log}`, figure `figures/exp63_stage0_engine.{png,pdf}`

**Interfaces:**
- Consumes: `engine.build_curves/predict/discrete_sum/mass_beyond_truncation`, `exp57/stage0_cost.build` (records, data, mask, incumbent theta, base spec).
- Produces: `stage0_engine.npz` with keys `gate_names, gate_values, gate_thresholds, passed, timing_full_sample_s, frac_beyond_3r200_gompertz (n,), d3_decision (str)`.

- [ ] **Step 1: Write the gate table** (thresholds frozen here): G0a max |log10(discrete_sum(1)/forward)| ≤ 1e-6 on no-gap galaxies (full sample); G0b |predict(catalog) − discrete_sum(16)| median ≤ 2e-3 dex and the sub=2,4,8,16 sequence decreasing; G0c closed forms ≤ 1e-8; G0d wall time of one full-sample `predict` (5 epochs, 2397 galaxies), reported, no threshold. D3: fraction of z=0.4 deposited mass beyond 3 R200c for the untruncated gompertz_log kernel — decision rule from the spec: keep truncation iff population median > 1 per cent.
- [ ] **Step 2: Run `--smoke`**; expect the gates to run end to end below one minute and write `*_smoke.*` only.
- [ ] **Step 3: Run full**; expect all gates to pass and the D3 number printed with its rule applied.
- [ ] **Step 4: Figure**: (a) log10(72-step / exact) vs radius, median and 16–84 %, z=0.4 and z=2; (b) the sub-step convergence sequence; (c) histogram of the D3 fraction.
- [ ] **Step 5: Commit**: `exp63 Stage 0: engine gates G0a-G0d pass; D3 measured`.

### Task 3: `stage0_rebaseline.py` — Result 0, the incumbent on the exact integral

**Files:**
- Create: `experiments/exp63_analytic_growth/stage0_rebaseline.py`
- Output: `outputs/stage0_rebaseline.{npz,log}`, `figures/qa/qa_*_exp63_{72step,exact-catalog,exact-analytic}.*`, `figures/exp63_stage0_compare.{png,pdf}`
- Modify: `experiments/exp63_analytic_growth/README.md` (Stage 0 section), `doc/todo.md` (Stage 0 ticked + review)

**Interfaces:**
- Consumes: `engine.predict` (two modes), `exp54/model.forward` via `S35.StackedProblem.predict` (the 72-step product, exactly as exp61 built it), `hongshao.qa.evaluate`, exp61's both-sample bias-table pattern (`stage3_qa.py:188-260`).
- Produces: `stage0_rebaseline.npz` with `names`, `bias_<model>_<key>` (5,) for `kpc:M(<10)`, `kpc:M(<30)`, `kpc:M(<100)`, `kpc:M(50-100)`, on `all` and `fitmask`, `mr_out_<model>`, `inner_slope_<model>` (n,) at z=0.4, `inner_slope_data`.

- [ ] **Step 1: Products.** `72step` = `S35.StackedProblem(base, recs, data, mask).predict(theta)`; `exact-catalog` = `engine.predict(mode="catalog")`; `exact-analytic` = `engine.predict(mode="analytic")`; truncation as decided by Task 2's D3 rule (both variants computed if the rule is within a factor 2 of its threshold, and both reported).
- [ ] **Step 2: Gate.** REFUSE unless `exact-catalog` reproduces Task 2's G0b product bit-for-bit (same code path) and `72step` reproduces exp61's incumbent tier-3 medians (0.2765 at z=0.4 on the QA sample) to 1e-4.
- [ ] **Step 3: Battery.** `qa.evaluate` on the common good sample for the three products; both-sample bias tables for `kpc:M(<10)` and `kpc:M(<100)`; the inner-slope (2–5 kpc) distributions vs the data's 1.45.
- [ ] **Step 4: Compare figure.** Median (model−data)/data vs radius per epoch, three products, halo-mass terciles, raw and amplitude-pinned — the exp61 `qa_bins` layout, so the discretisation share of the central deficit is read off directly.
- [ ] **Step 5: README + todo.** Result 0 stated three ways (measured / meaning / reference / figure); the D3 decision frozen; the early-step and mid-gap mass now included quoted.
- [ ] **Step 6: Commit**: `exp63 Stage 0: Result 0 — the incumbent re-baselined on the exact integral`.

## Self-review

- Spec coverage: G0a–G0d (Task 2), Result 0 on both samples (Task 3), D1 analytic R200c (engine default `mode="analytic"`, catalog kept as a comparison column — Task 3 product 2), D3 measure-first (Task 2 + Task 3 rule). Stage 1 is not in this plan by design.
- Placeholder scan: thresholds are numbers; every product is named; the figure layouts are named after existing ones.
- Type consistency: `predict` returns `(n, 5, len(R))` everywhere; `discrete_sum` returns `(5, len(R))` per galaxy; `mode` strings `"analytic"|"catalog"` used identically in Tasks 1–3.
