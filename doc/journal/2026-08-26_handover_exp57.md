# Session Handover — 2026-08-26, exp57 (the expansion term)

**Branch `exp57-expansion-term`, not merged, not pushed.** Read
`experiments/exp57_expansion_term/README.md` first, then
`experiments/exp57_expansion_term/PROBLEMS.md`. Plan:
`doc/plans/2026-08-26-exp57-expansion-term.md`.

---

## The one-line result

**An expansion term is the first thing in this program to move exp54's central
defect — and the first three versions of it fail the reach gate, for a reason
that is structural rather than a bad parameter value.** A fourth version,
designed in response, is fitting.

---

## What was asked and why

exp54 established that its remaining defect is **outside its model class**: with
deposits non-negative and their profiles fixed once made, `M*(<R,t)` never
falls, while 42% of these galaxies lose a median 14% of their central stellar
mass. Four exp54 enrichments failed against a target the model class forbids.

The user's binding constraint on trying expansion again: **the program's first
model was deposition + transport and it failed**, and the failure must not
repeat. `exp52_growth_mechanism` diagnosed it as **the right source with ~10×
too much reach** — 93% of moved mass came from inside 5 kpc (correct) but 58.5%
was delivered beyond 50 kpc and 12% beyond 500 kpc where the normalisation
discarded it, and by z=0.7→0.4 **96.5%** of outskirt growth was redistribution.

The user also ruled out leaning on TNG's Stellar Assembly catalog: no radial
information, and its in-situ definition traces the main branch only, so z>3
major mergers count as ex-situ and land centrally.

---

## What is done

### Stage 0 — cost and nesting. COMPLETE.
Every law reproduces exp54's incumbent (loss **1.874253**) with an exact NaN
pattern, **1.74e-15** relative profiles (7.8 ulp) and a loss difference of
2.2e-16. Cost **4.3×** for the homologous laws and **5.0×** for `X3`, against
exp54's 112 ms — the technical note guessed "several times"; that number now
sets the schedule.

### Stage 1 — the contract, and the capability. COMPLETE.
exp54's monotone-in-time property, previously an analytic claim, is now measured
over **230,112 galaxy-radius-epoch pairs**: exactly zero fall. With expansion on,
declining centres are reachable — **but not as one number.** Matching the
measured overall median needs `A = 0.76`, the declining fraction `A = 0.98`, the
decliner median `A = 1.65`. The reason is measured: the model's 10th-to-90th
spread is **0.33× the data's**. Expansion shifts a distribution; it cannot widen
one. That is exp54 open question **C4** reappearing as the binding constraint on
a different mechanism.

### Stage 2 — the gates and their null. COMPLETE.
Four gates ported from exp52, thresholds fixed in the plan **before** any fit,
anchored on values exp52 measured on the failing model. The null (expansion off)
is measured first so every fitted number has a reference.

### Stage 3 — the fits.

| model | loss | vs incumbent | shape error | `score_A` | fitted |
|---|---|---|---|---|---|
| incumbent | 1.874253 | — | 13.787% | 1.0558 | — |
| `X1` bounded homologous | 1.840466 | −1.80% | 13.454% | 1.0570 | `A = +1.4657` |
| `X2` + time shape | 1.839900 | −1.83% | — | — | `A = +1.29, p = +1.90` |
| `X0` old power law | 1.840977 | −1.78% | 13.478% | 1.0560 | `alpha = +0.3811` |
| **`X3` core-only** | **1.833344** | **−2.18%** | — | — | `A = +2.71, Rc = 2.6 kpc` (start 0 of 9) |

**This is the largest gain any exp54 enrichment has produced** — Stage 3.5's
entire budget was 0.18 percentage points and a *free* shared time curve reached
only 13.62%. And the gain is in the **shape**, not the amplitude, so it is not
gate G3's amplitude-regulator trap.

### THE FAILURE, and it is the main finding

**G2 fails for every homologous law, worse than the model exp52 condemned.**

| | beyond 50 kpc | beyond 148 kpc |
|---|---|---|
| limit | 25% | 10% |
| exp52's failing model | 58.5% | 29.4% |
| `X1` (z=0.7→0.4) | **85.0%** | **52.1%** |
| `X0` (z=0.7→0.4) | 72.5% | 42.1% |

`A = 1.4657` means no deposit ever grows by more than 2.47× — the factor really
is bounded, as designed. But homologous scaling multiplies *every* radius by it
and the deposit has a **power-law tail**, so material at 50 kpc lands at 123 kpc.
**Bounding the factor does not bound the reach in kpc.** Both forms fail, so it
belongs to homologous expansion, not to a time law.

Confirmed independently by G1b, which has no threshold of mine in it: at
z=0.7→0.4 the null lands 0.364 of added mass beyond 50 kpc against a measured
**0.366**, and `X1` overshoots to **0.420**.

### What did NOT fail

- **G1 passes comfortably**: expansion supplies 12.9–18.5% of `M*[50,100]`
  growth against exp52's 76.9→96.5% ramp. **The first model's *dominance*
  failure does not recur.**
- **G5(a) passes**: the declining group's central-error span closes
  **0.270 → 0.169**, and the all-galaxies span **0.145 → 0.049**.
- **G5(c) passes essentially exactly**: the core-density gap against the
  measurement closes **+0.0455 → −0.0019 dex**.
- **G5(b) fails**: the non-declining 58% open **0.045 → 0.076** against a 0.060
  limit. One global strength shifts every galaxy, including those that never
  needed it — exactly Stage 1's spread argument.
- **G4 passes** its preregistered criterion (`c` moves −0.100, inside Stage
  3.9's −0.25/+0.51 width) **but the two are partially degenerate**: freezing
  `c` halves the fitted `A` and gives up 40% of the shape gain.
- **G6**: the outskirts do **not** degrade — `M[50,148]` improves from −0.076 to
  −0.055. Expansion does not pay Stage 3.7's price of emptying the outskirts.

### Stage 6 — `X3`, the fix

```
r(R) = R [ 1 − (1 − 1/G) / (1 + (R/Rc)^2) ]        G = 1 + A (1 − t_j/t_k)
```

read as *where the material now at `R` came from*. Core expands by `G`, nothing
beyond `Rc` moves. Mass-conserving by construction (a monotone rearrangement),
`A = 0` reproduces the incumbent to 2e-16. **It does not nest `X1`** — two
drafts claimed it did and the self-check falsified both (P5).

Start 0 of 9 gives **`A = +2.71`, `Rc = 2.61 kpc`, loss 1.833344** — the fit
strongly prefers *localized* core expansion over homologous, which is what the
physical picture says AGN-driven expansion does.

---

## What is running

`stage3_fit.py --only X3` (9 starts, ~25–35 min each, so ~4–5 h total) and
`--only X2` (7 starts). Logs in
`/Users/shuang/.claude/jobs/eb8b15e6/tmp/logs57/`.

## What is owed when they land

1. `stage2_gates.py --fit gompertz_log-E2-S2+X3` and `stage4_target.py --fit
   ...` — **the question is whether `X3` passes G2.** If it does, the
   experiment has a positive result; if it does not, homologous or not, the
   reach problem is deeper than the radial law.
2. Same for `X2`.
3. Rebuild `stage5_figures.py` with all four laws.
4. Update the README's verdict section, `doc/lessons.md`, `doc/todo.md`,
   `doc/open_questions.md`, and the technical note.

## If `X3` passes G2 but G5(b) still fails

A diagnostic already run (`diagnose_decliners.py`) says what a shared law could
condition on, and it was measured **before** fitting anything so the choice is
not made after seeing which helps:

| property, median | declining (1003) | not (1394) | r with the flag |
|---|---|---|---|
| log10 halo mass at z=0.4 | 13.294 | 13.282 | **+0.025** |
| `t_half(t)/t` at z=0.4 | 0.548 | 0.617 | **−0.317** |

**Halo mass cannot separate them; formation time partly can** — the centres
that fall are the ones built early. `f_form` is already carried per deposit, so
conditioning on it costs one shared parameter and no stellar information. The
ceiling is that `r = −0.317`, about a tenth of the flag's variance.

## Standing constraints for whoever continues

- **Gates first, loss last.** exp52's model reproduced the summed profile at all
  five epochs while getting the mechanism wrong.
- **Do not widen a threshold that fails.** The plan's stop conditions were fixed
  in advance.
- `export HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof`,
  `OMP_NUM_THREADS=1` (the evaluator is not BLAS-bound; threads only contend),
  keep the 27 GB cache at `~/Desktop/tng300_halo_structure`, import `fit` before
  `halo`/`model`, and `--smoke` must never write a production filename.
- **`exp56` is another agent's**, in the worktree
  `hongshao_exp56_compact_shape`. This is exp57.
