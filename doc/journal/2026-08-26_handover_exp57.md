# Session Handover — 2026-08-26, exp57 (the expansion term)

**Branch `exp57-expansion-term`, not merged, not pushed.** Read
`experiments/exp57_expansion_term/README.md` first, then
`experiments/exp57_expansion_term/PROBLEMS.md`. Plan:
`doc/plans/2026-08-26-exp57-expansion-term.md`.

---

## The one-line result

**An expansion term is the first thing in this programme to move exp54's central
defect — and its RADIAL FORM decides everything.** Three homologous versions
improve the fit and fail the reach gate worse than the model this experiment was
designed not to repeat. A fourth, non-homologous, passes it decisively while
achieving the best profile-shape error of any model tried. **Nothing is
adopted**: it still fails one target gate, and the loss prefers a different
basin that the gates reject.

## The scoreboard, ordered by the gates and never by the loss

| model | loss | shape error | G2 >50 kpc | G2 >148 | G5(a) | G5(b) | G5(c) | verdict |
|---|---|---|---|---|---|---|---|---|
| incumbent | 1.874253 | 13.787% | — | — | 0.270 | 0.045 | +0.0455 | reference |
| **`X3` SMALL core, Rc = 2.61 kpc** | 1.833344 | **13.369%** | **2.2%** | **0.2%** | 0.210 | **0.039** | +0.0668 | **G2 PASS**, G5 ab**C** |
| `X0` old power law | 1.840977 | 13.478% | 72.5% | 42.1% | 0.151 | 0.079 | +0.0121 | G2 FAIL |
| `X1` + `c` frozen (control) | 1.851403 | 13.582% | 81.3% | 44.9% | 0.190 | 0.052 | −0.0196 | G2 FAIL, G5 all pass |
| `X1` bounded homologous | 1.840466 | 13.454% | 85.0% | 52.1% | 0.169 | 0.076 | −0.0019 | G2 FAIL |
| `X2` + time shape | 1.839900 | 13.463% | 76.1% | 45.0% | 0.156 | 0.079 | +0.0037 | G2 FAIL |
| `X3` large core, Rc = 92 kpc | **1.831172** | 13.436% | 53.4% | 8.2% | 0.158 | 0.086 | −0.0049 | G2 FAIL |

Limits: G2 ≤ 25% and ≤ 10%; G5(b) ≤ 0.060. Lower case = that part of G5 passes.

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
| `X2` + time shape | 1.839900 | −1.83% | 13.463% | — | `A = +1.29, p = +1.90` |
| `X0` old power law | 1.840977 | −1.78% | 13.478% | 1.0560 | `alpha = +0.3811` |
| **`X3` core-only, small core** | **1.833344** | **−2.18%** | **13.369%** | 1.0579 | `A = +2.71, Rc = 2.61 kpc` |
| `X3` core-only, large core | **1.831172** | −2.30% | 13.436% | 1.0535 | `A = +1.68, Rc = 92 kpc` |

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

### What did NOT fail — these are `X1`'s numbers; see the scoreboard at the top for all six

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

**It passes G2 decisively**: displaced mass beyond 50 kpc falls from `X1`'s
85.0% to **2.2%**, and beyond 148 kpc from 52.1% to **0.2%**. Its expansion
share of outskirt growth falls to **0.0%** and the outskirt structure
*improves* (population-summed `M[50,148]` from −0.076 to −0.007 dex). It does
this by moving **1.4%** of the stellar mass.

**But the search found TWO basins** — `Rc = 2.61 kpc` (loss 1.833344, passes
G2) and a near-homologous `Rc = 92 kpc` (loss **1.831172**, fails G2). **The
loss prefers the one the gates reject.** Both were reproduced independently and
both are gated; see the scoreboard.

**And `X3` small-core fails G5(c)** where every long-reach model passes it. The
attribution test says why: its expansion moves the core-density gap −0.025
while re-optimising the other seven moves it +0.046, and the compensation wins.

---

## What is running: `X4`, a bonus stage

`stage3_fit.py --only X4`, 10 starts, ~5 h from 05:56 on 2026-08-26. **exp57 is
complete without it** — everything above is fitted, gated, documented and
committed. `X4` is `X3` with the expansion strength conditioned on the halo's
formation time,

    G = 1 + [A0 + A_f (f_form − 0.6)] (1 − t_j/t_k)

which is the one extension the pre-registered diagnostic points at (see below).
`A0 = A_f = 0` nests exactly, both bounds are two-sided, and the starts place
`A_f` at **both signs** so the diagnostic's predicted negative slope is
falsifiable by the optimiser rather than assumed. When it lands: gate it with
`stage2_gates.py --fit gompertz_log-E2-S2+X4` and `stage4_target.py --fit ...`,
then re-run `stage7_summary.py` and `stage5_figures.py`.

**What it is aimed at**: `X3` small-core passes G2 and G5(a) and G5(b) and fails
G5(c). Giving early-forming haloes — the ones whose centres decline — more
expansion than the rest should move the population's median core density (c)
without disturbing the galaxies that never needed it (b). The ceiling is the
`r = −0.317` below.

## The 9-start `X3` confirmation

It completed and reproduces both basins: starts 0, 4,
7 → small core (1.833344); starts 1, 2, 5, 6, 8 → large core (1.831172); start 3
rails at the `log_Rc` bound with a worse loss. Start-to-start spread **2.2e-2**
against 1e-11 for the single-basin laws — **a single start would have reported
whichever basin it fell into.**

## What is owed

**The decision is the user's**: `X3` small-core is the only candidate that
   passes the mechanism gates and it has the best profile-shape error of every
   model tried, but it fails G5(c) and the loss prefers the gate-failing basin.

## Stage 8 — the trade, mapped, and a correction to the first framing

Sweeping the core radius with everything else held fixed, at **both** fitted
anchors (a conditional scan, labelled as one in its own output):

| `Rc` | G2 reach | G5(c) core gap | G5(b) non-decliners |
|---|---|---|---|
| ≲ 3 kpc | **pass** (≈2%) | **fails** | **pass** (0.024–0.038) |
| 4–40 kpc | **pass** (3–22%) | **pass** | **fails** (0.078–0.094) |
| ≳ 60 kpc | **fails** (41–99%) | pass | fails |

**The reach gate is NOT the binding constraint once the mechanism is
non-homologous** — it holds for every `Rc` up to about 40 kpc. The conflict is
between the two halves of G5: the core radius that fixes the median core
density is larger than the one that protects the 58% of galaxies which never
needed correcting. My first framing — "reach and the core-density target are in
direct tension" — was too coarse and is corrected here.

## The one extension the evidence points at

`diagnose_decliners.py`, run **before** anything was fitted so the choice could
not be made after seeing which helped: the galaxies whose centres decline differ
from the rest **in formation time and not in mass** — median `t_half/t` 0.548
against 0.617, `r = −0.317` with the flag, against `r = +0.025` for halo mass.
**The centres that fall are the ones that were built early.** A
formation-time-conditioned expansion strength is therefore the one shared
extension (one parameter, no stellar information, `f_form` already carried per
deposit) with something real to condition on, and its ceiling is that
`r = −0.317`. Open questions **C11** and **C12**.

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
