# exp76 — the growth-rate split under the honest input (PROPOSAL, awaiting the user)

**Branch** `exp76-growth-rate-split` from `master` `360d2d6`. **Status**:
proposed 2026-09-07; nothing run. **Prerequisite** the user's own next task —
adopt the measured history as the application input and re-baseline the
incumbent and the v1 stochastic layer on it. exp76 runs on the measured
input from the start (exp63's model refitted on it exists:
`exp74/outputs/stage1_refit_measured.npz`).

## Why this, out of everything the record leaves open

The last eight experiments (exp61–exp74) closed most of the levers the mean
model had: a richer time law (D2, exp73 E), a conditioned efficiency (exp59),
the concentration history, the objective and its weights (C20, established
twice), the error model (C17), the coordinate (C21), per-epoch freedom (it
does not transfer), and the halo history's pinning to the future (C19). What
they left is the same short list under every objective and every input:

1. **the centre** — −11 to −13 per cent at 2 kpc at z = 2, +6 at z = 0.4,
   the early-formed decliners (C8, exp58: "z = 2 and the central defect are
   one phenomenon");
2. **the width collapse** — every mean model's size scatter at fixed mass is
   0.2–0.6 of the truth's, falling with z (exp73 Block D; C16);
3. under the honest input, the z = 1.0–1.5 massive progenitors (+5–9 per
   cent) and the high-z sizes (+13 per cent at z = 2).

One open problem in exp63 connects the first two and has a lever that was
found by evaluation and never fitted. **P9**: the model's compact share is a
logistic in the halo mass at the deposit's time, which makes early-forming
haloes EXTENDED at fixed final mass; the data say the reverse (inner share vs
`late` +0.25, `f_form` −0.26, `logtc` −0.18; the model −0.34, +0.34, +0.29).
A no-fit leverage probe showed that a compact share depending on the halo's
growth RATE at the deposit, $w_c = \sigma((m_{1/2} - \log M + g(\alpha-1))/d)$
with $\alpha = d\ln M/d\ln t$, reproduces every sign and magnitude at
$g \approx -0.5$ to $-1$: **deposits laid down during fast growth (mergers)
are extended; those during slow growth are compact.** It was not fitted
because $\alpha$ was DiffMAH's slope — a smooth function of the final mass
(C19) — and the assembly variables it was scored against (`late`, `f_form`,
`logtc`) were DiffMAH-derived too (exp46's lesson: use the raw MAH and
mass-matched controls, not DiffMAH slopes).

Under the measured input both objections fall: $\alpha$ at every node is the
measured history's local slope, and the assembly variables can be measured
ones (formation time from the running peak, the late growth over the last
few snapshots). And the effect is the kind the record says is missing: a
per-galaxy split driven by the galaxy's own history is exactly what widens
the size distribution at fixed mass (the honest input alone took the z = 2
R50 width ratio 0.27 → 0.35–0.40 by giving each halo its own history), and a
compact share that rises for slow, early growth is what the early-formed
compact progenitors (exp46: a 0.3 dex head start by 1 Gyr) need at the
centre.

## What exp76 does

**Stage 0 — leverage by evaluation, no fit (an afternoon).**
- Re-measure P9's table on the measured input: the partial correlations at
  fixed halo mass of the model's compact share (exp63's model refitted on
  measured curves, `g = 0`) with MEASURED assembly variables (formation time
  $t_{50}$, $t_{80}$ from the running peak; late growth $\Delta\log M$ over
  the last 2 Gyr; the DiffMAH set alongside for continuity), and the same
  for the data's inner share (exp63 Stage 1's measurement, on the same
  galaxies).
- Sweep $g$ by evaluation on the measured curves with everything else held
  at the refit's values, as P9 did: do the signs cross at the same $g$, and
  what does the sweep do to the size-width ratio, the z = 2 centre, and the
  loss? A conditional slice — used to show the shape of the trade, never to
  locate an optimum (exp54 Stage 3.9).
- **Gate A**: a $g$ range exists where every measured assembly correlation
  has the data's sign. If none does, stop: the lever is not there under the
  honest input, and that is the result.

**Stage 1 — the fit (four starts as processes, ~1.5 h).** exp63's model with
$g$ free (one parameter; bounds from the sweep), on the measured curves,
exp63's objective, the standard starts plus the measured-input refit as a
start. The refit without $g$ is the null. Judge: the standard battery and
the size gate (offset AND width — a mean model that widens is the point),
the future-dependence gate, P9's correlation table, the z = 2 mh-complete
numbers, the centre at 2 kpc per epoch, and the 2 × 2 against the $g = 0$
refit.

**Decision rule.** $g$ earns its place if (a) the correlation signs match
the data's, (b) the width ratio rises at every epoch, and (c) the loss does
not worsen — or worsens by less than the width and the centre improve, in
which case the user decides (C20: the loss is a dial). If (a) holds and (b)
fails, the split's diversity is real but not size diversity, and the next
layer inherits the finding.

## What it is not, and the alternatives considered

- **Not the merger lag.** The catalog has sixteen snapshots about a
  gigayear apart at high z; ordinary growth between snapshots exceeds
  0.2 dex, so merger jumps cannot be separated from smooth growth and a lag
  cannot be tested with this data (checked 2026-09-07). Would need the full
  merger tree at every snapshot.
- **Not the expansion term.** exp57 showed a non-homologous expansion moves
  the centre, and that the loss ranks its basins backwards and the
  core-density and non-decliner gates conflict (C8, C11, C12). The user's
  reading — the decline is partly projection geometry plus feedback — needs
  3-D or multi-axis profiles to size the target first; blocked on data.
- **Not the layer yet.** The v2 layer needs an outer size component
  (exp73 Block D, second use) and should be built on whatever mean model
  exp76 leaves, after the re-baseline.
- **Not the early history below 1.18 Gyr.** Snapshots at z = 6–12 exist for
  74 → 0 per cent of galaxies; using them where present is heterogeneous by
  construction. Worth a coverage-controlled test later, since the compact
  head start sits before 1 Gyr, but not before the growth-rate split has
  been tried on the measured slope.

## Costs

Stage 0 ~2 h wall (evaluation only). Stage 1 ~1.5 h with four parallel
starts at 0.4 GB each, `caffeinate -i -w <pid>` on every process.
