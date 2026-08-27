# exp61 — problem tracker

Severity: **S1** could change a conclusion · **S2** blocks work or wastes time ·
**S3** cosmetic or bookkeeping.

---

## Open

### P4 — every tier 2e number quoted for hongshao v1 describes the MEAN alone
**S2. Open, filed as open question C16 (2026-08-27).**

`qa.evaluate` accepts `draw_cogs`, and the name reads as "score the
generative layer". It does not: the draws are measured once and passed to the
PLANE figure only (`hongshao/qa.py:563-565`). Every tier, including 2e (the
mass CDFs) and the tier-2 dex scatters, is computed from the mean
`model_cogs`. The tell was available and unnoticed until now — exp60 Stage
3d's adopted-layer tier 2e table is identical **digit for digit** to the bare
incumbent's, and an option that changes nothing measurable is not being
measured.

Why it matters here: exp61 Stage 4's finding is that the predicted population
is too NARROW, worst in the outer annuli. The adopted hongshao v1 is the mean
PLUS the Option A stochastic layer, and supplying that width is exactly what
the layer is for. So the tier 2e numbers in this experiment, in exp59 Stage 5
and in exp60 Stage 3d all describe a component of v1 rather than v1, and read
as a population failure the delivered model may not have.

**Not a defect in any exp61 conclusion**: the three models compared here are
all MEAN models, evaluated identically, which is the comparison that was
asked for. **Fix, when someone takes C16**: compute tier 2e (and the tier-2
scatters) from the drawn populations when a layer is present, keeping the
mean's value alongside; `evaluate` already measures the draws.

---

## Resolved

### P3 — the Stage 2 README overclaimed the per-epoch fix as population-level
**S1. Found and resolved 2026-08-27, by the Stage 3 battery.** *The claim had
already been written into the README and reported to the user.*

Stage 2 wrote, from B1 and B3: "Given z=2 alone, the SAME seven parameters put
both back to zero." B1 and B3 — like every fitted loss in this experiment —
are defined on the **mh-complete + sane fit mask**, which at z=2 keeps 839 of
2397 galaxies and keeps the massive-halo end. On the whole clean population
the per-epoch fits do not zero the z=2 bias, they **overshoot** it: +5.5 per
cent inside 10.25 kpc and +12.4 per cent inside 100 kpc, against +0.3 and
+0.6 on the mask. The shared law moves 4 points between the two samples on
that quantity; the per-epoch ceiling moves 12.

Caught only because Stage 3 was written to print every bias on BOTH samples.
Located by the halo-mass-binned figure: at z=2 the ceiling runs +21 to +25 per
cent in the lowest halo-mass tercile and −5 to +4 per cent in the highest, and
the lowest tercile is what the mask removes.

**Resolution**: the Stage 2 paragraph now carries the qualification inline;
Stage 3's README section states the two-sample table as its headline. Neither
of exp61's two answers changes — what changes is what may be claimed from the
15.9 per cent recoverable at z=2 (real on the fit sample, not a
population-level fix), which promotes open question C13 from optional to
load-bearing. **Lesson**: report every population number on both samples
whenever the fit mask and the QA population differ; this is the programme's
third instance of that split (exp58 P-C, here twice).

### P2 — `matplotlib.patheffects` is unusable under this machine's `usetex`
**S3. Found and resolved 2026-08-27.**

Adding a `withStroke` path effect to the transfer matrix's cell labels — to
guarantee contrast over cividis's mid-luminance greys — raised
`RendererBase._draw_text_as_path() missing 1 required positional argument:
'mtext'` at savefig time, not at artist-creation time, so the script built
fine and died on the first export. **Resolution**: choose the text colour from
the cell's own luminance instead (`im.cmap(im.norm(v))` →
`0.299R + 0.587G + 0.114B`, white below 0.45). Thresholding on the VALUE
rather than the luminance is what put white text on the worst-contrast greys
in the first draft.

### P1 — three separate usetex string bugs in one figure set
**S3. Found and resolved 2026-08-27.** *One of them presented as a layout bug.*

(a) A bare `_` in a text-mode label (`score_A`) is a LaTeX error. (b) A bare
`-0.0461` renders as a hyphen, not a minus, so a figure's quoted numbers look
inconsistent with the table beside them. (c) Slicing a math string to reuse
part of it (`ZLE[1:-1]`) strips the `$` delimiters and puts `\leq` into text
mode, which raises "Missing $ inserted" from inside `tight_layout` — the
symptom appears in the layout engine and the cause is in the string, exactly
as exp57's double-`$` bug did.

**Resolution**: `$z \leq 1.0$` is defined once as a module constant and
concatenated, never sliced; signed numbers are formatted as `f"${v:+.4f}$"`;
symbols with subscripts are written in math mode.
