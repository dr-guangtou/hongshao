# exp57 — problem tracker

Every problem found during exp57, whether fixed, open, or decided against.
**Nothing is deleted**; items are marked resolved with the date and the answer,
so a problem that comes back is recognisable as a repeat. Newest first within
each status.

Severity: **S1** could change a conclusion · **S2** blocks work or wastes time ·
**S3** cosmetic or bookkeeping.

---

## Open

*(none yet)*

---

## Resolved

### P1 — the nesting gate was specified as "bit-for-bit" and that was unachievable
**S2. Found and resolved 2026-08-26, Stage 0.**

`stage0_cost.py` was written to demand that `ExpandingProblem` with the law
switched off reproduce exp54's `StackedProblem` **bit-for-bit**. It refused to
run: the profiles disagreed by **1.7e-15 relative, about 8 units in the last
place**.

**Cause, and it is not a bug.** exp54 forms one matrix product over all five
epochs at once. Expansion makes the deposit basis depend on the viewing epoch,
which forces one product per epoch, so the same sum is accumulated in a
different order. Floating-point addition is not associative.

**Resolution.** The *specification* was wrong, not the code, and it was
corrected in the open rather than the tolerance quietly widened — which is the
failure mode this project has recorded before. The gate now requires:

| | requirement | measured |
|---|---|---|
| NaN pattern | **exact** (a different set of failing galaxies is a real difference) | exact |
| profiles | < 1e-12 relative | 1.74e-15 (7.8 ulp) |
| loss | < 1e-9 absolute | 2.2e-16 |

The tolerances sit three and six orders of magnitude above what is measured, so
a real refactor error moves them by far more. This is the same standard exp54's
own `check_stacking` adopted when it stacked the per-galaxy loop, and for the
same reason.

**Lesson**: when a gate fails, decide whether the code or the specification is
wrong *before* touching either. Writing down "bit-for-bit" felt like rigour and
was simply not a property the refactor could have.

### P2 — the anchor-epoch cosmic times could not be read the way the first draft read them
**S3. Found and resolved 2026-08-26, Stage 0.**

`expand.ExpandingProblem` needs the cosmic time of each of the five anchor
epochs, because the expansion factor depends on how long ago a deposit was laid
down. The first draft looked for a `snap` field on the `Halo` record; there is
none (`halo.Halo` carries `t`, `z`, `epoch_mask`, and no snapshot index).

**Resolution.** Read it from `epoch_mask` instead — `epoch_mask[k]` is
`snap <= ANCHOR_SNAP[k]`, so its last True entry sits exactly at that anchor —
and then **check two things rather than assume them**: that the redshift there
is the anchor's redshift, and that every galaxy in the sample gives the same
cosmic time, since the whole tabulation rests on the snapshot grid being shared.

**A real finding fell out of the check.** `halo.ANCHOR_Z = (0.4, 0.7, 1.0, 1.5,
2.0)` are **rounded labels**; the snapshots actually sit at **0.39993, 0.70011,
0.99730, 1.49551, 2.00203**. The first tolerance (2e-3) was tighter than the
rounding and the check fired. The expansion factor uses the snapshot's own
cosmic times, never the labels, and the tolerance is now the label's rounding
with that stated in a comment.
