# exp60 — problem tracker

Severity: **S1** could change a conclusion · **S2** blocks work or wastes
time · **S3** cosmetic or bookkeeping.

## Open

*(none)*

## Resolved

### P3 — the first "S1-first" variant was mis-calibrated and understated the variant
**S2. Found and resolved 2026-08-27, Stage 3c.** The first run used pool
shift 0 for the S1-first variant, which neither hit the declining fraction
(34.0% against 41.8%) nor preserved the mean — the worst of both. The
tension scan's own table shows fraction AND median improve together with a
positive shift, so the variant's shift is now CALIBRATED on the calibration
half against the frozen fraction target (bisection, amplitude noise active,
noise width re-calibrated at the final shift). Calibrated (+0.514), the
variant hits 42.3% held out. **Lesson**: a variant named for a gate must be
calibrated to that gate before it is judged, or the judgment is of the
naming, not the design.

### P2 — a hard monotonicity tolerance excluded a third of the sample
**S2. Found and resolved 2026-08-27, Stage 1.** The response curves are
differences of two decreasing curves, so per-galaxy wiggles of a few
thousandths of a dex are expected; a 1e-9 tolerance excluded 835-1246
galaxies. Replaced by a declared 0.005 dex tolerance with the sample's worst
uphill step printed (0.037 dex), so the cut is auditable. 583 galaxies still
excluded from the ANATOMY (not from the layer).

### P1 — the gate-optimal core radius has no leverage on the target aperture
**S1. Found and resolved 2026-08-27, Stage 1.** *Would have invalidated the
core component silently.* The first run used the shared-mechanism's
gate-optimal Rc = 1.98 kpc and could reproduce only 9.8% of the measured
central changes: a 2 kpc remap moves mass from inside 2 kpc to just outside
it, which stays INSIDE the 4.92 kpc aperture the decline targets are defined
on — near-flat response curves, spuriously flagged non-monotone, 55% of
galaxies "needing A > 20". Resolution: a declared coverage rule (smallest
candidate Rc with >= 90% coverage; none reached it, the largest — 20 kpc —
used with the 85.3% shortfall reported). The per-galaxy draw is free of the
shared-A trade that forced 2 kpc, and reach is re-gated on the drawn
population. **Lesson**: a mechanism's geometry must be sized to the aperture
its TARGET is measured on, not inherited from a different objective's
optimum.
