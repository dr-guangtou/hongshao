# exp59 — problem tracker

Severity: **S1** could change a conclusion · **S2** blocks work or wastes time ·
**S3** cosmetic or bookkeeping.

---

## Open

*(none)*

---

## Resolved

### P2 — the pre-registered sign for `e_f` was derived from the wrong variable
**S1. Found and resolved 2026-08-26, before any fit.** *Caught by the module's
direction self-check, then settled by measurement.*

The plan pre-registered `e_f < 0` from exp58 P-B's correlation of the z=2
amplitude residual with the EPOCH-LEVEL formation time (the galaxy's
`t_half/t` at z=2). The law's variable is the PER-DEPOSIT `f_form_j`, which is
a different quantity: it sits near 1 while the halo is actively assembling and
decays afterwards, so a galaxy's deposit-weighted mean of it is dominated by
its assembly bursts, not by its state at the observation epoch. The
self-check's first draft asserted the epoch-level intuition and failed;
measurement (the leverage sweep) then showed the group contrast has the same
sign in both time windows — decliners sit LOWER in deposit-level `f_form`
before z=2 (+0.231 against +0.262) and after (+0.126 against +0.186) — which
is what makes the crossing unreachable (see the README).

**Resolution**: the self-check now asserts the MECHANISM (each galaxy's mass
change tracks its deposit-weighted conditioning mean, r = +0.999) and leaves
every group-level claim to the sweep, which measures it. **Lesson**: a
pre-registered sign must be derived on the law's own variable, not on a
correlated summary of it; the two can differ, and here the difference is the
result.

### P1 — the leverage script's first draft transcribed the z=2 split with the wrong sign
**S1. Found and resolved 2026-08-26.** *Would have spent five fits on
disqualified candidates.*

`SPLIT_NOW["z=2"]` was transcribed as **+0.122** from exp58's magnitude table;
the SIGNED split (decliner minus rest, model over measured) is **−0.122**. The
bug flipped the required coefficient's sign, so the draft declared `F1` and
`C1` "compatible — worth a fit" when the correctly-signed reading shows that
closing z=2 doubles the z=0.4 split. Caught by comparing the sweep's `e = 0`
row (−0.1216, +0.0816) against exp58 P-D's group medians, which is why the
null row is printed and checked first. This is the project's fifth instance of
a bound/target mismatched to what it reports ([[bounds-must-match-what-they-report]]);
the new twist is that a MAGNITUDE is not a TARGET — signed quantities must be
transcribed signed.
