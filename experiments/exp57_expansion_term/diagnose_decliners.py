"""exp57 — do the galaxies whose centres DECLINE differ in anything the model
already knows about their halo?

**Why this exists.** Gate G5(b) fails for every expansion law tried: a single
global expansion strength shifts every galaxy, including the 58% whose centres
never needed shifting. The only way a SHARED law could fix that — no per-object
tuning, Contract B intact — is if the two groups differ in a quantity the model
can condition on. This asks whether they do, BEFORE any parameter is added,
because a conditioning row with nothing to condition on is freedom, not physics.
This project has recorded that per-object freedom always looks reachable.

**THE ANSWER (2026-08-26).** They differ in formation time and NOT in mass:

| property, median | declining (1003) | not (1394) | in sd units | r with the flag |
|---|---|---|---|---|
| log10 halo mass at z=0.4 | 13.294 | 13.282 | **0.04** | **+0.025** |
| `t_half(t)/t` at z=0.4 | 0.548 | 0.617 | **-0.74** | **-0.317** |
| effective concentration | 6.365 | 6.031 | 0.21 | — |

**A halo-mass-conditioned expansion cannot separate these two groups** — the
mass distributions are indistinguishable. **A formation-time-conditioned one
could, partly**: galaxies whose centres decline assembled EARLIER, having half
their present mass in place at 0.548 of their current age against 0.617 for the
rest.

That is a real handle and it is a quantity the model already carries per
deposit (`halo.Halo.f_form`), so conditioning on it adds one shared parameter
and no stellar information. But the ceiling is set by that `r = -0.317`: it
accounts for about a tenth of the flag's variance, so a conditioning row can
move the two groups apart by a fraction of the gap, not close it.

It also connects to a result this program already has: compact high-redshift
progenitors formed early. The centres that fall are the ones that were built
first.
"""
import sys
from pathlib import Path
import numpy as np
HERE = Path("experiments/exp57_expansion_term").resolve()
ROOT = HERE.parents[1]
for p in (ROOT, ROOT/"experiments/exp38_deposit_rethink",
          ROOT/"experiments/exp54_unpinned_amplitude", HERE):
    sys.path.insert(0, str(p))
import stage0_cost as S0, stage4_target as S4, fit as F
import numpy as np

recs, data, mask, lmh, base, theta, slow = S0.build(False)
fell, ok = S4.decline_flag(data)
lm = lmh[:, 0]                     # log10 halo mass at z=0.4 (DiffMAH)
# formation time proxies the model already carries, per galaxy at z=0.4
fform = np.array([h.f_form[-1] for h in recs])
c_eff = np.array([h.c_eff[-1] for h in recs])
# halo mass growth since z=2
lm2 = lmh[:, 4] if lmh.ndim > 1 and lmh.shape[1] > 4 else np.full(len(recs), np.nan)

print(f"  {int((fell&ok).sum())} declining, {int(((~fell)&ok).sum())} not\n")
print(f"  {'property':<34}{'declining':>12}{'not':>12}{'difference':>13}"
      f"{'in sd units':>13}")
for name, v in (("log10 halo mass at z=0.4", lm),
                ("t_half(t)/t at z=0.4 (formation)", fform),
                ("effective concentration", c_eff)):
    a, b = v[fell & ok], v[(~fell) & ok]
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    sd = np.nanstd(np.concatenate([a, b]))
    d = np.median(a) - np.median(b)
    print(f"  {name:<34}{np.median(a):>12.3f}{np.median(b):>12.3f}"
          f"{d:>13.3f}{d / sd if sd > 0 else np.nan:>13.2f}")

# and the quantity that actually matters: the model's own central error
pred = slow.predict(theta)
import stage37_size_epoch as S37
err = S37.central_error(pred, data, mask, shell=0)
print(f"\n  For reference, the model's z=2 central error in each group:")
print(f"    declining {np.nanmedian(err[fell & ok, 4]):+.3f} dex, "
      f"not {np.nanmedian(err[(~fell) & ok, 4]):+.3f} dex")
# how well could halo mass alone separate them?
from numpy.polynomial import polynomial as P
good = ok & np.isfinite(lm)
r = np.corrcoef(lm[good], fell[good].astype(float))[0, 1]
print(f"\n  correlation of the decline FLAG with halo mass: r = {r:+.3f}")
r2 = np.corrcoef(fform[good & np.isfinite(fform)],
                 fell[good & np.isfinite(fform)].astype(float))[0, 1]
print(f"  ... with the formation-time summary            : r = {r2:+.3f}")
print(f"\n  A shared conditioning row can only help in proportion to these.")
