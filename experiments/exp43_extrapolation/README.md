# exp43 — the extrapolation ladder (user plan, 2026-07-19)

**Motivation (user):** in observations we rely mostly on low-redshift
data with few or incomplete high-redshift constraints, so in a real
forward-modeling run the kernel's ability to EXTRAPOLATE upward in
redshift is a first-class model-selection criterion — not just held-out
accuracy at fitted epochs. This experiment measures it directly, for
BOTH kernels (the adopted 1ch-mof and the 2ch-exp alternative): fit a
progressively shorter low-z epoch range, then evaluate everything the
fit never saw.

## The ladder

Scopes (joint plain-loss fits, the exp40 protocol; * marks fits reused
from the record rather than re-run):

| scope | fitted epochs | 1ch-mof theta | 2ch-exp theta |
|---|---|---|---|
| z04 | 0.4 | exp38 `stage3_single.npz[theta_1ch-mof_k0]`* | exp38 `stage3_single.npz[theta_2ch-exp_k0]`* |
| z07 | 0.4, 0.7 | THIS EXP | THIS EXP |
| z10 | 0.4–1.0 | exp40 `latestart.npz[theta_z10]`* | THIS EXP |
| z15 | 0.4–1.5 (official) | exp40 `latestart.npz[theta_z15]`* | THIS EXP |
| z20 | 0.4–2.0 | exp38 `stage2_multiepoch.npz[theta_1ch-mof]`* | exp38 `...[theta_2ch-exp]`* |

(The reused z04 fits came from a single warm start; the new fits use
two starts like exp40. If a z04 conclusion ends up load-bearing, refit
with two starts first.)

## Honesty of the readout

- **Extrapolated epochs never enter the loss**, so their metrics are
  out-of-sample *in the epoch direction* even on the full sample; the
  galaxies themselves are in-sample, but population-shared thetas
  showed a ~zero in-sample-vs-held-out gap (exp32: 30.5% vs 30.4% —
  capacity-limited, not overfitting), so full-sample fits are the
  honest instrument here. Fitted-epoch numbers ARE in-sample
  (optimistic by ~0-0.3 points vs the recorded CVs).
- **The per-epoch M(<500 kpc) normalization uses the measured total at
  every epoch, including extrapolated ones.** The ladder therefore
  tests the extrapolation of the *transport/shape*, at known totals;
  the pinned-shape metric is amplitude-independent anyway, and in a
  real forward-modeling run the high-z totals would come from the
  statistical emulator or an abundance constraint, not from this
  kernel.

## Judged readouts (fixed before fitting)

1. **The extrapolation curves**: pinned shape R>5 and M(<5)/M(<10)
   bias at every epoch for every (variant, scope), extrapolated
   epochs starred; the figure plots metric vs evaluation epoch, one
   line per fit scope, open markers = extrapolated.
2. **Physics extrapolation**: the differential-deposition pairs
   (massive tercile, data 0.37/0.11 // 0.36/0.10 // 0.27/0.07 //
   0.23/0.06 from z0.7->0.4 up) predicted by fits that never saw the
   pair — the record's precedent: the z15 fit predicted the unfitted
   z2.0->1.5 pair BETTER (0.21/0.06) than the z2-anchored fit trained
   on it (0.18/0.05).
3. **Parameter stability along the ladder** — especially the
   migration knobs: the z04-only 1ch-mof fit chose q = 0.00 (a single
   epoch cannot see the migration envelope), so the question is how
   many epochs lock the transport in (does z07 already recover
   q ~ 0.9?).

Marks from the record (1ch-mof): z15 in-sample shape by epoch
18.2/17.4/16.5/16.3 | z2.0* 15.4; z10 held-out extrapolation degrades
hard (z1.5*/z2.0* shape 17.9/17.7, M(<5) -14.2/-15.5 — "too short a
lever arm"); z04-only own-epoch M(<5) -7.6%.

## Run

```bash
export HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof
PYTHONPATH=. uv run python experiments/exp43_extrapolation/ladder.py \
    {demo|fit|report} [--dev]
```

`fit` runs the four missing fits (1ch-z07, 2ch-z07, 2ch-z10, 2ch-z15;
resumable — already-saved scopes are skipped); `report` assembles the
full 2-variant x 5-scope ladder from new + reused thetas, prints the
bias tables and differential lines, and writes the extrapolation
figure. Long runs via nohup + caffeinate -im + disown.

## Results

(pending)
