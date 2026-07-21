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

## Results (2026-07-19, full n=2397; `outputs/run_full.log`,
## figure `figures/exp43_ladder`)

**Headline: two epochs is the minimum viable fit scope, and for
1ch-mof the PHYSICS extrapolates before the shape does — a z<=0.7 fit
predicts the unfitted high-z differential-deposition pairs essentially
exactly. 2ch-exp posts equal-or-better extrapolated SHAPE numbers but
its high-z differential overshoots at every scope: the two-channel
freedom extrapolates the metric, not the physics.**

Extrapolated pinned shape / M(<5) at the two hardest epochs
(z=1.5 / z=2.0; values marked * were never in the fit):

| fit scope | 1ch-mof z=1.5 | 1ch-mof z=2.0 | 2ch-exp z=1.5 | 2ch-exp z=2.0 |
|---|---|---|---|---|
| z04 | 41.2* / -43.6* | 44.2* / -46.5* | 15.6* / +8.0* | 12.7* / +6.1* |
| z07 | 20.8* / -19.7* | 21.1* / -21.1* | 19.2* / -14.6* | 19.1* / -15.5* |
| z10 | 18.0* / -13.8* | 17.2* / -14.8* | 18.0* / -12.8* | 18.0* / -14.1* |
| z15 | 16.3 / -10.7 (fitted) | 15.4* / -12.0* | 16.4 / -9.8 (fitted) | 15.8* / -10.3* |
| z20 | 17.2 / -10.1 | 14.1 / -8.0 (fitted) | 15.8 / -8.7 | 14.6 / -8.6 (fitted) |

1. **One epoch is not a fit scope — for either model, but they fail
   in OPPOSITE, instructive ways.** 1ch-mof at z04 collapses honestly:
   the migration envelope is unidentifiable (q -> 0), so extrapolation
   up in z degrades monotonically to 44% shape / -47% inner mass, and
   the differential trend comes out INVERTED (0.21 -> 0.28 rising with
   z where the data fall 0.37 -> 0.23). 2ch-exp at z04 instead posts
   the best z=2.0 extrapolated shape in the whole table (12.7%*) —
   while its inner masses SIGN-FLIP (+3 to +8%) and the differential
   nearest its own fitted epoch breaks badly (0.54/0.20 vs data
   0.37/0.11): the split + q=1.41 degeneracy mimics the shape trend
   for the wrong reasons. A flexible model can extrapolate a metric
   while being wrong; judge extrapolation on the physics tests too.
2. **The second epoch is the transformative one.** It makes the
   transport clock identifiable (1ch q: 0.00 -> 0.66; 2ch q: 1.41 ->
   0.74; both then converge monotonically, 0.79/0.91 and 0.74/0.83 up
   the ladder), collapses the 1ch extrapolation error from 44 to 21
   points at z=2, and — most striking — **the 1ch z07 fit predicts
   the unfitted differential pairs at data precision: z1.5->z1.0*
   0.27/0.08 (data 0.27/0.07) and z2.0->z1.5* 0.23/0.06 (data
   0.23/0.06, EXACT)** — better than the z20 fit trained on those
   epochs (0.18/0.05). The exp40 finding (late-anchored transport
   extrapolates up better than high-z-anchored transport fits down)
   extends all the way to a two-epoch anchor.
3. **The third epoch buys most of the remaining shape.** z10 -> z=2.0*
   at 17.2% (1ch), within 1.8 points of the five-epoch fit's own
   14.1%; scope z15 adds ~1 more point. Diminishing returns beyond
   three epochs.
4. **2ch-exp's high-z physics does not extrapolate at ANY scope**: the
   z2.0->z1.5 differential sits at 0.31-0.33 (data 0.23/0.06) for
   every scope >= z07, fitted or not — the wide exponential channel
   keeps feeding outskirt light at high z. 1ch-mof's late-anchored
   fits hit 0.21-0.23 on the same pair.
5. **Parameter stability along the 1ch ladder is excellent from z07
   up** (log_rc 2.73/2.69/2.74, window peak/width nearly frozen);
   only q walks (0.66 -> 0.79 -> 0.91) — consistent with the exp40
   reading that the late epochs mostly refine the migration envelope.

**Demonstration figures** (`ladder.py figures`):
`figures/exp43_differential` — the measured differential curve vs every
(variant, scope), open markers = the pair was not in the fit (shows the
z04 inversion, the z07 exactness, and the 2ch high-z overshoot);
`figures/exp43_residual_profiles` — 148-pinned median residual vs
radius at z=1.5/2.0 per scope (shows the error is inner-radius
concentrated, the 1ch z04 collapse, and the 2ch z04 sign-flipped core);
`figures/qa_*_exp43_1ch-mof_z07.*` — the STANDARD QA set for the star
rung (the two-epoch fit evaluated at all five epochs, three of them
extrapolation). Plus the ladder overview `figures/exp43_ladder`.

**Reading for the forward-modeling use case (the user's motivation):
a 1ch-mof kernel anchored on two to three low-z epochs is a genuinely
capable upward extrapolator — the transport physics locks in at two
epochs and the shape at three — while 2ch-exp should not be trusted
outside its fitted range despite (because of) its better-looking
extrapolated shape numbers.** In-sample caveat: fitted-epoch numbers
here are in-sample (~0-0.3 points optimistic vs the recorded CVs);
extrapolated epochs are out-of-sample in the epoch direction by
construction. CV of the z07/z10 rungs is the natural follow-up if a
scope decision is to be made on them.
