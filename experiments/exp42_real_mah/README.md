# exp42 — the adopted kernel on the real MAH (completing the record)

The adopted 1ch-mof kernel has only ever been fitted in the DiffMAH
configuration (the smooth reconstructed halo history, ~99 evenly-spaced
deposits). The raw-input head-to-head — the de-dipped real main-branch
peak history, which keeps merger bursts (`real_mah`: running-max, same
deposit-dict keys) — was run on the earlier model generations and never
repeated for the Moffat kernel:

- per-galaxy fits (exp29, Gaussian deposits): the smooth curve FLATTERS
  the model by ~2 points (joint multi-epoch max|rel| 4.4% -> 6.1% on the
  real MAH);
- population-shared fits (exp30 phase 4 / exp32 step 2, 5-7-param
  transport): DiffMAH is BETTER by 3-5 points held-out (LOGO 30.6% vs
  33.6%; 10-fold CV epoch-avg 30.4% vs 35.2%) — the smooth regular basis
  suits one shared theta, the bursty gappy real history demands
  per-galaxy adaptation;
- the real MAH's virtue was population DIVERSITY (centered plane energy
  3.5x floor, the era's best) at a badly biased amplitude — a job now
  done by the exp41 stochastic layer.

This experiment closes the gap: the SAME 12-parameter 1ch-mof structure,
the SAME official z<=1.5 fit scope and protocol (exp40 `latestart`), with
config = "real" instead of "diffmah". Model code is exp38
`stage2_multiepoch.py` unchanged; only the loader config differs.

## Judged comparison (the exp40 z15 records are the marks)

| metric | DiffMAH (official, exp40) | real MAH (this exp) |
|---|---|---|
| joint z<=1.5 loss | 0.1538 | ? |
| params [log_rc, g, q, mu, sig, gamma] | [2.74, 4.00g, 0.91, 1.52, 0.24, 1.38] | ? |
| held-out shape R>5 by epoch (z=2.0 = extrapolation) | 18.2/17.4/16.6/16.4 / 15.4* | ? |
| held-out M(<5) z=0.4 | -9.5% | ? |
| differential massive z0.7->0.4 (data 0.37/0.11) | 0.40/0.13 | ? |
| overshoot T1 [30-60 / 60-148 kpc] | +0.025/+0.028 | ? |
| bounds | g=4.00 (stress-verified benign) | ? |

Also measured: the TRANSFER cross — the official DiffMAH-fitted theta
evaluated on the real-MAH input without refitting — which separates "the
input changed" from "the fit adapted".

Expectation from the record: the real input loses a few points of
held-out accuracy at the population level; open questions are whether the
Moffat tail (which absorbs shape freedom the Gaussian lacked) shrinks
that gap, and whether the physics tests (fitted to nothing) move.

## Run

```bash
export HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof
PYTHONPATH=. uv run python experiments/exp42_real_mah/compare.py \
    {demo|fit|physics|cv} [--dev]
```

`fit` = joint z<=1.5 plain-loss fit on the real MAH (warm-started from
the official theta + a weaker-envelope nudge; parent-side penalties);
`physics` = differential + overshoot for {official theta on real input
(transfer), real-fitted theta}; `cv` = the exp40 10-fold CV protocol on
the real input (z=2.0 column = extrapolation). Long runs via
nohup + caffeinate -im + disown with a log tail.

## Results (2026-07-19, full n=2397; `outputs/run_full.log`)

**The earlier-generation verdict is REVERSED for the Moffat kernel: the
real MAH no longer loses held-out accuracy — it ties at z<=1.0 and wins
at z>=1.5 — but its refit strains the outskirt physics band.**

| metric | DiffMAH (official, exp40) | real MAH (this exp) |
|---|---|---|
| joint z<=1.5 loss | 0.1538 | 0.1571 (both starts converge here) |
| params [log_rc, g, q, mu, sig, gamma] | [2.74, 4.00g, 0.91, 1.52, 0.24, 1.38] | [1.56, 2.43, 0.79, 1.78, 0.52, 1.28] |
| bounds | g=4.00 (stress-benign) | **NONE** |
| held-out shape R>5 by epoch | 18.2/17.4/16.6/16.4 / 15.4* | 18.1/17.1/16.7/**15.5** / **13.6*** |
| held-out M(<5) by epoch | -9.5/-8.0/-7.8/-10.6 / -11.8* | -8.9/-7.6/-7.4/**-8.2** / **-9.2*** |
| differential massive z0.7->0.4 (data 0.37/0.11) | 0.40/0.13 | **0.37/0.11 — exact** |
| differential earlier pairs (data 0.36/0.27/0.23) | 0.31/0.29/0.21-0.23 | 0.26/0.21/0.16 — undershoots |
| overshoot T1 [30-60 / 60-148 kpc] | +0.028/+0.022 (in-sample) | +0.025/**+0.080** — 60-148 kpc OUT of band |

(* = z=2.0, extrapolated, not in any fit. Shape = held-out 148-pinned
max|rel| R>5 kpc, per-galaxy median, 10-fold CV; differential = median
fraction of massive-tercile inter-epoch growth landing beyond 50/100
kpc; overshoot = median model-data dlog Sigma at z=0.4 by stellar-mass
tercile.)

1. **The accuracy gap is gone.** The Gaussian-era 3-5-point real-MAH
   penalty (exp30/32) does not survive the power-law-tail deposit: raw
   loss +2% (0.1571 vs 0.1538), held-out shape equal at z<=1.0 and
   better at z=1.5 (-0.9) and at the z=2.0 extrapolation (-1.8 points,
   with M(<5) -9.2 vs -11.8). The Moffat tail absorbs the shape freedom
   the bursty basis demanded, as hypothesized.
2. **A different, bound-free basin.** The real input prefers a much
   smaller, shallower-growing birth radius (log_rc 1.56, g 2.43), an
   efficiency window peaking earlier (z=4.9 vs 3.6) and twice as broad
   (sig 0.52 vs 0.24), and a slightly softer tail (gamma 1.28). The
   g=4.0 rail vanishes — it was a property of the smooth basis, not of
   the model.
3. **The transfer cross isolates the input effect**: the official theta
   dropped onto the real input unrefit loses 3.6 shape points at z=0.4
   (21.8 vs 18.2) and mildly strains the differential (0.42/0.13); the
   refit recovers everything — the two inputs genuinely demand
   different transport configurations.
4. **Physics: the flagship pair is an exact match, the rest strains.**
   The refit reproduces the z0.7->0.4 differential exactly (0.37/0.11,
   the program's best-ever pass) but undershoots every earlier pair
   (0.26/0.21/0.16 vs data 0.36/0.27/0.23) and quadruples the low-mass
   far-outskirt overshoot (60-148 kpc T1 +0.080 vs the adopted band
   +0.02-0.03) with a massive-tercile undershoot appearing (-0.036).
   Reading: the broad early window front-loads the deposit budget, so
   late inter-epoch growth is carried by migration alone — right for
   the latest pair, too weak earlier — and the bursty late deposits
   that do arrive land too wide at low mass.

**Verdict (recommendation; adoption unchanged pending user review): the
DiffMAH configuration remains the operating point** — it stays inside
the physics band everywhere, carries the differentiability, and the
exp41 stochastic layer is calibrated on it — but the record now shows
the real-MAH input is an accuracy PEER of the smooth input under the
heavy-tailed deposit, with a strictly interior optimum. The tech-note-2
caveat ("unverified extrapolation") is closed in both directions:
accuracy verified equal-or-better, physics band verified strained.
