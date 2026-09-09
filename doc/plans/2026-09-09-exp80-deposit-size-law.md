# exp80 (proposed) — the deposit size law under the measured input, read from the data first

Status: EXECUTED 2026-09-09/10 (branch `exp80-deposit-size-law`; verdict in `experiments/exp80_deposit_size_law/README.md`). Was: PROPOSAL, awaiting the user (2026-09-09). exp79 is TAKEN by the
Codex agent (`hongshao_exp79_deposition_reach`: the deposit truncation
boundary C x R200c at deposition and the stellar-mass budget; incomplete,
no model adopted). exp80 is free as of 2026-09-09 06:00 (`git worktree
list`, `git branch -a`, `ls experiments`); check again before creating.
Relation to exp79: exp79 asks where a deposit is CUT OFF (the reach, C = 3);
this asks where a deposit is CENTRED (the size law s_c, s_e and their time
dependence). Both touch the outskirts; coordinate the boundary value with
exp79's result before Stage 1, and do not change C here.

## Why this, now

The programme's standing failure is the high-redshift SIZE OFFSET of the
mean model: R50 +0.037 / +0.054 dex at z = 1.5 / 2 (R20 +0.07 / +0.09), in
the top halo-mass tercile +0.08–0.12, with a mass–size slope at z = 2 of
0.33 against the truth's 0.11. exp78 established that this is not an
objective problem: a size term the loss can see (Stage 0 PASS) buys the
sizes at z ≤ 1 where the gate already passes and leaves z = 2 worse, because
the per-galaxy terms defend the high-z sizes. Every size-aware start also
railed the "compact" channel's size at 31.6 kpc with a steep time exponent
(early deposits 3–4 kpc, late ones 9–17 kpc). Both readings say the same
thing: THE SIZE LAW'S TIME DEPENDENCE IS THE WRONG SHAPE under the honest
input. The exp74 addendum already owes "re-tune the extended deposit's r200
scaling under the measured input" (the honest history's larger early halo
radius pushes the extended deposits outward: the 50–100 kpc shell at z = 2
+30 per cent per galaxy).

The model's size laws are s_c = 10^log_f_c (1+z')^b_c [kpc] and
s_e = 10^log_f_e (1+z')^b_e R200c(t') at the DEPOSIT time t' — one power of
(1+z) per channel, and the extended size tied to the halo radius at deposit.

## Stage 0 — no fit: what size does the data demand, epoch by epoch

exp63 Stage 1 read the deposit-size distribution W(s) from each z = 0.4
curve by non-negative least squares under the kernel (bimodal: ~5 kpc and
43–74 kpc). Run the same deconvolution at EVERY epoch on that epoch's
curves (the mh-complete progenitors, and the fitting sample), per galaxy,
under the incumbent kernel and the fitted Sersic compact kernel, and read:

  1. the extended mode's size and share vs halo mass at each epoch, and its
     ratio to R200c(z) of the halo AT THAT EPOCH and to the population's
     R200c at earlier times;
  2. the compact mode's size at each epoch (exp63 found ~5 kpc fixed at
     z = 0.4; does the late-time compact mode grow, as the railed bound says
     the model wants?);
  3. the MODEL's own mass-weighted deposit sizes at the same epochs
     (`model2._deposits2`: s_c, s_e per node, with the deposit masses), for
     the baseline, so the comparison is law against data in the same units;
  4. the mass–size slope of each mode at z = 2 vs the truth's 0.11.

Deliverable: one table per epoch (data vs model, compact and extended size,
share, by halo-mass tercile), and the candidate law change it implies. The
plausible candidates, to be chosen by the table and not before: (a) the
extended size tied to the halo's R200c at the OBSERVED epoch rather than at
deposit (deposits made early then expand with the halo — "no transport" is
kept for the mass, not the coordinate); (b) a halo-mass dependence of f_e;
(c) a second (1+z) exponent, or a break, in either channel; (d) the compact
channel re-parametrised as "early" (size in kpc, fixed) plus a late deposit
that scales with R200c. Parameter count of each stated in the table.

Gate for proceeding: the data's extended size at z = 2 must differ from the
model's by more than the tercile scatter of the deconvolution, and the
implied change must reduce the z = 2 R50 offset at frozen amplitude by more
than 0.02 dex when applied by hand (a frozen-theta test, exp74's lesson).

## Stage 1 — the fit, under the STANDARD objective

The chosen law change, fitted under exp63's A²+F²+S²+B² on the measured
input with the adopted references (no size term: exp78 showed it cannot buy
this; the radius term is scored AFTER the fit as a report), four starts
(baseline, 14.63 basin, exp63 official, nested) + continuation, each its own
process under nohup/caffeinate; then the standard battery, the size gate
(offset and width separately, both conditionings), the future-dependence
gate, the profile on both samples, the mass planes (the exp78 (base) lesson:
read the planes, not only the medians). Parameter count 12 or 13, stated.

Success is judged by the gates: R50 offset at z = 1.5 / 2 within 0.05 dex on
BOTH samples with the z ≤ 1 entries kept; the z = 2 mass–size slope moved
toward 0.11; the z = 2 centre not worse than −13 per cent; the leak gate
within 0.01 dex per dex of the baseline; the 14.63 basin not the winner.

## Not this (and why)

- Another objective term or weight sweep: exp78, six loss-vs-gates cases.
- The v1 stochastic layer's re-baseline: owed, but it is built on the mean;
  it cannot fix an offset and will inherit whichever mean this fixes.
- exp76 option 2 (g fixed by physics): a width lever; after the mean.
- The jumper snapshot check: ±4 per cent on the z = 2 amplitude, no gate.

## Cost

Stage 0 about an hour of compute (five deconvolutions of 2356 galaxies +
the model's deposit sizes); Stage 1 four fits in parallel, ~30 min, judge
~10 min.
