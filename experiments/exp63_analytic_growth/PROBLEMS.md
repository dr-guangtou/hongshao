# exp63 — problems, caveats and self-corrections

Numbered in the order they were found. Open items are marked.

## P1 — the probe's radius grid was wrong for a few hours (closed)

The tech-note probe (`doc/tech_note/scripts/04_analytic_check.py`) first read the
measured curves of growth on `geomspace(2, 148, 24)` instead of the measured
grid `(2^0.25 + 0.1 k)^4`. Same end points, same count, so nothing failed; but
grid point 4 is 4.2 kpc on one and 6.4 kpc on the other, and the "data rise as
R^1.45 over 2–5 kpc" headline was that mismatch. Caught when Stage 0's script
(which uses `fit.R_GRID`) printed a different data median. Corrected everywhere
the same day (tech note 04, the science plan, memory, lessons); the true numbers
are 1.00 (data, 2–4.9 kpc) against 0.88 (incumbent). The integral, the closed
forms and the discretisation numbers were never affected.

## P2 — tech note 04 overstated the numerical share of the central deficit (closed)

Its first version said "about half of the incumbent's central deficit is
numerics". Stage 0's Result 0 measured that the exact integral changes the
incumbent's *level* (M(<10 kpc) at z=0.4 from −2.4 to +3.6 per cent) but leaves
the amplitude-pinned *shape* defect (dip at 3–5 kpc, excess at 10–30 kpc)
unchanged: the fit had absorbed the level into its parameters. The note now says
so; the README's Result 0 section carries the correction.

## P3 — the engine gates' first thresholds were below the inputs' precision (closed)

G0a was first set at 1e-6 dex; the record stores the DiffMAH curve as float32
(2.7e-6 dex), so the bridge cannot beat that. Re-set to 1e-5 with the reason
stated in the code. Likewise the node-convergence check was first set at 1e-6
dex, but the reused `cog_truncated` kernel table (linear interpolation on 1024
points) has a 5e-6 dex floor that moves with node placement; re-set to 1e-5 with
the reason stated. Neither is a widening of a failing scientific threshold: both
are the precision of what is being reproduced.

## P4 — the incumbent drops stars where the catalog has no R200c (open, inherited)

`exp54/model.forward` silently drops any deposit at a snapshot without a catalog
R200c: 13.5 per cent of galaxies have such a gap mid-history (up to 8.2 per cent
of a galaxy's deposited stellar mass), and every galaxy loses its first one to
five steps. Not a defect of exp63 (the engine has neither problem) but every
number hongshao v1 has ever reported carries it. Filed in `doc/todo.md`.

## P5 — Stage 1's time labels are the incumbent's (open, by design)

The size-versus-time law of Stage 1b uses the incumbent's efficiency law to place
each part of the deposit-size distribution in time. A law that deposits more
stars late would move the early labels to smaller sizes. Stage 2's earlier-epoch
predictions are the test; until then the early end of the required s*/R200c
(0.04–0.07) is conditional.

## P6 — the deconvolution cannot see below 2 kpc (open, by construction)

Deposits smaller than the first measured radius are inside every data point, so
their size is unconstrained from the curve alone; Stage 1's "compact mode at
4–6 kpc" is a lower edge. Multi-epoch data or a smaller first aperture would be
needed to resolve it.

## P7 — the sweep rule was amended after the smoke run (closed, documented)

The Stage 3 plan's leverage rule named only the `M(30-50)|M(<30)` plane; the
smoke sweep showed the extended-size source acts on the outer plane
`M(50-100)|M(<30)` (which is where the mean is narrowest, exp61 Stage 4). The rule
now counts leverage on either kpc plane. Changed before the full run, stated here.

## P8 — the process cannot decline (open, the class limit)

Every noise source is drawn once per galaxy and applied to the history, so each
draw is a deposition-only galaxy and is monotone in time at every radius. The
declining-centre fraction of any drawn population is exactly zero; the truth's
is 43 per cent. This is the class limit stated by a prediction, as the science
plan intended; it is not a defect of the layer. The exp60 output layer reached
26.7 per cent (Option A) by adding a post-hoc core expansion to finished
profiles.

## P9 — the mean's split predicts the wrong sign of every assembly correlation (open)

Stage 2's compact share is a logistic in the halo mass at the deposit's time. At
fixed final halo mass that makes early-forming haloes extended and late-forming
ones compact; Stage 1 measured the reverse in the data (inner share vs `late`
+0.25, `f_form` −0.26, `logtc` −0.18; the model gives −0.34, +0.34, +0.29). A
physics-set accretion-to-deposition delay (Stage 2b) did not change the signs and
destroyed the predicted epochs. The split therefore needs a variable other than
the instantaneous halo mass — the growth *rate* at deposit (the DiffMAH slope,
which the forward model has) is the obvious candidate. **Leverage probe (by
evaluation, no fit), 2026-08-28**: with the compact share
$w_c=\sigma((m_{1/2}-\log M+g\,(\alpha-1))/d)$, $\alpha={\rm d}\ln M/{\rm d}\ln t$
at the deposit, the partial correlations of the $z=0.4$ compact share with
(`late`, `f_form`, `logtc`, `t50`) at fixed halo mass are

| g | median share | late | f_form | logtc | t50 |
|---|---|---|---|---|---|
| data (Stage 1 inner share) | — | +0.25 | −0.26 | −0.18 | −0.06 |
| 0 (Stage 2) | 0.56 | −0.34 | +0.34 | +0.29 | +0.76 |
| −0.5 | 0.43 | +0.22 | −0.29 | −0.06 | +0.26 |
| −1.0 | 0.30 | +0.35 | −0.45 | −0.21 | +0.06 |
| −2.0 | 0.15 | +0.49 | −0.70 | −0.48 | −0.18 |

$g\approx-0.5$ to $-1$ reproduces every sign and magnitude: deposits laid down
during fast halo growth (mergers) are extended, those during slow growth (smooth
accretion) compact — the in-situ / ex-situ picture in one parameter. Promoted to
**Stage 2c** (`stage2_fit.py --growth`, thirteen parameters, nests at $g=0$),
fitted at $z=0.4$ only like Stage 2; judged by the same gates and predictions.
