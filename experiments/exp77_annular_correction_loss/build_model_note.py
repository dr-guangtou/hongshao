# %%
"""Build the source-checked Exp77 explanatory note; no science fits or data edits.

Run from this worktree with:
uv run --no-project --with reportlab --with pymupdf python \
    experiments/exp77_annular_correction_loss/build_model_note.py
"""

import hashlib
import json
import subprocess
from pathlib import Path

import pymupdf
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
)

root = Path(__file__).resolve().parents[2]
destination = root / "output/pdf"
previews = root / "tmp/pdfs/exp77_model_note"
destination.mkdir(parents=True, exist_ok=True)
previews.mkdir(parents=True, exist_ok=True)
pdf_path = destination / "exp77_model_and_profile_correction.pdf"
styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        "body_note",
        fontName="Helvetica",
        fontSize=10.5,
        leading=15,
        spaceAfter=10,
        textColor=colors.HexColor("#253342"),
    )
)
styles.add(
    ParagraphStyle(
        "section_note",
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=27,
        spaceAfter=17,
        textColor=colors.HexColor("#153f58"),
    )
)
styles.add(
    ParagraphStyle(
        "subheading_note",
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        spaceBefore=9,
        spaceAfter=7,
        textColor=colors.HexColor("#153f58"),
    )
)
styles.add(
    ParagraphStyle(
        "equation_note",
        fontName="Courier",
        fontSize=9,
        leading=13,
        backColor=colors.HexColor("#edf3f6"),
        borderPadding=8,
        spaceBefore=6,
        spaceAfter=13,
        alignment=TA_LEFT,
    )
)
story = []


def paragraph(text):
    story.append(Paragraph(text, styles["body_note"]))


def heading(text):
    story.append(Paragraph(text, styles["subheading_note"]))


def equation(text):
    story.append(Preformatted(text, styles["equation_note"]))


def page(number, title):
    if number > 1:
        story.append(PageBreak())
    paragraph(f"HONGSHAO / EXPLANATORY NOTE / 09 SEPTEMBER 2026 / {number:02d}")
    story.append(Paragraph(title, styles["section_note"]))


page(1, "Exp77: what the model does")
paragraph(
    "<b>A halo-only physical prediction followed by a learned radial adjustment.</b> "
    "Exp77 retains the measured-MAH deposition baseline from Exp75. It changes "
    "how the subsequent correction is calibrated, explicitly including annular "
    "stellar masses alongside cumulative masses. This note describes the actual "
    "experiment, not an already integrated production model."
)
heading("Three models that should not be confused")
paragraph(
    "<b>Measured baseline:</b> the 12-parameter, two-channel deposition model "
    "integrates stellar contributions along a measured halo history to predict "
    "the CoG. Its population parameters are shared across galaxies and epochs."
)
paragraph(
    "<b>Measured hybrid:</b> that baseline plus four profile adjustments per "
    "epoch, predicted statistically from halo properties. Exp75 calibrated "
    "these adjustments against cumulative masses. Exp77 uses the same decoder "
    "but calibrates against cumulative and annular masses together."
)
paragraph(
    "<b>Measured direct:</b> a halo-to-five-profile-coordinate prediction "
    "without the deposition equation. It is an independent empirical comparator, "
    "not a hybrid with a different correction strength."
)
heading("How to read the QA")
paragraph(
    "The user's visual assessment favors the direct model in several profile "
    "diagnostics, while noting a narrow high-redshift mass plane. The measured "
    "hybrid has systematic excess outer mass. These are distinct failure modes: "
    "a good cumulative fit can conceal a poor difference between two apertures."
)
paragraph(
    "On the same 842 held-out discovery galaxies, averaged across five epochs, "
    "the mean radial log-CoG RMS is 0.11762 dex for the measured baseline, "
    "0.10815 dex for the direct model, and 0.10771 dex for the measured hybrid. "
    "The last two pooled scores are very close; they do not establish a universal "
    "ordering across radii, epochs, density slopes, or mass planes."
)
paragraph(
    "All these QA predictions are deterministic. A narrow mass plane can reflect "
    "missing conditional scatter, a biased mean relation, or both. Adding arbitrary "
    "noise is not a demonstrated remedy. Density here means a CoG-derived radial "
    "diagnostic, not an independently fitted isophotal surface-density measurement."
)

page(2, "The four-coordinate decoder")
paragraph(
    "Fix one galaxy and epoch. Let B[j] be its baseline cumulative projected "
    "stellar mass on increasing radii r[j], j=1,...,J. The current experiment "
    "uses 24 radii and five epochs: z=0.4, 0.7, 1.0, 1.5, 2.0. Masses are "
    "h-free solar masses. All totals below refer to the last grid radius "
    "R=148.22 kpc (rounded), not an extrapolated infinite-radius total."
)
equation("B[0] = 0;  s[j] = B[j] - B[j-1];  T = B[J]\na = (a0, a1, a2, a3)")
paragraph(
    "The s[j] are projected radial-bin masses, not three-dimensional shells. "
    "Define representative bin radii rho[1]=r[1]/2 and "
    "rho[j]=sqrt(r[j-1]*r[j]) for j&gt;1. Rescale ln(rho) to x in [-1,1]."
)
equation(
    "x[j] = 2*(ln(rho[j])-min(ln(rho)))/range(ln(rho)) - 1\n"
    "P1(x) = x\nP2(x) = (3*x*x - 1)/2\n"
    "P3(x) = (5*x*x*x - 3*x)/2\n"
    "u[j] = a1*P1(x[j]) + a2*P2(x[j]) + a3*P3(x[j])"
)
paragraph(
    "Multiply each bin mass by exp(u), then normalize the bin fractions. "
    "A separate base-10 amplitude adjustment sets the new total:"
)
equation(
    "p[j] = s[j]*exp(u[j]) / sum_k(s[k]*exp(u[k]))\n"
    "T_corrected = T * 10**a0\n"
    "C[j] = T_corrected * sum_{k<=j} p[k]"
)
paragraph(
    "Numerically, the implementation uses softmax(ln(s)+u), which evaluates "
    "the same normalized weights more stably. Zero baseline bin mass stays zero. "
    "For valid nonnegative baseline bins, corrected bins remain nonnegative and "
    "the CoG is nondecreasing with radius. All-zero a reproduces B exactly."
)
paragraph(
    "a0 is in dex; a1-a3 are dimensionless natural-log tilts. Positive a1 "
    "favors outer bins relative to inner bins. P2 and P3 allow curvature and "
    "asymmetry in the radial weighting; they are not separate physical components. "
    "Normalization makes shape adjustments conserve the chosen total. This does "
    "not guarantee a decreasing density slope or sensible evolution with time."
)

page(3, "How the corrections are learned")
heading("Step 1: obtain correction targets on calibration galaxies")
paragraph(
    "Let Y[j] be a calibration galaxy's measured CoG, and A[q](C) the annular "
    "mass computed by interpolating C linearly in radius and mass and subtracting "
    "adjacent aperture masses. The six annuli have edges 0, 2, 10, 30, 50, 100, "
    "and the exact last grid radius. Solve separately for each galaxy and epoch:"
)
equation(
    "a_star = argmin_a L(a)\n"
    "L = mean_j( log10(C[j](a)/Y[j])**2 )\n"
    "    + w * mean_q( (h(A[q](C)) - h(A[q](Y)))**2 )\n"
    "h(m) = asinh(m/(0.001*Y[J])) / ln(10)"
)
paragraph(
    "The transform behaves approximately like a logarithm for large positive "
    "masses but remains finite at zero and for negative measured annular masses. "
    "The same measured Y[J] scales predicted and observed annuli during fitting. "
    "It is never used to normalize predictions for a new halo. This objective "
    "is a fitting distance, not a likelihood or a covariance-weighted error model."
)
heading("Step 2: predict those targets from halo properties")
paragraph(
    "The input vector contains five measured log M200c values, plus either "
    "final log concentration (6 inputs total) or its five-epoch history "
    "(10 inputs). Concatenate the four a_star values across five epochs into "
    "20 targets. Missing inputs are imputed with calibration-training medians; "
    "features and targets are standardized using those training rows only."
)
equation(
    "Z = standardized halo feature matrix\n"
    "Y_a = standardized matrix of 20 correction targets\n"
    "W = (Z.T @ Z / n + lambda*I)^(-1) @ (Z.T @ Y_a / n)\n"
    "a_pred = undo_target_scaling(Z_new @ W)"
)
paragraph(
    "This is linear regression with a penalty that discourages unnecessarily "
    "large coefficients. Quadratic feature maps were also considered, but all "
    "five selected Exp77 fits were linear and selected w=0.25. The saved choices "
    "of features, scalings, imputation values, and penalty accompany each fit."
)
paragraph(
    "Each outer split uses approximately 60% of galaxies for the physical "
    "baseline, 20% for correction calibration, and 20% for evaluation. Candidate "
    "selection uses an additional split inside calibration. All epochs of a galaxy "
    "stay together. Five rotations provide held-out predictions for all 842 "
    "galaxies; Exp67 selection and validation samples remain unused."
)

page(4, "How many free parameters?")
paragraph(
    "<b>Yes: the hybrid adds fitted complexity relative to the physical baseline. "
    "No: Exp77 does not add correction coordinates relative to the Exp75 hybrid.</b> "
    "Three different parameter counts must be kept separate."
)
heading("1 / Profile coordinates")
paragraph(
    "There are four adjustable numbers per galaxy-epoch during calibration, "
    "or 20 across the five epochs. For a new galaxy these are outputs of the "
    "halo map, not independently fitted parameters and not user-supplied inputs. "
    "The numerous per-calibration-galaxy optimum corrections are training labels; "
    "they are not retained as per-galaxy free parameters at prediction time."
)
heading("2 / Population-level fitted coefficients")
equation(
    "linear correction map: 20 outputs * (d slopes + 1 intercept)\n"
    "d = 6:   140 effective affine coefficients\n"
    "d = 10:  220 effective affine coefficients\n"
    "plus 12 physical baseline parameters: 152 or 232"
)
paragraph(
    "These are algebraic coefficient counts per fitted model, not statistical "
    "effective degrees of freedom: regularization reduces flexibility. The "
    "implementation stores centered regression weights and target means/scales "
    "rather than an explicit intercept column. Standardization and imputation "
    "statistics are additional saved calibration state, not extra unconstrained "
    "affine prediction directions for complete inputs."
)
paragraph(
    "Saved Exp77 discovery folds 0, 1, 2, and 4 use the 10-input history map "
    "(220 correction coefficients each); fold 3 uses the 6-input map (140). "
    "These five fits are evaluation rotations, not five components that must be "
    "summed in a production predictor. A final production calibration has not "
    "been established by this experiment."
)
paragraph(
    "The annular weight, regression penalty, and linear/quadratic choice are "
    "calibration choices. They are frozen for prediction. Thus '12 physical "
    "parameters plus four corrections' is not a correct total model-complexity "
    "count: the four corrections have to be predicted by a learned map."
)
heading("3 / Parameters varied in downstream forward use")
paragraph(
    "A frozen, simulation-calibrated hybrid need not add any freely varied "
    "parameters in downstream use. Its regression coefficients can remain fixed, "
    "just as an emulator's coefficients do. Additional deformation knobs are a "
    "separate modeling choice, not all 140 or 220 regression coefficients. "
    "Changing the physical baseline after calibration can invalidate its learned "
    "residual correction and must be tested."
)

page(5, "Pseudocode: calibration and prediction")
paragraph(
    "Schematic pseudocode, matched to the implemented data flow. The physical "
    "baseline is frozen throughout Exp77. The direct model is not used here."
)
equation(
    "# CALIBRATION: stellar profiles are allowed as labels.\n"
    "for outer_split in five_galaxy_splits:\n"
    "    baseline = load_exp75_training_only_baseline(outer_split)\n"
    "    calibration, evaluation = fixed_remaining_roles()\n"
    "    for weight in [0, 0.25, 1]:\n"
    "        for galaxy in calibration:\n"
    "            for epoch in five_epochs:\n"
    "                labels[weight, galaxy, epoch] = minimize_loss(\n"
    "                    baseline[galaxy, epoch],\n"
    "                    measured_cog[galaxy, epoch], weight)\n"
    "    choice = select_inside_calibration_only(labels, halo_inputs)\n"
    "    halo_map = fit_regularized_map(\n"
    "        halo_inputs[calibration], labels[choice.weight])\n"
    "    save_held_out_predictions(\n"
    "        predict(halo_inputs[evaluation], baseline, halo_map))"
)
paragraph(
    "Selection minimizes held-out outer-annular error inside calibration, "
    "subject to the declared CoG-error allowance; it does not inspect the outer "
    "evaluation galaxies. Zero annular weight reproduces the saved Exp75 "
    "reference. Positive-weight fits begin from zero corrections."
)
equation(
    "# PREDICTION: no measured stellar mass or size enters.\n"
    "def predict(halo_catalog, physical_model, frozen_halo_map):\n"
    "    baseline = physical_model.predict(halo_catalog.full_mah)\n"
    "    features = measured_masses_and_concentrations(halo_catalog)\n"
    "    a = frozen_halo_map.predict(features).reshape(-1, 5, 4)\n"
    "    bins = radial_difference(baseline, initial_mass=0)\n"
    "    u = a[..., 1:] @ fixed_legendre_basis.T\n"
    "    fractions = softmax(log(bins) + u, axis=-1)\n"
    "    total = baseline[..., -1] * 10**a[..., 0]\n"
    "    return total[..., None] * cumsum(fractions, axis=-1)"
)
paragraph(
    "The implementation validates finite inputs and a positive, nondecreasing "
    "baseline. log(0) is handled as minus infinity for zero-mass bins. In actual "
    "code, zero corrections have an exact-identity shortcut. Coefficients are "
    "predicted jointly as 20 outputs, but no temporal constraint couples their "
    "five epoch-specific profile reconstructions."
)

page(6, "Forward use, evidence, and limits")
heading("A natural component, but not yet a production interface")
paragraph(
    "The conceptual chain is halo history and concentration -> physical CoG "
    "-> halo-predicted correction -> stellar masses and profiles. The full MAH, "
    "including times later than the prediction epoch, is legitimate input. No "
    "stellar-mass amplitude pin is introduced. This is compatible with HongShao's "
    "halo-to-galaxy boundary; downstream lensing or inference remains elsewhere."
)
paragraph(
    "The current hongshao/forward.py expects the established emulator objects, "
    "not this deposition-plus-correction object. Integration would require an "
    "explicit adapter and identity tests. A reasonable design is to freeze the "
    "calibrated hybrid as the reference and apply documented deformations to "
    "that reference. This ordering is a proposal, not implemented Exp77 behavior."
)
paragraph(
    "A probabilistic production model also needs calibrated residual scatter "
    "across radii and epochs, with validation of mass-plane widths and temporal "
    "histories. Exp77 has no such scatter layer and only evaluates five epochs; "
    "arbitrary-redshift interpolation and extrapolation beyond the trained halo "
    "population have not been validated. Radial positivity alone is insufficient."
)
heading("Promising, with specific qualifications")
paragraph(
    "Relative to the measured Exp75 hybrid, Exp77 reduces RMS outer-annular "
    "mass residual divided by true total from 0.03298 to 0.02843, a 13.81% "
    "improvement. Mean radial log-CoG RMS changes from 0.10771 to 0.10786 dex, "
    "essentially unchanged. At z=2 the median 100-148.22 kpc envelope bias falls "
    "from 88% above the measured mass to 5% above it. These support continued "
    "development, not a claim that every diagnostic or model comparison is won."
)
paragraph(
    "The predeclared relative size gate was missed at z=1.5: median R50 bias "
    "changes from 0.6% to 5.1% too large, exceeding the allowed 0.01-dex increase "
    "in absolute bias. Standard absolute R50-bias checks nevertheless pass at "
    "all five epochs. The z=2 conditional halo-growth response also deteriorates "
    "slightly. Preserve these recorded failures without treating the strict "
    "relative radius gate as a universal scientific rejection threshold."
)
heading("Implementation sources and reproducibility")
paragraph(
    "Repo-relative sources: <b>experiments/exp75_halo_residual_correction/"
    "correction.py</b> (basis, decoder, regression); <b>continuation.py</b> and "
    "<b>run.py</b> in that directory (halo inputs, physical/direct comparisons); "
    "<b>experiments/exp77_annular_correction_loss/annular.py</b> (loss, selection); "
    "that experiment's <b>outputs/discovery_fold*.json</b> and "
    "<b>outputs/summary.json</b> (measured choices and results); "
    "<b>hongshao/forward.py</b> and <b>docs/SPEC.md</b> (interface and contract)."
)
paragraph(
    "This explanatory document changes no fit, sample, QA figure, or interface. "
    "The accompanying build record stores the source revision and file hashes. "
    "Generated directly from build_model_note.py; equations use explicit indexed "
    "notation matching the array implementation."
)


def footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#cbd9df"))
    canvas.line(48, 39, 547, 39)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#526675"))
    canvas.drawString(48, 26, "HongShao | Exp77 | Model and profile correction")
    canvas.drawRightString(547, 26, str(document.page))
    canvas.restoreState()


document = SimpleDocTemplate(
    str(pdf_path),
    pagesize=(595.28, 841.89),
    rightMargin=48,
    leftMargin=48,
    topMargin=42,
    bottomMargin=53,
    title="Exp77: model and profile correction",
    author="HongShao research notes",
)
document.build(story, onFirstPage=footer, onLaterPages=footer)
pdf = pymupdf.open(pdf_path)
assert len(pdf) == 6, f"Unexpected pagination: {len(pdf)} pages"
for index, pdf_page in enumerate(pdf):
    assert pdf_page.get_text().strip(), f"Empty page {index + 1}"
    pdf_page.get_pixmap(matrix=pymupdf.Matrix(1.25, 1.25)).save(
        previews / f"page_{index + 1}.png"
    )
source_files = [Path(__file__), root / "hongshao/forward.py"]
for relative in (
    "exp75_halo_residual_correction/correction.py",
    "exp75_halo_residual_correction/run.py",
    "exp75_halo_residual_correction/continuation.py",
    "exp77_annular_correction_loss/annular.py",
    "exp77_annular_correction_loss/outputs/summary.json",
):
    source_files.append(root / "experiments" / relative)
source_files.extend(
    (root / "experiments/exp77_annular_correction_loss/outputs").glob(
        "discovery_fold*.json"
    )
)
record = {
    "git_sha": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip(),
    "document_date": "2026-09-09",
    "pages": len(pdf),
    "source_sha256": {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files
    },
    "pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
}
(destination / "exp77_model_note_build.json").write_text(
    json.dumps(record, indent=2) + "\n"
)
print(pdf_path)
print(f"Rendered {len(pdf)} pages to {previews}")
