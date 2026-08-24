"""APIC patient report (PDF) for one slide — v1's branded template, without v1's defects.

Page 1  = v1's own EMORY/APIC template (POSITIVE or NEGATIVE variant) with our values overlaid as
          VECTOR text/graphics via pypdf. v1 rasterised the 2592x3456 pt template at 300 dpi
          (a ~466 MB pixmap per page) and drew it into A4 with preserveAspectRatio=False, stretching
          every branded element ~6% and producing a 6 MB file. We merge onto the original page, so
          the output stays vector, sharp and small.
Page 2  = OURS, generated from scratch. v1's page 2 rasterises page 2 of the hand-filled MOCK, so
          every v1 report leaks a fake patient ("John Doe ... PSA 40 ... approved by ... MD"). We do
          not touch that file. Our page 2 carries provenance and the caveats instead.

NO PHI. The only identifier printed is the slide/stem the viewer already shows on screen. There are
no name / DOB / physician / PSA fields to fill, so none can be mis-filled or leaked.

Honest omissions (deliberate, and stated on the report):
  * the "5-year risk of death" / "2-year risk of castration resistance" boxes stay EMPTY — we have a
    Cox linear predictor, not a calibrated absolute-risk model. v1 left them empty too; the numbers
    in the mock are illustrative and must never be reused as if they were a result.
  * nuclei / spaTIL thumbnails are omitted unless the run saved them: the pipeline streams masks and
    discards them, so re-creating those pictures would mean a full GPU re-inference for decoration.

Usage:
    python scripts/apic_report.py --apic <apic.json> --out <report.pdf> [--thumb <slide_thumb.jpg>]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import math
import os

TEMPLATE_DIR = os.environ.get("APIC_TEMPLATE_DIR", "/opt/apic/v1/data")
POS_TEMPLATE = "HUG-REPORT-EMPTY-POSITIVE.pdf"
NEG_TEMPLATE = "HUG-REPORT-EMPTY-NEGATIVE.pdf"

# Template geometry (pt), measured from the shipped PDFs.
PAGE_W, PAGE_H = 2592.0, 3456.0
S = PAGE_W / 210.0                     # template units per mm, so layout can be written in mm

# Published CHAARTED (ECOG-ACRIN E3805) validation figures, exactly as stated on v1's page 2. These
# are COHORT statistics from the paper, never patient-specific predictions, and are labelled as such.
CHAARTED = {
    "n": 208,
    "pos": {"pct": 56, "n": 118, "adt": ("2.9", "2.1-4.2"), "doce": ("4.1", "3.4-6.6")},
    "neg": {"pct": 44, "n": 90, "adt": ("5.3", "3.7-NR*"), "doce": ("4.4", "3.7-NR*")},
}
CITATION = ("Medina S, Tokuyama N, et al. Computational pathology to predict docetaxel benefit in "
            "patients with metastatic hormone-sensitive prostate cancer from the CHAARTED trial "
            "(ECOG-ACRIN E3805). JCO 43, 329-329 (2025). DOI:10.1200/JCO.2025.43.5_suppl.329")
VALIDATION = ("APIC is a predictive biomarker classifier validated on H&E prostate core-needle biopsies "
              "scanned at 20x (0.4598 um/pixel). Clinical validation on CHAARTED (n=208) and "
              "NRG/RTOG-0521 (n=266) demonstrated significant treatment-biomarker interactions (p<0.05). "
              "Requires artifact removal with pathologist review and adequate staining. Known limitations "
              "include unknown performance on post-treatment biopsies and slides with insufficient tissue. "
              "Revalidation required for scanner, staining or software changes. "
              "Research use only - not FDA approved.")

NAVY = (0.106, 0.204, 0.412)
RED = (0.851, 0.325, 0.310)            # matches the viewer card's #d9534f
GREEN = (0.290, 0.616, 0.357)          # matches #4a9d5b
GREY = (0.42, 0.45, 0.50)


def _fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float) and not math.isfinite(v):
        return "—"
    return ("%%.%df" % nd) % v if isinstance(v, (int, float)) else str(v)


def _pos_on_bar(risk, thr, span=3.0):
    """Where the pointer sits, 0=bottom .. 1=top. risk is exp(lp), i.e. a hazard ratio, so map it on
    a LOG scale centred on the threshold — v1 used a two-slope linear 0-2 scale that made equal
    ratios look unequal on either side of the cut."""
    if not risk or risk <= 0 or not thr or thr <= 0:
        return 0.5
    return 0.5 + 0.5 * max(-1.0, min(1.0, math.log(risk / thr) / math.log(span)))


def build(apic, out_path, thumb=None, template_dir=TEMPLATE_DIR):
    from pypdf import PdfReader, PdfWriter
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    risk = apic.get("risk_score")
    thr = apic.get("threshold")
    positive = (apic.get("risk_group") == "High Risk")
    feats = apic.get("features") or {}
    oor = {o["feature"]: o for o in (apic.get("features_out_of_train_range") or [])}

    # ---------- page 1: overlay on v1's template ----------
    tpl_path = os.path.join(template_dir, POS_TEMPLATE if positive else NEG_TEMPLATE)
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(PAGE_W, PAGE_H))

    # Identifier block — the ONLY identifier is the slide stem, which the viewer already displays.
    c.setFont("Helvetica-Bold", 9 * S / 3.6)
    c.setFillColorRGB(*NAVY)
    c.drawString(14 * S, PAGE_H - 62 * S, "Slide ID:")
    c.setFont("Helvetica", 9 * S / 3.6)
    c.drawString(30 * S, PAGE_H - 62 * S, str(apic.get("stem") or "—")[:60])
    c.setFont("Helvetica-Bold", 9 * S / 3.6)
    c.drawString(14 * S, PAGE_H - 68 * S, "Analysed:")
    c.setFont("Helvetica", 9 * S / 3.6)
    made = _dt.datetime.fromtimestamp(apic.get("_mtime") or _dt.datetime.utcnow().timestamp(), _dt.timezone.utc)
    c.drawString(30 * S, PAGE_H - 68 * S, made.strftime("%Y-%m-%d"))

    # Risk score next to the POSITIVE/NEGATIVE bar (the bar art is part of the template).
    c.setFont("Helvetica-Bold", 13 * S / 3.6)
    c.setFillColorRGB(*(RED if positive else GREEN))
    bar_x, bar_y0, bar_h = 27.0 * S, 236.0 * S, 40.0 * S
    y = bar_y0 + _pos_on_bar(risk, thr) * bar_h
    c.drawString(bar_x + 6 * S, y - 1.4 * S, _fmt(risk, 3))
    p = c.beginPath()                                   # pointer triangle
    p.moveTo(bar_x, y); p.lineTo(bar_x - 4 * S, y + 2.2 * S); p.lineTo(bar_x - 4 * S, y - 2.2 * S)
    p.close(); c.drawPath(p, fill=1, stroke=0)

    # The caveat banner v1 has no concept of: this score may be extrapolating.
    if apic.get("extrapolating"):
        c.setFillColorRGB(0.88, 0.63, 0.31)
        c.rect(14 * S, PAGE_H - 88 * S, 182 * S, 9 * S, fill=1, stroke=0)
        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica-Bold", 7.4 * S / 3.6)
        c.drawString(16 * S, PAGE_H - 84.2 * S,
                     "CAUTION — score extrapolates: %s outside the model's training range "
                     "(model fit at 20x; this slide %s)."
                     % (", ".join(oor) or "a feature",
                        ("%gx" % apic["objective_power"]) if apic.get("objective_power") else "magnification unknown"))
    c.showPage(); c.save(); buf.seek(0)

    writer = PdfWriter()
    base = PdfReader(tpl_path).pages[0]
    base.merge_page(PdfReader(buf).pages[0])
    writer.add_page(base)

    # ---------- page 2: ours — provenance, not a mock patient ----------
    b2 = io.BytesIO()
    c = canvas.Canvas(b2, pagesize=(PAGE_W, PAGE_H))
    x0, y = 14 * S, PAGE_H - 20 * S

    def head(txt, size=11):
        nonlocal y
        c.setFillColorRGB(*NAVY); c.rect(x0, y - 2 * S, 182 * S, 7 * S, fill=1, stroke=0)
        c.setFillColorRGB(1, 1, 1); c.setFont("Helvetica-Bold", size * S / 3.6)
        c.drawString(x0 + 2 * S, y, txt); y -= 11 * S

    def line(label, value, warn=False):
        nonlocal y
        c.setFillColorRGB(*GREY); c.setFont("Helvetica", 8 * S / 3.6)
        c.drawString(x0 + 2 * S, y, label)
        c.setFillColorRGB(*(RED if warn else (0.12, 0.14, 0.18)))
        c.setFont("Helvetica-Bold" if warn else "Helvetica", 8 * S / 3.6)
        c.drawString(x0 + 70 * S, y, str(value)); y -= 5.4 * S

    c.setFillColorRGB(*NAVY); c.setFont("Helvetica-Bold", 15 * S / 3.6)
    c.drawString(x0, y, "APIC — analysis detail"); y -= 6 * S
    c.setFillColorRGB(*GREY); c.setFont("Helvetica", 8 * S / 3.6)
    c.drawString(x0, y, "Technical provenance for the result on page 1. Research use only."); y -= 12 * S

    head("RESULT")
    line("APIC risk score", _fmt(risk, 6))
    line("Risk group", "%s  (%s)" % (apic.get("risk_group") or "—", apic.get("apic_status") or "—"))
    line("Decision threshold", _fmt(thr, 6))
    y -= 3 * S

    # ---- the evidence, from v1's page 2: what this GROUP looked like in CHAARTED --------------
    head("WHAT THIS GROUP LOOKED LIKE IN CHAARTED  (published cohort data, not a patient prediction)")
    c.setFillColorRGB(0.12, 0.14, 0.18); c.setFont("Helvetica", 7.8 * S / 3.6)
    for chunk in _wrap(
        "Retrospective analysis of %d men with metastatic hormone-sensitive prostate cancer in "
        "CHAARTED (ECOG-ACRIN E3805), testing the addition of docetaxel to androgen deprivation "
        "therapy. %d%% (%d) were APIC-Positive and had a significant survival benefit from adding "
        "docetaxel; %d%% (%d) were APIC-Negative and showed no benefit."
            % (CHAARTED["n"], CHAARTED["pos"]["pct"], CHAARTED["pos"]["n"],
               CHAARTED["neg"]["pct"], CHAARTED["neg"]["n"]), 112):
        c.drawString(x0 + 2 * S, y, chunk); y -= 4.8 * S
    y -= 4 * S

    # median-survival table; the patient's own group is highlighted so the relevant row is obvious
    def survival_block(gx, key, title, colour, mine):
        yy = y
        if mine:                                    # highlight the group this slide falls into
            c.setFillColorRGB(*[min(1.0, ch + 0.86 * (1 - ch)) for ch in colour])
            c.rect(gx - 2 * S, yy - 20 * S, 88 * S, 26 * S, fill=1, stroke=0)
        c.setFillColorRGB(*colour); c.circle(gx + 6 * S, yy - 6 * S, 5.5 * S, fill=1, stroke=0)
        c.setFillColorRGB(1, 1, 1); c.setFont("Helvetica-Bold", 13 * S / 3.6)
        c.drawCentredString(gx + 6 * S, yy - 8.4 * S, "+" if key == "pos" else "–")
        c.setFillColorRGB(*NAVY); c.setFont("Helvetica-Bold", 8.6 * S / 3.6)
        c.drawString(gx + 15 * S, yy - 2 * S, title + (" ← this slide" if mine else ""))
        d = CHAARTED[key]
        c.setFont("Helvetica", 7.6 * S / 3.6); c.setFillColorRGB(0.12, 0.14, 0.18)
        for i, (lab, val) in enumerate((("ADT alone", d["adt"]), ("ADT + docetaxel", d["doce"]))):
            yr = yy - (8 + i * 6) * S
            c.drawString(gx + 15 * S, yr, lab)
            c.setFont("Helvetica-Bold", 7.6 * S / 3.6)
            c.drawString(gx + 52 * S, yr, "%s yr" % val[0])
            c.setFont("Helvetica", 6.4 * S / 3.6); c.setFillColorRGB(*GREY)
            c.drawString(gx + 66 * S, yr, "95%%CI %s" % val[1])
            c.setFont("Helvetica", 7.6 * S / 3.6); c.setFillColorRGB(0.12, 0.14, 0.18)

    c.setFillColorRGB(*GREY); c.setFont("Helvetica", 6.8 * S / 3.6)
    c.drawString(x0 + 2 * S, y + 3 * S, "Median overall survival by group and treatment:")
    survival_block(x0 + 2 * S, "pos", "APIC-Positive", RED, positive)
    survival_block(x0 + 96 * S, "neg", "APIC-Negative", (0.32, 0.60, 0.75), not positive)
    y -= 26 * S

    head("MODEL FEATURES  (value, and the range the model was trained on)")
    for k, v in feats.items():
        o = oor.get(k)
        rng = ("%s – %s" % (_fmt(o["train_min"], 4), _fmt(o["train_max"], 4))) if o else ""
        line(k, "%s   %s" % (_fmt(v, 6), ("[outside training range %s]" % rng) if o else ""), warn=bool(o))
    y -= 3 * S

    head("SPECIMEN & ACQUISITION")
    line("Magnification", ("%gx" % apic["objective_power"]) if apic.get("objective_power") else "unknown",
         warn=not apic.get("objective_power"))
    line("Resolution (mpp)", _fmt(apic.get("mpp_x"), 4))
    line("Model trained at", apic.get("model_trained_mag") or "20x (CHAARTED)")
    line("Tissue tiles analysed", "%s (%s contained nuclei)" % (apic.get("n_tiles_tissue"), apic.get("n_tiles_with_nuclei")))
    line("Nuclei segmented", "{:,}".format(apic.get("n_nuclei") or 0))
    line("Tissue mask", apic.get("tissue") or "—", warn=(apic.get("tissue") not in (None, "histoqc")))
    y -= 3 * S

    if apic.get("warning"):
        head("CAUTION")
        c.setFillColorRGB(*RED); c.setFont("Helvetica", 8 * S / 3.6)
        for chunk in _wrap(apic["warning"], 105):
            c.drawString(x0 + 2 * S, y, chunk); y -= 5 * S
        y -= 3 * S

    # validation statement + citation, mirroring v1's footer
    c.setFillColorRGB(*GREY); c.setFont("Helvetica", 6.4 * S / 3.6)
    y = max(y, 34 * S)
    for chunk in _wrap(VALIDATION, 150):
        c.drawString(x0, y, chunk); y -= 3.8 * S
    y -= 2 * S
    for chunk in _wrap("Which group benefits depends on the therapy: APIC-Positive is associated with "
                       "benefit from docetaxel (CHAARTED); APIC-Negative with benefit from enzalutamide "
                       "(ENZAMET). Absolute-risk estimates are not reported: APIC yields a relative risk "
                       "score, not a calibrated absolute risk.", 150):
        c.drawString(x0, y, chunk); y -= 3.8 * S
    y -= 2 * S
    for chunk in _wrap(CITATION, 150):
        c.drawString(x0, y, chunk); y -= 3.8 * S

    if thumb and os.path.isfile(thumb):
        try:
            c.drawImage(ImageReader(thumb), x0, 14 * S, width=60 * S, height=45 * S,
                        preserveAspectRatio=True, anchor="sw", mask="auto")
            c.setFillColorRGB(*GREY); c.setFont("Helvetica", 7 * S / 3.6)
            c.drawString(x0, 10 * S, "Slide overview")
        except Exception:
            pass
    c.showPage(); c.save(); b2.seek(0)
    writer.add_page(PdfReader(b2).pages[0])

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        writer.write(f)
    return out_path


def _wrap(text, n):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > n:
            out.append(line); line = w
        else:
            line = (line + " " + w).strip()
    if line:
        out.append(line)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apic", required=True, help="path to apic.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thumb", default=None)
    ap.add_argument("--template-dir", default=TEMPLATE_DIR)
    a = ap.parse_args()
    d = json.load(open(a.apic))
    try:
        d["_mtime"] = os.path.getmtime(a.apic)
    except OSError:
        pass
    print(build(d, a.out, thumb=a.thumb, template_dir=a.template_dir))


if __name__ == "__main__":
    main()
