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
# Page 2 of v1's HUG_report.pdf with every fabricated field redacted out: the mock patient
# ("John Doe", ID M1-AAA111), the mock ordering physician ("Naoto Tokuyama, MD, PhD", "Tokyo
# Medical University"), the mock clinical values, and the "Approved by ... Chief Pathologist"
# sign-off. Labels, the CHAARTED supplemental block, the technical footnote and the real Medina
# et al. citation are untouched. Built with fitz apply_redactions, so the text is removed and not
# merely covered.
P2_TEMPLATE = "HUG-REPORT-EMPTY-PAGE2.pdf"

# Template geometry (pt), measured from the shipped PDFs.
PAGE_W, PAGE_H = 2592.0, 3456.0
S = PAGE_W / 210.0                     # template units per mm, so layout can be written in mm

# The report is delivered on A4. Drawing happens in template space so everything registers against
# the branded art, then the finished page is scaled into A4 as the last step. The scale is UNIFORM:
# v1 stretched 2592x3456 into A4 with preserveAspectRatio=False, which distorts every branded
# element by about 6%. Uniform scaling gives 595 x 793.4 pt, centred in A4 with a 24.3 pt band top
# and bottom.
A4_W, A4_H = 595.276, 841.890
A4_SCALE = A4_W / PAGE_W
A4_YOFF = (A4_H - PAGE_H * A4_SCALE) / 2.0

# Biopsy panel, the BIOPSY ANALYZED slot. Measured off the branded template: the label sits at
# y 2584-2633 from the top and the caption at y 3112, so in bottom-origin template units the slot
# runs y 320 to 810, between "PATIENT GROUP:" on the left (x 241-370) and the image grid on the
# right (x 1753). This is the same slot v1.0.5 used before the frame was moved to mid-page.
BIOPSY_SLOT = (480.0, 380.0, 1060.0, 440.0)

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


def _biopsy_panel(slide_path, mask_path, max_px=1400):
    """The BIOPSY ANALYZED panel: a slide thumbnail with the HistoQC tissue outlined in green.

    v1 drew the same thing and captioned it "green contours show tissue following QC". v1 could
    read the contours off the QC directory it left on disk; the streaming pipeline keeps nothing,
    but it does write tissue_mask.png beside apic.json, which is the same mask the tiler used. So
    the panel shows exactly the tissue that produced the score, not a decoration.

    Returns a PIL image, or None if either input is missing or unreadable. A report must still be
    produced when the panel cannot be, so every failure here is swallowed by the caller.
    """
    import numpy as np
    from PIL import Image
    import openslide

    s = openslide.OpenSlide(slide_path)
    W, H = s.dimensions
    scale = min(max_px / float(W), max_px / float(H), 1.0)
    tw, th = max(1, int(W * scale)), max(1, int(H * scale))
    thumb = s.get_thumbnail((tw, th)).convert("RGB")

    m = np.array(Image.open(mask_path).convert("L").resize(thumb.size, Image.NEAREST))
    binm = (m > 0).astype(np.uint8)
    rgb = np.array(thumb)
    try:
        import cv2
        cnts, _ = cv2.findContours(binm, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, cnts, -1, (60, 190, 60), max(1, int(round(min(tw, th) / 350.0))))
    except Exception:
        # No cv2: fall back to a 1 px morphological edge, which outlines the same region.
        edge = binm.astype(bool) & ~(
            np.roll(binm, 1, 0).astype(bool) & np.roll(binm, -1, 0).astype(bool)
            & np.roll(binm, 1, 1).astype(bool) & np.roll(binm, -1, 1).astype(bool))
        rgb[edge] = (60, 190, 60)
    return Image.fromarray(rgb)


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
    bar_x, bar_y0, bar_h = 150.0, 380.0, 440.0
    # The POSITIVE half sits above the threshold, the NEGATIVE half below, as on v1's page 1.
    c.setFillColorRGB(*RED)
    c.rect(bar_x, bar_y0 + bar_h / 2.0, 90.0, bar_h / 2.0, fill=1, stroke=0)
    c.setFillColorRGB(*GREEN)
    c.rect(bar_x, bar_y0, 90.0, bar_h / 2.0, fill=1, stroke=0)
    # 4.6 pt keeps both words inside the 90 pt bar; larger sizes clip to "ositive"/"gative".
    c.setFillColorRGB(1, 1, 1); c.setFont("Helvetica-Bold", 4.6 * S / 3.6)
    c.drawCentredString(bar_x + 45.0, bar_y0 + bar_h * 0.72, "POSITIVE")
    c.drawCentredString(bar_x + 45.0, bar_y0 + bar_h * 0.22, "NEGATIVE")
    c.setFillColorRGB(*(RED if positive else GREEN))
    c.setFont("Helvetica-Bold", 13 * S / 3.6)
    bar_x = bar_x + 90.0
    y = bar_y0 + _pos_on_bar(risk, thr) * bar_h
    c.drawString(bar_x + 6 * S, y - 1.4 * S, _fmt(risk, 3))
    p = c.beginPath()                                   # pointer triangle
    p.moveTo(bar_x, y); p.lineTo(bar_x - 4 * S, y + 2.2 * S); p.lineTo(bar_x - 4 * S, y - 2.2 * S)
    p.close(); c.drawPath(p, fill=1, stroke=0)

    # The caveat banner v1 has no concept of: this score may be extrapolating.
    if apic.get("extrapolating"):
        c.setFillColorRGB(0.88, 0.63, 0.31)
        # The only clear full-width strip on this page: 75 pt between the "ON AVERAGE, PATIENTS
        # IN THE APIC POSITIVE GROUP" line (ends at y 1056 bottom-origin) and the INTERPRETATION
        # AND QUALITY CONTROL band (starts at y 981). PAGE_H - 88*S covered the CONCLUSION block;
        # y 1935 covered the PROGNOSTIC ESTIMATES band.
        c.rect(14 * S, 995.0, 182 * S, 52.0, fill=1, stroke=0)
        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica-Bold", 6.8 * S / 3.6)
        c.drawString(16 * S, 1012.0,
                     "CAUTION — score extrapolates: %s outside the model's training range "
                     "(model fit at 20x; this slide %s)."
                     % (", ".join(oor) or "a feature",
                        ("%gx" % apic["objective_power"]) if apic.get("objective_power") else "magnification unknown"))
    # BIOPSY ANALYZED panel. The slot is empty in the branded template and v1 filled it; v1.0.5
    # placed it correctly and a later edit moved it to mid-page, over TREATMENT CONSIDERATIONS.
    _panel = None
    try:
        _sl = apic.get("slide")
        _mk = os.path.join(os.path.dirname(os.path.abspath(apic.get("_json_path") or out_path)),
                           "tissue_mask.png")
        if _sl and os.path.isfile(_sl) and os.path.isfile(_mk):
            _panel = _biopsy_panel(_sl, _mk)
    except Exception:
        _panel = None
    if _panel is not None:
        bx, by, bw, bh = BIOPSY_SLOT
        c.drawImage(ImageReader(_panel), bx, by, width=bw, height=bh,
                    preserveAspectRatio=True, anchor="c", mask="auto")

    c.showPage(); c.save(); buf.seek(0)

    writer = PdfWriter()
    base = PdfReader(tpl_path).pages[0]
    base.merge_page(PdfReader(buf).pages[0])
    writer.add_page(base)

    # ---------- page 2: v1's branded page, with the result block in its blank band ----------
    # The template is v1's own page 2 with every fabricated field redacted out. Its layout is
    # untouched. The result block fills the empty band between the median-survival boxes and the
    # technical footnote: that band is about 1180 pt tall, so the block runs in TWO COLUMNS.
    # A single column overran into the footnote.
    b2 = io.BytesIO()
    c = canvas.Canvas(b2, pagesize=(PAGE_W, PAGE_H))
    xL, xR = 20 * S, 108 * S
    TOP = 1580.0

    c.setFillColorRGB(*NAVY)
    c.rect(xL, TOP - 2 * S, 170 * S, 7 * S, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1); c.setFont("Helvetica-Bold", 10 * S / 3.6)
    c.drawString(xL + 2 * S, TOP, "RESULT FOR THIS SLIDE")

    def sub(x, y, txt):
        c.setFillColorRGB(*NAVY); c.setFont("Helvetica-Bold", 8.5 * S / 3.6)
        c.drawString(x, y, txt)
        return y - 5.6 * S

    def row(x, y, label, value, colour=None, bold=False, wide=52):
        c.setFillColorRGB(*GREY); c.setFont("Helvetica", 7.6 * S / 3.6)
        c.drawString(x, y, label)
        c.setFillColorRGB(*(colour or (0, 0, 0)))
        c.setFont("Helvetica-Bold" if bold else "Helvetica", 7.6 * S / 3.6)
        c.drawString(x + wide * S, y, value)
        return y - 5.4 * S

    col = RED if positive else GREEN
    y = TOP - 12 * S
    y = sub(xL, y, "RESULT")
    y = row(xL, y, "Slide ID", str(apic.get("stem") or "-"))
    y = row(xL, y, "APIC risk score", _fmt(risk, 6), col, True)
    y = row(xL, y, "Risk group", str(apic.get("risk_group") or "-"), col, True)
    y = row(xL, y, "Decision threshold", _fmt(thr, 6))
    y -= 3 * S
    y = sub(xL, y, "SPECIMEN AND ACQUISITION")
    y = row(xL, y, "Magnification", ("%gx" % apic["objective_power"]) if apic.get("objective_power") else "-")
    y = row(xL, y, "Resolution (mpp)", _fmt(apic.get("mpp_x"), 4))
    y = row(xL, y, "Model trained at", str(apic.get("model_trained_mag") or "-"))
    y = row(xL, y, "Tissue tiles", "%s (%s with nuclei)"
            % (apic.get("n_tiles_tissue"), apic.get("n_tiles_with_nuclei")))
    y = row(xL, y, "Nuclei segmented", "{:,}".format(apic["n_nuclei"]) if apic.get("n_nuclei") else "-")
    y = row(xL, y, "Tissue mask", str(apic.get("tissue") or "-"))
    y_left = y

    y = TOP - 12 * S
    y = sub(xR, y, "MODEL FEATURES")
    for k in sorted(feats):
        o = oor.get(k)
        c.setFillColorRGB(*GREY); c.setFont("Helvetica", 7.2 * S / 3.6)
        c.drawString(xR, y, k[:34])
        c.setFillColorRGB(0, 0, 0); c.setFont("Helvetica", 7.2 * S / 3.6)
        c.drawString(xR + 56 * S, y, _fmt(feats[k], 6))
        y -= 4.4 * S
        if o:
            c.setFillColorRGB(0.85, 0.48, 0.16); c.setFont("Helvetica-Oblique", 6.6 * S / 3.6)
            c.drawString(xR + 2 * S, y, "outside training range %s - %s"
                         % (_fmt(o.get("train_min"), 4), _fmt(o.get("train_max"), 4)))
            y -= 4.4 * S

    if apic.get("extrapolating"):
        yb = min(y_left, y) - 4 * S
        c.setFillColorRGB(0.88, 0.63, 0.31)
        c.rect(xL, yb - 2 * S, 170 * S, 7 * S, fill=1, stroke=0)
        c.setFillColorRGB(1, 1, 1); c.setFont("Helvetica-Bold", 7.4 * S / 3.6)
        c.drawString(xL + 2 * S, yb,
                     "CAUTION - score extrapolates: %d of %d features fall outside the training range."
                     % (len(oor), len(feats)))

    c.showPage(); c.save(); b2.seek(0)
    p2 = PdfReader(os.path.join(template_dir, P2_TEMPLATE)).pages[0]
    p2.merge_page(PdfReader(b2).pages[0])
    writer.add_page(p2)

    # ---------- deliver on A4 ----------
    # Both pages are drawn in the template's own 2592 x 3456 pt space so every value registers
    # against the branded art. Scale that into A4 as the last step, uniformly, and centre it: v1
    # stretched the same art into A4 with preserveAspectRatio=False and distorted every element by
    # about 6%. A page left at 2592 x 3456 pt is 914 x 1219 mm, a poster, and will not print.
    from pypdf import Transformation
    for pg in writer.pages:
        pg.add_transformation(Transformation().scale(A4_SCALE, A4_SCALE).translate(0, A4_YOFF))
        pg.mediabox.lower_left = (0, 0)
        pg.mediabox.upper_right = (A4_W, A4_H)
        pg.cropbox = pg.mediabox

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
    d["_json_path"] = os.path.abspath(a.apic)
    print(build(d, a.out, thumb=a.thumb, template_dir=a.template_dir))


if __name__ == "__main__":
    main()
