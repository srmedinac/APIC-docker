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

# v1.0.5's page-1 furniture, converted from its A4 millimetres into template units by fraction of
# page: x * 2592/210 across, y * 3456/297 down. K converts v1's A4 point sizes (fonts, the pointer
# triangle, padding) into template points.
K = PAGE_W / 595.276
BAR_X, BAR_Y, BAR_W, BAR_H = 123.4, 314.2, 148.1, 465.5      # 10mm, 27mm, 12mm, 40mm
GRID_X, GRID_Y = 1752.9, 640.0                                # 142mm, 55mm
CELL_W, CELL_H = 271.5, 256.0                                 # 22mm square
CELL_GAP_X, CELL_GAP_Y = 86.4, 81.5                           # 7mm
POS_FILL = (0.914, 0.490, 0.455)                              # v1's #e97d74
NEG_FILL = (0.353, 0.592, 0.675)                              # v1's #5a97ac

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


def _example_patches(slide_path, mask_path, n=4, px=1024, out_px=420):
    """Example tiles from the tissue the score was computed on, for v1's 2x2 grid.

    v1 filled this grid from spatil_visualizations/, which the streaming pipeline does not keep.
    These are the H&E tiles themselves, taken from the densest well-separated regions of the same
    HistoQC mask the tiler used, so they are examples of the analysed tissue. They carry no nuclei
    overlay: the masks are streamed and discarded, and redrawing them would mean a second GPU pass.
    """
    import numpy as np
    from PIL import Image
    import openslide

    sl = openslide.OpenSlide(slide_path)
    W, H = sl.dimensions
    m = np.array(Image.open(mask_path).convert("L"))
    mh, mw = m.shape
    sx, sy = W / float(mw), H / float(mh)

    step = max(1, int(round(px / max(sx, sy))))
    cand = []
    for r in range(0, max(1, mh - step), step):
        for cc in range(0, max(1, mw - step), step):
            if float((m[r:r + step, cc:cc + step] > 0).mean()) >= 0.9:
                cand.append((r, cc))
    if not cand:
        return []
    picked, sep = [], max(step * 3, 1)
    for r, cc in cand:
        if all(abs(r - pr) > sep or abs(cc - pc) > sep for pr, pc in picked):
            picked.append((r, cc))
        if len(picked) == n:
            break
    out = []
    for r, cc in picked:
        try:
            t = sl.read_region((int(cc * sx), int(r * sy)), 0, (px, px)).convert("RGB")
            out.append(t.resize((out_px, out_px), Image.LANCZOS))
        except Exception:
            pass
    return out


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

    # v1.0.5's vertical NEGATIVE-to-POSITIVE bar with the pointer triangle, ported unchanged.
    # The split sits at the threshold on a fixed 0..2 scale and the pointer maps the score
    # piecewise inside whichever region it falls in. An earlier version here used a log scale
    # centred on the threshold, which put the triangle in a different place for the same score.
    _x, _y, _w, _h = BAR_X, BAR_Y, BAR_W, BAR_H
    cut = float(max(0.0, min(2.0, thr)))
    split_y = _y + (cut / 2.0) * _h
    c.setFillColorRGB(*NEG_FILL); c.rect(_x, _y, _w, max(0, split_y - _y), fill=1, stroke=0)
    c.setFillColorRGB(*POS_FILL); c.rect(_x, split_y, _w, max(0, (_y + _h) - split_y), fill=1, stroke=0)
    c.setStrokeColorRGB(1, 1, 1); c.setLineWidth(0.8 * K)
    c.line(_x - 2 * K, split_y, _x + _w + 2 * K, split_y)

    c.setFillColorRGB(1, 1, 1); c.setFont("Helvetica-Bold", 6.5 * K)
    pos_h, neg_h = (_y + _h) - split_y, split_y - _y
    pos_mid = split_y + max(6 * K, pos_h / 2.0) - 4 * K
    neg_mid = _y + max(6 * K, neg_h / 2.0) - 4 * K
    if pos_h < 8 * K: pos_mid = split_y + 6 * K
    if neg_h < 8 * K: neg_mid = _y + 6 * K
    c.drawCentredString(_x + _w / 2.0, pos_mid, "POSITIVE")
    c.drawCentredString(_x + _w / 2.0, neg_mid, "NEGATIVE")

    pad = 2.0 * K
    sc = float(risk)
    if sc <= cut:
        frac = (sc / cut) if cut > 0 else 0.0
        py = _y + pad + frac * (max(0, split_y - _y) - 2 * pad)
    else:
        pmin, pmax = cut + 0.01, 2.0
        if sc > pmax:
            py = _y + _h - pad
        elif sc <= pmin:
            py = split_y + pad
        else:
            py = split_y + pad + ((sc - pmin) / (pmax - pmin)) * (max(0, (_y + _h) - split_y) - 2 * pad)

    ptr = POS_FILL if sc >= (cut + 0.01) else NEG_FILL
    c.setFillColorRGB(*ptr); c.setStrokeColorRGB(*ptr)
    aw, ah = 8 * K, 10 * K
    axl = _x + _w + 4 * K
    pth = c.beginPath()
    pth.moveTo(axl, py - ah / 2); pth.lineTo(axl + aw, py); pth.lineTo(axl, py + ah / 2)
    pth.close(); c.drawPath(pth, stroke=0, fill=1)
    c.setFont("Helvetica-Bold", 10 * K)
    c.drawString(axl + aw + 3 * K, py - 3 * K, "%.2f" % sc)

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

    # The 2x2 example grid, at v1.0.5's GRID_X / GRID_Y / CELL geometry.
    try:
        _tiles = _example_patches(_sl, _mk) if (_sl and _mk and os.path.isfile(_mk)) else []
    except Exception:
        _tiles = []
    for _i, _t in enumerate(_tiles[:4]):
        _row, _col = divmod(_i, 2)
        c.drawImage(ImageReader(_t),
                    GRID_X + _col * (CELL_W + CELL_GAP_X),
                    GRID_Y - _row * (CELL_H + CELL_GAP_Y),
                    width=CELL_W, height=CELL_H,
                    preserveAspectRatio=True, anchor="c", mask="auto")

    c.showPage(); c.save(); buf.seek(0)

    writer = PdfWriter()
    base = PdfReader(tpl_path).pages[0]
    base.merge_page(PdfReader(buf).pages[0])
    writer.add_page(base)

    # ---------- page 2: v1's branded page, unchanged ----------
    # v1's own page 2 with every fabricated field redacted out and nothing added. The result block
    # and the extrapolation banner that used to be drawn here are gone: page 2 is the supplemental
    # information sheet, exactly as the stable report had it.
    writer.add_page(PdfReader(os.path.join(template_dir, P2_TEMPLATE)).pages[0])

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
