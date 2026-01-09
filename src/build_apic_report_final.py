# build_apic_report_final.py
import io, sys, argparse, random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from reportlab.platypus import Paragraph, Frame
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors
import fitz  # PyMuPDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4  # 595 x 842 pt
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader

PAGE_W, PAGE_H = A4

# ==============================
# STYLE / COLOR (match PDF2)
# ==============================
POS_FILL = colors.HexColor("#e97d74")  # salmon/red (Positive)
NEG_FILL = colors.HexColor("#5a97ac")  # teal/blue (Negative)
TEXT_COLOR = colors.black

# ==============================
# LAYOUT: PAGE 1 INSERT COORDS
# (adjust these values a few pt/mm if you need micro-alignment)
# All sizes/positions in points; 1 mm = 2.83465 pt
# ==============================
# Conclusion block
CONC_ICON_CENTER = (20*mm, PAGE_H - 92*mm)     # center (x,y) for +/- icon
CONC_ICON_RADIUS = 8*mm
CONC_TEXT_FRAME  = (42*mm, PAGE_H - 110*mm, 155*mm, 30*mm)  # (x, y, w, h)

# Treatment Considerations (one-line uppercase string)
TREAT_TEXT_FRAME = (43*mm, PAGE_H - 216*mm, 170*mm, 28*mm)

# Interpretation & Quality Control:
# Part-1: vertical bar + pointer + value
BAR_X, BAR_Y = (10*mm, PAGE_H - 270*mm)  # bottom-left of bar
BAR_W, BAR_H = (12*mm, 40*mm)
POINTER_RIGHT_PAD = 12  # px gap for arrow width to the right of the bar
# Pointer value text
POINTER_VALUE_DX = 28
POINTER_VALUE_DY = -3

# Part-2: biopsy analyzed image
BIOPSY_IMG_FRAME = (45*mm, PAGE_H - 270*mm, 85*mm, 48*mm)  # (x,y,w,h)

# Part-3: 4-grid spatil visualizations
GRID_X, GRID_Y = (142*mm, PAGE_H - 242*mm)  # top-left of grid
CELL_W, CELL_H, CELL_GAP = (22*mm, 22*mm, 7*mm)



# ==============================
# UTILITIES
# ==============================
def mm2pt(xmm: float) -> float:
    return xmm * mm

def _pil_reader(img: Image.Image) -> ImageReader:
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    bio.seek(0)
    return ImageReader(bio)

def rasterize_page(pdf_path: Path, page_index: int, dpi=300) -> ImageReader:
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    pm = page.get_pixmap(matrix=fitz.Matrix(dpi/72.0, dpi/72.0), alpha=False)
    img = Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
    doc.close()
    return _pil_reader(img)

# def draw_image_keep_ar(c: canvas.Canvas, img_path: Path, x, y, w, h):
#     try:
#         im = Image.open(img_path)
#         im = ImageOps.exif_transpose(im)
#         iw, ih = im.size
#         s = min(w/iw, h/ih)
#         dw, dh = iw*s, ih*s
#         c.drawImage(_pil_reader(im), x + (w-dw)/2, y + (h-dh)/2, dw, dh,
#                     preserveAspectRatio=False, mask='auto')
#     except Exception:
#         pass

def _draw_pil_keep_ar(c: canvas.Canvas, im: Image.Image, x, y, w, h):
    """Draw a PIL image into (x,y,w,h) while preserving aspect ratio."""
    iw, ih = im.size
    s = min(w / iw, h / ih)
    dw, dh = iw * s, ih * s
    c.drawImage(
        _pil_reader(im),
        x + (w - dw) / 2,
        y + (h - dh) / 2,
        dw,
        dh,
        preserveAspectRatio=False,
        mask='auto'
    )

def draw_image_keep_ar(c: canvas.Canvas, img_path: Path, x, y, w, h):
    try:
        im = Image.open(img_path)
        im = ImageOps.exif_transpose(im)
        _draw_pil_keep_ar(c, im, x, y, w, h)
    except Exception:
        pass


def read_threshold_and_score(final_features_dir: Path):
    csvs = sorted(final_features_dir.glob("*_prediction.csv"))
    if not csvs:
        raise FileNotFoundError(f"No *_prediction.csv in {final_features_dir}")
    df = pd.read_csv(csvs[0])

    thresh_cols = ["threshold"]
    score_cols  = ["risk_score"]

    threshold = None
    for c in thresh_cols:
        if c in df.columns:
            try:
                threshold = float(df[c].iloc[0])
                break
            except Exception:
                pass
    if threshold is None:
        raise ValueError(f"Could not find a numeric threshold column in {csvs[0]}")

    risk_score = None
    for c in score_cols:
        if c in df.columns:
            try:
                risk_score = float(df[c].iloc[0])
                break
            except Exception:
                pass
    if risk_score is None:
        # fallback: pick a numeric column that is not the threshold
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cand = [c for c in num_cols if c not in thresh_cols]
        if not cand:
            raise ValueError(f"No numeric risk score column in {csvs[0]}")
        risk_score = float(df[cand[0]].iloc[0])

    return threshold, risk_score

def posneg_label(score: float, cutoff: float):
    return ("POSITIVE", "+") if score >= cutoff else ("NEGATIVE", "-")



# ==============================
# DRAWING PRIMITIVES (DYNAMIC)
# ==============================
def draw_plus_minus_icon(c: canvas.Canvas, cx, cy, r, label_symbol: str):
    # [DYNAMIC] Conclusion icon (+ red or - green)
    fill = POS_FILL if label_symbol == "+" else NEG_FILL
    c.setFillColor(fill)
    c.circle(cx, cy, r, fill=1, stroke=0)
    c.setStrokeColor(colors.white)
    c.setLineWidth(6)
    # symbol strokes
    c.line(cx - r*0.65, cy, cx + r*0.65, cy)         # horizontal
    if label_symbol == "+":                           # vertical only for plus
        c.line(cx, cy - r*0.65, cx, cy + r*0.65)

def draw_conclusion_text(c: canvas.Canvas, x, y, w, h, posneg: str):
    # Text is now in the PDF template - no dynamic text needed
    # The template already contains the appropriate text for POSITIVE/NEGATIVE
    pass

def draw_treatment_consideration_line(c: canvas.Canvas, x, y, w, h, posneg: str):
    # Text is now in the PDF template - no dynamic text needed
    # The template already contains the "ON AVERAGE..." text
    pass

def draw_vertical_posneg_bar_with_pointer(c: canvas.Canvas, x, y, w, h, score: float, cutoff: float):
    """
    Draws a vertical NEG→POS bar where the split occurs at `cutoff` on a 0..2 scale.
    - 0.0 .. cutoff                => NEGATIVE region
    - cutoff+0.01 .. 2.0           => POSITIVE region
    Pointer shows `score`. If score > 2.0 the pointer is placed at the top of the positive bar
    but the numeric value is shown as-is (not clipped).

    Note: cutoff is interpreted on the 0..2 scale. If cutoff is outside [0,2] it will be clamped.
    """
    # clamp cutoff to [0,2]
    cutoff_clamped = float(max(0.0, min(2.0, cutoff)))
    s_raw = float(score)

    # compute split position in the bar based on cutoff relative to 2.0
    split_ratio = cutoff_clamped / 2.0  # 0..1
    split_y = y + split_ratio * h

    # draw NEGATIVE region (from y up to split)
    c.setFillColor(NEG_FILL)
    c.rect(x, y, w, max(0, split_y - y), fill=1, stroke=0)

    # draw POSITIVE region (from split to top)
    c.setFillColor(POS_FILL)
    c.rect(x, split_y, w, max(0, (y + h) - split_y), fill=1, stroke=0)

    # thin threshold tick line
    c.setStrokeColor(colors.white)
    c.setLineWidth(0.8)
    c.line(x - 2, split_y, x + w + 2, split_y)

    # labels inside the two regions
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 6.5)
    pos_region_height = (y + h) - split_y
    neg_region_height = split_y - y
    pos_mid_y = split_y + max(6, pos_region_height / 2.0) - 4
    neg_mid_y = y + max(6, neg_region_height / 2.0) - 4
    # guard: if a region is essentially zero-height, push label slightly inward
    if pos_region_height < 8:
        pos_mid_y = split_y + 6
    if neg_region_height < 8:
        neg_mid_y = y + 6
    c.drawCentredString(x + w / 2.0, pos_mid_y, "POSITIVE")
    c.drawCentredString(x + w / 2.0, neg_mid_y, "NEGATIVE")

    # Determine pointer Y:
    # Negative mapping: s in [0, cutoff_clamped] -> y..split_y
    # Positive mapping: s in [cutoff_clamped+0.01, 2.0] -> split_y..(y+h)
    # If s in (cutoff_clamped, cutoff_clamped+0.01] -> place at split_y (ambiguous zone)
    # If s > 2.0 -> place at top
    pad = 2.0
    # handle special cases where cutoff_clamped == 0 or == 2 to avoid div by zero
    s = s_raw
    if s <= cutoff_clamped:
        # map [0, cutoff_clamped] -> [y+pad, split_y-pad] if cutoff_clamped > 0 else bottom
        if cutoff_clamped <= 0:
            py = y + pad
        else:
            frac = (s - 0.0) / (cutoff_clamped - 0.0) if cutoff_clamped > 0 else 0.0
            py = y + pad + frac * (max(0, split_y - y) - 2*pad)
    else:
        # s > cutoff_clamped
        positive_min = cutoff_clamped + 0.01
        positive_max = 2.0
        if s > positive_max:
            # place pointer at top of bar
            py = y + h - pad
        elif s <= positive_min:
            # ambiguous tiny zone -> place at split line just above it
            py = split_y + pad
        else:
            # map [positive_min, positive_max] -> [split_y+pad, y+h-pad]
            denom = (positive_max - positive_min)
            if denom <= 0:
                py = split_y + pad
            else:
                frac = (s - positive_min) / denom
                py = split_y + pad + frac * (max(0, (y + h) - split_y) - 2*pad)

    # pointer color by category: positive only if s >= cutoff_clamped + 0.01
    ptr_is_positive = (s >= (cutoff_clamped + 0.01))
    ptr_color = POS_FILL if ptr_is_positive else NEG_FILL
    c.setFillColor(ptr_color); c.setStrokeColor(ptr_color)

    # pointer triangle to the right of bar
    arrow_w, arrow_h = 8, 10
    ax_left = x + w + 4
    ax_right = ax_left + arrow_w
    ay_mid = py
    p = c.beginPath()
    p.moveTo(ax_left, ay_mid - arrow_h / 2)
    p.lineTo(ax_right, ay_mid)
    p.lineTo(ax_left, ay_mid + arrow_h / 2)
    p.close()
    c.drawPath(p, stroke=0, fill=1)

    # numeric value next to pointer: show original score value (not clipped) with 2 decimals
    c.setFont("Helvetica-Bold", 10)
    # If it's >2, still print as-is (e.g., "2.35")
    display_val = f"{s_raw:.2f}"
    c.drawString(ax_right + 3, ay_mid - 3, display_val)


def draw_biopsy_overlay(c: canvas.Canvas, qc_dir: Path, patient_id: str, frame):
    x, y, w, h = frame
    img = qc_dir / f"{patient_id}_tissue_overlay.png"
    if not img.exists():
        return

    try:
        im = Image.open(img)

        # Fix EXIF rotation
        im = ImageOps.exif_transpose(im)

        # Rotate image to horizontal orientation if it's taller than wide
        # This ensures the image fits better in the horizontal frame
        if im.height > im.width:
            im = im.rotate(90, expand=True)

        # Use high-quality resampling for better output
        # Calculate target size to fit frame while maintaining aspect ratio
        iw, ih = im.size
        scale = min(w / iw, h / ih) * 0.9  # 90% of frame for some padding
        target_w = int(iw * scale)
        target_h = int(ih * scale)

        # High-quality resize
        im = im.resize((target_w, target_h), Image.Resampling.LANCZOS)

        _draw_pil_keep_ar(c, im, x, y, w, h)

    except Exception:
        pass



def draw_spatil_grid(c: canvas.Canvas, viz_dir: Path):
    # [DYNAMIC] 2x2 grid of spatil visualizations
    imgs = sorted(viz_dir.glob("*.png"))[:4]
    for i, p in enumerate(imgs):
        row, col = divmod(i, 2)
        xx = GRID_X + col*(CELL_W + CELL_GAP)
        yy = GRID_Y - row*(CELL_H + CELL_GAP)
        draw_image_keep_ar(c, p, xx, yy, CELL_W, CELL_H)


# ==============================
# COMPOSER (PER PATIENT)
# ==============================
def make_report_for_patient(pdir: Path, page1_template_pos: Path, page1_template_neg: Path, page2_template: Path, out_root: Path):
    patient_id = pdir.name
    ff_dir = pdir / "final_features"
    qc_dir = pdir / "qc"
    viz_dir = pdir / "spatil_visualizations" / patient_id

    threshold, risk_score = read_threshold_and_score(ff_dir)

    # decide POS/NEG by comparing risk_score to threshold from CSV
    posneg, symbol = posneg_label(risk_score, threshold)

    # choose Page 1 background by outcome
    if posneg == "NEGATIVE" and page1_template_neg and page1_template_neg.is_file():
        page1_bg = rasterize_page(page1_template_neg, 0, dpi=300)
    else:
        page1_bg = rasterize_page(page1_template_pos, 0, dpi=300)

    # page backgrounds
    # page1_bg = rasterize_page(page1_template_pos, 0, dpi=300)
    page2_bg = rasterize_page(page2_template, 1, dpi=300)

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    # ---------- PAGE 1 BACKGROUND ----------
    c.drawImage(page1_bg, 0, 0, width=PAGE_W, height=PAGE_H, preserveAspectRatio=False, mask='auto')

    # [DYNAMIC] Conclusion icon (+/-)
    draw_plus_minus_icon(c, CONC_ICON_CENTER[0], CONC_ICON_CENTER[1], CONC_ICON_RADIUS, symbol)

    # [DYNAMIC] Conclusion text block
    draw_conclusion_text(c, *CONC_TEXT_FRAME, posneg=posneg)

    # [DYNAMIC] Treatment considerations line
    draw_treatment_consideration_line(c, *TREAT_TEXT_FRAME, posneg=posneg)

    # [DYNAMIC] Interpretation & QC — Part 1: bar + pointer + value
    draw_vertical_posneg_bar_with_pointer(c, BAR_X, BAR_Y, BAR_W, BAR_H, score=risk_score, cutoff=threshold)

    # [DYNAMIC] Interpretation & QC — Part 2: biopsy overlay
    draw_biopsy_overlay(c, qc_dir, patient_id, BIOPSY_IMG_FRAME)

    # [DYNAMIC] Interpretation & QC — Part 3: 4-image grid from patient viz
    if viz_dir.exists():
        draw_spatil_grid(c, viz_dir)

    c.showPage()

    # ---------- PAGE 2 BACKGROUND ----------
    c.drawImage(page2_bg, 0, 0, width=PAGE_W, height=PAGE_H, preserveAspectRatio=False, mask='auto')
    c.showPage()
    c.save()

    # write outputs
    out_dir = pdir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf_name = f"APIC_report_{patient_id}.pdf"

    # patient-specific folder: results/<patient>/report/
    patient_report_dir = pdir / "report"
    patient_report_dir.mkdir(parents=True, exist_ok=True)
    patient_pdf = patient_report_dir / out_pdf_name
    with open(patient_pdf, "wb") as f:
        f.write(buf.getvalue())

    # centralized folder: results/reports/  (from --out-dir)
    if out_root:
        out_root.mkdir(parents=True, exist_ok=True)
        root_pdf = out_root / out_pdf_name
        with open(root_pdf, "wb") as f:
            f.write(buf.getvalue())
        print(f"[OK] {patient_id} -> {patient_pdf} and {root_pdf}")
    else:
        print(f"[OK] {patient_id} -> {patient_pdf}")


def find_patients(results_root: Path):
    pts = []
    for p in results_root.iterdir():
        if p.is_dir() and (p/"final_features").is_dir() and list((p/"final_features").glob("*_prediction.csv")):
            pts.append(p)
    return sorted(pts)

def main():
    ap = argparse.ArgumentParser(description="APIC report: Page1 from empty template + dynamic inserts; Page2 from styled template.")
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--page1-template-pos", required=True, help="Reference PDF 1 (empty page 1)")
    ap.add_argument("--page1-template-neg", default=None, help="Optional alternate Page 1 template for APIC NEGATIVE")
    ap.add_argument("--page2-template", required=True, help="Reference PDF 2 (styled page 2)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    results_root = Path(args.results_root).resolve()
    page1_template_pos = Path(args.page1_template_pos).resolve()
    page1_template_neg = Path(args.page1_template_neg).resolve() if args.page1_template_neg else None
    page2_template = Path(args.page2_template).resolve()
    out_root = Path(args.out_dir).resolve() if args.out_dir else None

    if not results_root.is_dir(): print("results-root not found", file=sys.stderr); sys.exit(1)
    if not page1_template_pos.is_file(): print("page1-template-positive not found", file=sys.stderr); sys.exit(1)
    if not page1_template_neg.is_file(): print("page1-template-negative not found", file=sys.stderr); sys.exit(1)
    if not page2_template.is_file(): print("page2-template not found", file=sys.stderr); sys.exit(1)

    for pdir in find_patients(results_root):
        try:
            make_report_for_patient(pdir, page1_template_pos, page1_template_neg, page2_template, out_root)
        except Exception as e:
            print(f"[FAIL] {pdir.name}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()


# Example usage:
# python build_apic_report_final.py \
#   --results-root /home/vputcha/emory/APIC-container-1/docker_container/data/results \
#   --page1-template /home/vputcha/emory/APIC-container-1/HUG-REPORT-EMPTY.pdf \
#   --page2-template /mnt/d/vputcha_projects/prostate_cancer/APIC/APIC-Container/HUG_report.pdf \
#   --out-dir /home/vputcha/emory/APIC-container-1/docker_container/data/results/reports \

# example usage:
# python build_apic_report_final.py \
#   --results-root /home/vputcha/emory/APIC-container-1/docker_container/data/results/PT3 \
#   --page1-template-pos /home/vputcha/emory/APIC-container-1/docker_container/data/HUG-REPORT-EMPTY-POSITIVE.pdf \
#   --page1-template-neg /home/vputcha/emory/APIC-container-1/docker_container/data/HUG-REPORT-EMPTY-NEGATIVE.pdf \
#   --page2-template /home/vputcha/emory/APIC-container-1/docker_container/data/HUG_report.pdf \
#   --out-dir /home/vputcha/emory/APIC-container-1/test_data/results/reports