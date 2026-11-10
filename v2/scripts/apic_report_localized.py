"""APIC patient report for localized (non-metastatic) prostate cancer, from one patient.json.

The mHSPC setting keeps the v1 report (apic_report.py), which carries the docetaxel and ARPI treatment
considerations. Treatment selection text belongs to mHSPC only, so this report shows the APIC result, the
prognosis on radiotherapy and ADT alone, and the docetaxel result of NRG/RTOG 0521 in a toned-down form.
Every clinical number on it is a published NRG/RTOG 0521 figure (Medina et al., the CHAARTED and NRG/RTOG 0521
APIC manuscript) and appears below as a constant. The look follows the MAP patient report.

    python3 apic_report_localized.py --patient-json /out/patient.json --out /out/report.pdf \\
        [--slides-dir /in] [--image-digest sha256:...] [--report-date YYYY-MM-DD]

Page 1: result, interpretation, overall survival at 10 years, sign-off line.
Page 2: slides (thumbnail with the HistoQC tissue outline), test information and traceability.
Report ID: APIC-<end_utc date>-<first 8 hex of the patient.json sha256, upper case>.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import sys
from io import BytesIO
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

_HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.environ.get("APIC_FONT_DIR", os.path.join(_HERE, "..", "fonts"))
rl_config.invariant = 1
for _name, _f in [("Sans", "IBMPlexSans-Regular"), ("Sans-B", "IBMPlexSans-SemiBold"), ("Head", "IBMPlexSans-Medium"),
                  ("Mono", "NotoSansMono-Regular")]:
    pdfmetrics.registerFont(TTFont(_name, os.path.join(FONT_DIR, f"{_f}.ttf")))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-B", italic="Sans", boldItalic="Sans-B")

# ---- published NRG/RTOG 0521 figures (high-risk localized; the only clinical numbers this report prints) ----
RTOG = {
    "neg": {"os10_rtadt": (79.1, 66.9, 91.4), "hr_doce": (1.17, 0.59, 2.30)},
    "pos": {"os10_rtadt": (52.7, 35.7, 69.8), "os10_doce": 74.4, "hr_doce": (0.49, 0.26, 0.92)},
}
HEADLINE = {"neg": "Favorable prognosis with radiotherapy and ADT alone",
            "pos": "May benefit from treatment intensification"}
BODY = {
    "neg": ("In NRG/RTOG 0521 (high-risk localized prostate cancer), APIC-negative patients had a 10-year overall "
            "survival of 79.1% (95% CI 66.9 to 91.4) with radiotherapy and ADT alone. Adding docetaxel: hazard "
            "ratio 1.17 (95% CI 0.59 to 2.30)."),
    "pos": ("In NRG/RTOG 0521 (high-risk localized prostate cancer), adding docetaxel to radiotherapy and ADT improved "
            "overall survival in APIC-positive patients: hazard ratio 0.49 (95% CI 0.26 to 0.92). 10-year overall "
            "survival: 74.4% with docetaxel, 52.7% with radiotherapy and ADT alone."),
}
NOTE_EXTRAP = "Scan magnification outside the 20x training range."
NOTE_RP = "Prostatectomy specimen: APIC validation on prostatectomies comes from mHSPC (ENZAMET)."
METHOD = ["Input: H&amp;E whole-slide images, all diagnostic slides of one patient.",
          "APIC measures nuclear shape diversity in the tumor and the spatial arrangement of tumor and immune cells.",
          "The 6 features are averaged over the scored slides and scored with the locked Cox model.",
          "APIC-positive above the threshold {thr}."]
USE = ["Research use only.", "High-risk localized prostate cancer."]
VALIDATION = ("Validated on H&amp;E core-needle biopsies in CHAARTED (mHSPC) and NRG/RTOG 0521 (high-risk localized), "
              "and on prostatectomy specimens in ENZAMET (mHSPC).")
CITATION = ("Medina S, Tokuyama N, et al. A computational pathology model to predict docetaxel benefit in localized "
            "high-risk and metastatic prostate cancer.")
MINUS = "−"

C = colors.HexColor
INK, HAIR, WHITE = C("#1B1F24"), C("#C5CBD2"), colors.white
TINT, FRAME, SHADE = C("#E9EDF2"), C("#D5DBE6"), C("#EEF1F6")
DARK = C("#3A5068")                                  # slate blue, as the MAP report
ART = C("#A82A2A")
OUTLINE = (15, 110, 110)                             # tissue outline color of the MAP report
W, H = A4
X0 = 40
CW = W - 2 * X0
TOP, BOT = H - 34, 50
PAD, GAP, BAND = 12, 14, 24
GX, GY, CP, CAPH = 10, 10, 8, 26                # CAPH: two caption lines, file, then the result


def style(size, font="Sans", color=INK, lead=1.38, **k):
    return ParagraphStyle(f"s{size}{font}{lead}", fontName=font, fontSize=size, leading=size * lead, textColor=color, **k)


def num(v, d):
    s = f"{v:.{d}f}"
    if re.fullmatch(r"-0\.?0*", s):
        s = s[1:]
    return s.replace("-", MINUS)


def width(s, font, size):
    return pdfmetrics.stringWidth(s, font, size)


def text(c, x, y, s, size=8, font="Sans", color=INK, anchor="l"):
    x -= {"l": 0, "r": width(s, font, size), "c": width(s, font, size) / 2}[anchor]
    c.setFillColor(color)
    c.setFont(font, size)
    c.drawString(x, y, s)


def lab(c, x, y, s, color=None, size=6.5, anchor="l"):
    """Spaced capitals for labels, in slate blue."""
    s = s.upper()
    color = DARK if color is None else color
    cs = size * 0.085
    w = pdfmetrics.stringWidth(s, "Sans-B", size) + cs * (len(s) - 1)
    x -= {"l": 0, "r": w, "c": w / 2}[anchor]
    t = c.beginText(x, y)
    t.setFont("Sans-B", size)
    t.setCharSpace(cs)
    t.setFillColor(color)
    t.textOut(s)
    t.setCharSpace(0)
    c.drawText(t)
    return w


def section(c, y, title, right=None):
    text(c, X0, y - 15, title, 13, "Head", DARK)
    if right:
        text(c, X0 + CW, y - 14, right, 7.6, "Sans", INK, "r")
    return y - BAND


def para(c, s, x, ytop, w, st):
    p = Paragraph(s, st)
    _, h = p.wrap(w, 1000)
    p.drawOn(c, x, ytop - h)
    return h


def para_h(s, w, st):
    return Paragraph(s, st).wrap(w, 1000)[1]


def fit(s, font, size, maxw):
    if pdfmetrics.stringWidth(s, font, size) <= maxw:
        return s
    k = len(s)
    while k > 4 and pdfmetrics.stringWidth(s[:k // 2] + "…" + s[len(s) - k // 2:], font, size) > maxw:
        k -= 1
    return s[:k // 2] + "…" + s[len(s) - k // 2:]


def alpha(col, a):
    return colors.Color(col.red, col.green, col.blue, a)


def group(rec):
    return "pos" if rec.get("apic_status") == "APIC-Positive" else "neg"


def chip_word(g):
    return "APIC-POSITIVE" if g == "pos" else "APIC-NEGATIVE"


def report_id(end_utc, sha):
    return f"APIC-{end_utc[:10].replace('-', '')}-{sha[:8].upper()}"


# ---------------------------------------------------------------- page 1

def letterhead(c, y, rep):
    text(c, X0, y - 25, "APIC", 27, "Head", DARK)
    text(c, X0, y - 41, "AI-based pathology classifier", 10.5, "Sans", INK)
    lab(c, X0 + CW, y - 16, "Patient report", DARK, 8, "r")
    text(c, X0 + CW, y - 34, rep, 9.5, "Mono", INK, "r")
    lab(c, X0 + CW - width(rep, "Mono", 9.5) - 8, y - 33.5, "Report ID", None, 6.2, "r")
    c.setStrokeColor(DARK)
    c.setLineWidth(0.6)
    c.line(X0, y - 51, X0 + CW, y - 51)
    return y - 53


def info(c, y, fields):
    cols, ch, pad = math.ceil(len(fields) / 2), 29, 7
    cw = (CW - 24) / cols
    c.setFillColor(TINT)
    c.rect(X0, y - 2 * ch - 2 * pad, CW, 2 * ch + 2 * pad, stroke=0, fill=1)
    for i, (k, v, bold) in enumerate(fields):
        x, top = X0 + 12 + (i % cols) * cw, y - pad - (i // cols) * ch
        lab(c, x, top - 10, k, None, 6)
        f = "Sans-B" if bold else "Sans"
        text(c, x, top - 23, fit(v, f, 9.5, cw - 12), 9.5, f)
    return y - 2 * ch - 2 * pad


def chip(c, x, ytop, w, h, g, size=20):
    if g == "pos":
        c.setFillColor(DARK)
        c.rect(x, ytop - h, w, h, stroke=0, fill=1)
        ink = WHITE
    else:
        c.setFillColor(WHITE)
        c.setStrokeColor(DARK)
        c.setLineWidth(1.6)
        c.rect(x + 0.8, ytop - h + 0.8, w - 1.6, h - 1.6, stroke=1, fill=1)
        ink = DARK
    text(c, x + w / 2, ytop - h / 2 - size * 0.36, chip_word(g), size * 0.86, "Sans-B", ink, "c")


def scale(c, x0, x1, ybar, risk, thr, g, top=2.0):
    """Risk score from 0 to 2: APIC-negative up to the threshold, APIC-positive above it, and the patient."""
    X = lambda v: x0 + (x1 - x0) * min(top, max(0.0, v)) / top
    hb = 12
    c.setFillColor(alpha(DARK, 0.10))
    c.rect(X(0), ybar, X(thr) - X(0), hb, stroke=0, fill=1)
    c.setFillColor(alpha(DARK, 0.30))
    c.rect(X(thr), ybar, X(top) - X(thr), hb, stroke=0, fill=1)
    lab(c, (X(0) + X(thr)) / 2, ybar - 13, "APIC-negative", None, 6.2, "c")
    lab(c, (X(thr) + X(top)) / 2, ybar - 13, "APIC-positive", None, 6.2, "c")
    c.setStrokeColor(INK)
    c.setLineWidth(0.8)
    c.setDash(2.5, 2)
    c.line(X(thr), ybar - 4, X(thr), ybar + hb + 4)
    c.setDash()
    text(c, X(0), ybar - 13, "0", 7, anchor="c")
    text(c, X(top), ybar - 13, num(top, 0), 7, anchor="c")
    text(c, X(thr), ybar - 23, f"Threshold {num(thr, 3)}", 6.6, anchor="c")
    px, py = X(risk), ybar + hb / 2
    c.setLineWidth(1.4)
    if g == "pos":
        c.setFillColor(DARK)
        c.setStrokeColor(WHITE)
    else:
        c.setFillColor(WHITE)
        c.setStrokeColor(DARK)
    c.circle(px, py, 6, stroke=1, fill=1)
    if g == "pos":
        c.setStrokeColor(INK)
        c.setLineWidth(0.6)
        c.circle(px, py, 7, stroke=1, fill=0)
    tw = width("Patient", "Sans-B", 7.5)
    text(c, min(max(px - tw / 2, x0), x1 - tw), ybar + hb + 14, "Patient", 7.5, "Sans-B", INK)
    c.setStrokeColor(INK)
    c.setLineWidth(0.5)
    c.line(px, ybar + hb + 1, px, ybar + hb + 11)


def result(c, y, rec):
    g, h = group(rec), 122
    yb = section(c, y, "Result")
    x = X0
    chip(c, x, yb - PAD, 170, 40, g)
    sy = yb - PAD - 40 - 22
    lab(c, x, sy, "Risk score")
    v = num(rec["risk_score"], 3)
    text(c, x, sy - 28, v, 26, "Head", INK)
    text(c, x + width(v, "Head", 26) + 8, sy - 28, f"threshold {num(rec['threshold'], 3)}", 8, "Sans", INK)
    scale(c, x + 250, X0 + CW - 20, yb - h / 2 - 2, rec["risk_score"], rec["threshold"], g)
    return yb - h


def interpretation(c, y, g, notes):
    st_body, st_note = style(8.6, lead=1.45), style(7.8, lead=1.4)
    inner = CW - 2 * 16
    hb = para_h(escape(BODY[g]), inner, st_body)
    hn = sum(para_h(escape(n), inner - 10, st_note) + 2 for n in notes)
    h = 14 + 14 + 8 + hb + (14 + hn if notes else 0) + 12
    yb = section(c, y, "Interpretation")
    c.setFillColor(TINT)
    c.rect(X0, yb - h, CW, h, stroke=0, fill=1)
    c.setFillColor(DARK)
    c.rect(X0, yb - h, 4, h, stroke=0, fill=1)
    x, yy = X0 + 16, yb - 14
    text(c, x, yy - 10, HEADLINE[g], 11.5, "Sans-B", INK)
    yy -= 14 + 8
    yy -= para(c, escape(BODY[g]), x, yy, inner, st_body)
    if notes:
        yy -= 10
        lab(c, x, yy - 4, "Notes", None, 6.2)
        yy -= 10
        for n in notes:
            c.setFillColor(ART)
            c.rect(x, yy - 7.5, 3, 3, stroke=0, fill=1)
            yy -= para(c, escape(n), x + 10, yy, inner - 10, st_note) + 2
    return yb - h


def outcomes(c, y, g):
    h = 86
    yb = section(c, y, "Overall survival at 10 years")
    bw, bg = 150, 10
    bx = X0 + CW - 2 * bw - bg
    para(c, "Radiotherapy and ADT alone, NRG/RTOG 0521.<br/>High-risk localized prostate cancer.",
         X0, yb - PAD + 1, bx - X0 - 16, style(7.8, lead=1.45))
    top, bh = yb - 8, h - 16
    for j, k in enumerate(("neg", "pos")):
        x, me = bx + j * (bw + bg), k == g
        rate, lo, hi = RTOG[k]["os10_rtadt"]
        c.setFillColor(alpha(DARK, 0.13 if me else 0.05))
        c.rect(x, top - bh, bw, bh, stroke=0, fill=1)
        c.setFillColor(DARK)
        c.rect(x, top - bh, 5 if me else 2.5, bh, stroke=0, fill=1)
        w = lab(c, x + 14, top - 14, "APIC-negative" if k == "neg" else "APIC-positive", None, 6.4)
        if me:
            text(c, x + 14 + w + 6, top - 14, "This patient", 6.6, "Sans", INK)
        text(c, x + 14, top - 36, f"{rate:.1f}%", 20, "Head", INK)
        text(c, x + 14, top - bh + 8, f"95% CI {lo:.1f}–{hi:.1f}", 6.8, "Sans", INK)
    return yb - h


def signoff(c, y):
    w = lab(c, X0, y, "Signed out by")
    c.setStrokeColor(INK)
    c.setLineWidth(0.5)
    c.line(X0 + w + 10, y - 2, X0 + 320, y - 2)
    w = lab(c, X0 + 350, y, "Date")
    c.line(X0 + 350 + w + 10, y - 2, X0 + CW, y - 2)


def footer(c, page, pages, rep):
    text(c, X0, 28, f"Research use only   |   Report {rep}", 7)
    text(c, X0 + CW, 28, f"Page {page} of {pages}", 7, anchor="r")


# ---------------------------------------------------------------- page 2

def running(c, patient, rep):
    y = TOP
    text(c, X0, y - 10, "APIC", 12, "Head", DARK)
    text(c, X0 + width("APIC  ", "Head", 12), y - 10, "Patient report", 11, "Sans", INK)
    text(c, X0 + CW, y - 9, f"Patient {patient}   |   Report {rep}", 7.5, anchor="r")
    c.setStrokeColor(DARK)
    c.setLineWidth(1.2)
    c.line(X0, y - 15, X0 + CW, y - 15)
    return y - 15


def thumb_with_outline(slide_path, mask_path, side=1100):
    """Slide thumbnail with the HistoQC tissue outline, cut to the tissue box plus 3 percent. None without a slide."""
    if not slide_path or not os.path.isfile(slide_path):
        return None
    try:
        import openslide
        s = openslide.OpenSlide(slide_path)
        Wd, Hd = s.dimensions
        k = min(side / float(Wd), side / float(Hd), 1.0)
        th = s.get_thumbnail((max(1, int(Wd * k)), max(1, int(Hd * k)))).convert("RGB")
        s.close()
    except Exception:
        return None
    rgb = np.array(th)
    box = (0, 0, th.width, th.height)
    if mask_path and os.path.isfile(mask_path):
        m = (np.array(Image.open(mask_path).convert("L").resize(th.size, Image.NEAREST)) > 0).astype(np.uint8)
        if m.any():
            try:
                import cv2
                cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgb, cnts, -1, OUTLINE, max(2, int(round(min(th.size) / 300.0))))
            except Exception:
                edge = m.astype(bool) & ~(np.roll(m, 1, 0).astype(bool) & np.roll(m, -1, 0).astype(bool)
                                          & np.roll(m, 1, 1).astype(bool) & np.roll(m, -1, 1).astype(bool))
                rgb[edge] = OUTLINE
            ys, xs = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
            x0, x1, y0, y1 = xs[0], xs[-1] + 1, ys[0], ys[-1] + 1
            mg = max(8, round(0.03 * max(x1 - x0, y1 - y0)))
            box = (max(0, x0 - mg), max(0, y0 - mg), min(th.width, x1 + mg), min(th.height, y1 + mg))
    return Image.fromarray(rgb).crop(box)


def ground(img):
    a = np.asarray(img.convert("RGB"))
    e = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    return colors.Color(*(np.median(e, 0) / 255))


def jpeg(img):
    b = BytesIO()
    img.convert("RGB").save(b, "JPEG", quality=88)
    b.seek(0)
    return ImageReader(b)


def grid_layout(aspects, n, gh):
    """Cards per row and the frame size that give the largest smallest image; aspect = height / width."""
    best = None
    for cols in (1, 2, 3):
        rows = math.ceil(n / cols)
        cell = (CW - (cols - 1) * GX) / cols
        fw = cell - 2 * CP
        fh = min((gh - (rows - 1) * GY) / rows - 2 * CP - CAPH, fw * max(aspects))
        fw = min(fw, fh / min(aspects))
        if fh < 30:
            continue
        area = min(min(fw, fh / a) * min(fh, fw * a) for a in aspects)
        if best is None or area > best[0] + 1e-6:
            best = (area, cols, cell, fw, fh)
    return best[1:]


def caption(s):
    if s.get("status") != "scored":
        return [("Failed", "Sans-B", ART)]
    return [(f"Risk score {num(s['risk_score'], 3)}", "Sans", INK), (chip_word(group(s)), "Sans-B", DARK),
            (f"Tiles {s.get('n_tiles_tissue') or 0}", "Sans", INK)]


def slide_card(c, x, ytop, cell, fw, fh, i, s, img):
    h = 2 * CP + fh + CAPH
    fx, fy = x + CP + (cell - 2 * CP - fw) / 2, ytop - CP - fh
    if img is not None:
        c.setFillColor(ground(img))
    else:
        c.setFillColor(SHADE)
    c.setStrokeColor(FRAME)
    c.setLineWidth(0.5)
    c.rect(fx, fy, fw, fh, stroke=1, fill=1)
    if img is not None:
        k = min(fw / img.width, fh / img.height)
        dw, dh = img.width * k, img.height * k
        c.drawImage(jpeg(img), fx + (fw - dw) / 2, fy + (fh - dh) / 2, dw, dh)
    else:
        text(c, fx + fw / 2, fy + fh / 2 - 3, "Thumbnail unavailable", 7.5, anchor="c")
    sep, sz = "   |   ", 7.2
    head = f"Slide {i}"
    room = fw - width(head, "Sans-B", sz) - width(sep, "Sans", sz) - 2
    lines = [[(head, "Sans-B", INK), (sep + fit(s.get("file") or "", "Sans", sz, room), "Sans", INK)],
             [((sep if k else "") + t, f, col) for k, (t, f, col) in enumerate(caption(s))]]
    for row, b in zip(lines, (ytop - CP - fh - 11, ytop - CP - fh - 21)):
        xx = fx
        for t, f, col in row:
            text(c, xx, b, t, sz, f, col)
            xx += width(t, f, sz)
    return h


TCOLS = {"slide": 0, "file": 26, "risk": 250, "group": 262, "tiles": 380, "nuclei": 432, "scan": 470, "status": 480}


def slide_table(c, yb, slides, rh=15):
    x = X0
    heads = [("slide", "Slide", "l"), ("file", "File", "l"), ("risk", "Risk score", "r"), ("group", "Group", "l"),
             ("tiles", "Tiles", "r"), ("nuclei", "Nuclei", "r"), ("scan", "Scan", "r"), ("status", "Status", "l")]
    for k, t, al in heads:
        text(c, x + TCOLS[k], yb - 12, t, 6.8, "Sans-B", DARK, al)
    c.setStrokeColor(DARK)
    c.setLineWidth(0.7)
    c.line(x, yb - 17, X0 + CW, yb - 17)
    ry = yb - 17
    for i, s in enumerate(slides, 1):
        b = ry - rh + 4.5
        ok = s.get("status") == "scored"
        text(c, x + TCOLS["slide"], b, str(i), 7.4)
        text(c, x + TCOLS["file"], b, fit(s.get("file") or "", "Sans", 7.4, TCOLS["risk"] - TCOLS["file"] - 50), 7.4)
        if ok:
            text(c, x + TCOLS["risk"], b, num(s["risk_score"], 3), 7.4, anchor="r")
            text(c, x + TCOLS["group"], b, chip_word(group(s)), 7.4, "Sans-B", DARK)
            text(c, x + TCOLS["tiles"], b, str(s.get("n_tiles_tissue") or 0), 7.4, anchor="r")
            text(c, x + TCOLS["nuclei"], b, f"{int(s.get('n_nuclei') or 0):,}", 7.4, anchor="r")
            if s.get("objective_power"):
                text(c, x + TCOLS["scan"], b, f"{num(s['objective_power'], 0)}x", 7.4, anchor="r")
        text(c, x + TCOLS["status"], b, "Scored" if ok else "Failed", 7.4, "Sans" if ok else "Sans-B", INK if ok else ART)
        ry -= rh
    return 17 + len(slides) * rh + 6


TI = {"key": 84, "mono": 5.7, "lead": 8.6, "st": style(7.6, lead=1.36)}


def ti_blocks(thr):
    """Method and intended use beside the traceability list; validation and citation across the page under them."""
    return ([("Method", [m.format(thr=thr) for m in METHOD]), ("Intended use", USE)],
            [("Validation", [VALIDATION]), ("Citation", [CITATION])])


def _blocks_h(blocks, w):
    return sum(6 + sum(para_h(t, w, TI["st"]) for t in lines) + 14 for _, lines in blocks) - 14


def _blocks(c, blocks, x, yy, w):
    for k, (title, lines) in enumerate(blocks):
        if k:
            yy -= 14
        lab(c, x, yy, title, size=6.2)
        yy -= 6
        for t in lines:
            yy -= para(c, t, x, yy, w, TI["st"])
    return yy


def test_info_geom(blocks, trace):
    top, bottom = blocks
    rw = TI["key"] + 64 * 0.6 * TI["mono"]
    lw = CW - rw - 22
    upper = max(_blocks_h(top, lw), 4 + len(trace) * TI["lead"])
    return rw, lw, upper, 20 + upper + 16 + _blocks_h(bottom, CW) + 12


def test_info(c, y, blocks, trace):
    rw, lw, upper, h = test_info_geom(blocks, trace)
    yb = section(c, y, "Test information")
    _blocks(c, blocks[0], X0, yb - 20, lw)
    _blocks(c, blocks[1], X0, yb - 20 - upper - 16, CW)
    rx, yy = X0 + CW - rw, yb - 20
    lab(c, rx, yy, "Traceability", size=6.2)
    lab(c, X0 + CW, yy, "SHA-256", INK, 6.2, "r")
    yy -= 4
    for k, v in trace:
        yy -= TI["lead"]
        text(c, rx, yy, fit(k, "Sans", 6.6, TI["key"] - 6), 6.6)
        text(c, rx + TI["key"], yy, v, TI["mono"], "Mono", INK)
    return yb - h


# ---------------------------------------------------------------- main

def build(patient_json, out_pdf, slide_paths=None, base=None, image_digest=None, report_date=None):
    """Write the localized report for patient_json. slide_paths maps a slide file name to its path; base is the
    folder that holds <stem>/tissue_mask.png (default: the folder of patient_json)."""
    jb = open(patient_json, "rb").read()
    rec = json.loads(jb)
    if rec.get("risk_score") is None:
        raise ValueError("patient.json has no APIC result")
    sha = hashlib.sha256(jb).hexdigest()
    end = rec.get("end_utc") or datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    rep = report_id(end, sha)
    report_date = report_date or datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    base = base or os.path.dirname(os.path.abspath(patient_json))
    slide_paths = slide_paths or {}
    g = group(rec)
    slides = rec.get("slides") or []
    n, used = len(slides), sum(s.get("status") == "scored" for s in slides)
    thr = num(rec["threshold"], 3)
    notes = ([NOTE_EXTRAP] if rec.get("extrapolating") else []) + \
            ([NOTE_RP] if rec.get("specimen") == "prostatectomy" else [])

    fields = [("Patient ID", str(rec.get("patient") or ""), True),
              ("Specimen", "Prostatectomy" if rec.get("specimen") == "prostatectomy" else "Biopsy", False),
              ("Setting", "Localized", False), ("Slides analyzed", f"{used} of {n}", False),
              ("Processed", end[:10], False), ("Report date", report_date, False)]
    blocks = ti_blocks(thr)
    trace = [("APIC version", str(rec.get("version") or "")),
             ("Container image", image_digest.split(":", 1)[1] if image_digest else ""),
             ("Frozen model", rec.get("model_sha256") or ""),
             *[(s.get("file") or "", s.get("sha256") or "") for s in slides],
             ("Run record", sha)]

    cv = canvas.Canvas(out_pdf, pagesize=A4, invariant=1, initialFontName="Sans", initialFontSize=8)
    cv.setTitle(f"APIC report {rep}")
    cv.setAuthor("APIC")
    cv.setSubject(f"Report {rep}, run record sha256 {sha}")
    cv.setCreator("APIC")
    pages = 2

    y = letterhead(cv, TOP, rep) - 8
    y = info(cv, y, fields) - GAP
    y = result(cv, y, rec) - GAP
    y = interpretation(cv, y, g, notes) - GAP
    y = outcomes(cv, y, g)
    if y < BOT + 38:
        sys.exit("apic: report page 1 overflow")
    signoff(cv, BOT + 22)
    footer(cv, 1, pages, rep)
    cv.showPage()

    y = running(cv, str(rec.get("patient") or ""), rep) - 8
    y = section(cv, y, "Slides", f"{n} slides" if n != 1 else "1 slide")
    if n <= 6:
        text(cv, X0 + CW, y - 6, "Tissue outline", 7, anchor="r")
        lx = X0 + CW - width("Tissue outline", "Sans", 7) - 5
        cv.setStrokeColor(colors.Color(*(v / 255 for v in OUTLINE)))
        cv.setLineWidth(1.6)
        cv.line(lx - 13, y - 3.5, lx, y - 3.5)
        y -= 12
        imgs = [thumb_with_outline(slide_paths.get(s.get("file")), os.path.join(base, s.get("stem") or "", "tissue_mask.png"))
                for s in slides]
        aspects = [(im.height / im.width) if im is not None else 0.75 for im in imgs] or [0.75]
        gh = y - BOT - GAP - BAND - test_info_geom(blocks, trace)[3] - 4
        cols, cell, fw, fh = grid_layout(aspects, max(1, n), gh)
        for row in range(math.ceil(n / cols)):
            hh = 0
            for col in range(cols):
                j = row * cols + col
                if j < n:
                    hh = slide_card(cv, X0 + col * (cell + GX), y, cell, fw, fh, j + 1, slides[j], imgs[j])
            y -= hh + GY
        y -= GAP - GY
    else:
        y -= slide_table(cv, y, slides) + GAP
    test_info(cv, y, blocks, trace)
    footer(cv, 2, pages, rep)
    cv.showPage()
    cv.save()
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patient-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slides-dir", default=None, help="folder holding the slide files named in patient.json")
    ap.add_argument("--image-digest", default=None)
    ap.add_argument("--report-date", default=None)
    a = ap.parse_args()
    if a.image_digest and not re.fullmatch(r"sha256:[0-9a-f]{64}", a.image_digest):
        sys.exit("--image-digest must be sha256: and 64 hex characters")
    if a.report_date and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", a.report_date):
        sys.exit("--report-date must be YYYY-MM-DD")
    rec = json.load(open(a.patient_json))
    paths = {}
    if a.slides_dir:
        for s in rec.get("slides") or []:
            p = os.path.join(a.slides_dir, s.get("file") or "")
            if os.path.isfile(p):
                paths[s["file"]] = p
    print(build(a.patient_json, a.out, paths, image_digest=a.image_digest, report_date=a.report_date))


if __name__ == "__main__":
    main()
