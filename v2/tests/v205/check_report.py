"""Check the v2.0.5 APIC patient reports (MAP style, localized and mHSPC) against their patient.json and the
published constants of the two APIC papers, and the --report v1 output against v2.0.3.

    python3 check_report.py WORKDIR IMAGE_DIGEST

Every number in the PDF text must come from patient.json, from the fixed published figures of its setting, or be
an axis, page, slide or count number. The references are checked as exact text and removed before the number check.
Exit 1 on any failure.
"""
import hashlib
import json
import os
import re
import subprocess
import sys

W, DIG = sys.argv[1], sys.argv[2]
REF1 = ("Medina S, Tokuyama N, et al. A Computational Pathology Model to Predict Docetaxel Benefit in Localized "
        "High-Risk and Metastatic Prostate Cancer. Clin Cancer Res 2026;32(3):645-653. doi:10.1158/1078-0432.CCR-25-3327")
REF2 = ("Medina S, Tokuyama N, et al. AI-Based Pathology Classifier Predicts Sensitivity to Enzalutamide in Metastatic "
        "Hormone-Sensitive Prostate Cancer: A Biomarker Analysis of the ENZAMET Trial. Clin Cancer Res "
        "2026;32(20):4742-4753. doi:10.1158/1078-0432.CCR-26-0871")
SET = {
    "localized": {
        "label": "Localized",
        "head": {"APIC-Negative": "Favorable prognosis with radiotherapy and ADT alone",
                 "APIC-Positive": "May benefit from treatment intensification"},
        "body": {"APIC-Negative": "In NRG/RTOG 0521 (high-risk localized prostate cancer), APIC-negative patients had a "
                                  "10-year overall survival of 79.1% (95% CI 66.9 to 91.4) with radiotherapy and ADT "
                                  "alone. Adding docetaxel: hazard ratio 1.17 (95% CI 0.59 to 2.30).",
                 "APIC-Positive": "In NRG/RTOG 0521 (high-risk localized prostate cancer), adding docetaxel to "
                                  "radiotherapy and ADT improved overall survival in APIC-positive patients: hazard "
                                  "ratio 0.49 (95% CI 0.26 to 0.92). 10-year overall survival: 74.4% with docetaxel, "
                                  "52.7% with radiotherapy and ADT alone."},
        "os_title": "Overall survival at 10 years", "os_note": "Radiotherapy and ADT alone, NRG/RTOG 0521.",
        "cards": ["79.1%", "95% CI 66.9–91.4", "52.7%", "95% CI 35.7–69.8"],
        "use": "Research use only. High-risk localized prostate cancer.",
        "refs": [REF1], "absent_refs": [REF2], "rp_note": True,
        "nums": {"79.1", "66.9", "91.4", "1.17", "0.59", "2.30", "0.49", "0.26", "0.92", "74.4", "52.7", "35.7",
                 "69.8", "10", "0521"},
        "banned": ("enzalutamide", "ENZAMET (mHSPC), enzalutamide", "CHAARTED (mHSPC), adding"),
    },
    "mhspc": {
        "label": "mHSPC",
        "head": {"APIC-Negative": "More likely to benefit from enzalutamide",
                 "APIC-Positive": "More likely to benefit from docetaxel"},
        "body": {"APIC-Negative": "In ENZAMET (mHSPC), enzalutamide improved overall survival in APIC-negative patients "
                                  "compared with a nonsteroidal antiandrogen: hazard ratio 0.50 (95% CI 0.35 to 0.73). "
                                  "60-month overall survival: 76.5% with enzalutamide, 58.3% with a nonsteroidal "
                                  "antiandrogen. In CHAARTED (mHSPC), adding docetaxel to ADT: hazard ratio 1.31 (95% "
                                  "CI 0.71 to 2.44).",
                 "APIC-Positive": "In CHAARTED (mHSPC), adding docetaxel to ADT improved overall survival in "
                                  "APIC-positive patients: hazard ratio 0.52 (95% CI 0.32 to 0.85). 5-year overall "
                                  "survival: 40.2% with docetaxel, 15.9% with ADT alone. In ENZAMET (mHSPC), "
                                  "enzalutamide compared with a nonsteroidal antiandrogen: hazard ratio 1.04 (95% CI "
                                  "0.67 to 1.62)."},
        "os_title": "Overall survival at 5 years", "os_note": "ADT alone, CHAARTED.",
        "cards": ["55.6%", "95% CI 40.9–75.6", "15.9%", "95% CI 6.1–41.6"],
        "use": "Research use only. Metastatic hormone-sensitive prostate cancer.",
        "refs": [REF1, REF2], "absent_refs": [], "rp_note": False,
        "nums": {"0.52", "0.32", "0.85", "40.2", "15.9", "1.04", "0.67", "1.62", "0.50", "0.35", "0.73", "76.5",
                 "58.3", "1.31", "0.71", "2.44", "55.6", "40.9", "75.6", "6.1", "41.6", "60", "5", "0521"},
        "banned": ("radiotherapy and ADT alone", "NRG/RTOG 0521 (high-risk localized prostate cancer)"),
    },
}
COMMON_NUMS = {"95", "20", "6", "0", "2", "1"}
NOTE_X = "Scan magnification outside the 20x training range."
NOTE_RP = "Prostatectomy specimen: APIC validation on prostatectomies comes from mHSPC (ENZAMET)."
FIX = ("neg_bx_1", "pos_bx_3", "neg_rp_3", "pos_rp_1", "pos_bx_8", "m_neg_bx_1", "m_pos_bx_3", "m_neg_rp_3", "m_pos_rp_1")
fails = []


def ok(cond, msg):
    if not cond:
        fails.append(msg)


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True).stdout


def f3(v):
    return f"{v:.3f}"


rows = []
for fx in FIX:
    d = os.path.join(W, fx)
    jb = open(os.path.join(d, "patient.json"), "rb").read()
    r = json.loads(jb)
    S = SET[r["setting"]]
    pdf = os.path.join(d, "report.pdf")
    n0 = len(fails)
    info = sh("pdfinfo", pdf)
    ok(re.search(r"Pages:\s+2\b", info), f"{fx}: 2 pages")
    ok("595.276 x 841.89" in info, f"{fx}: A4")
    txt = sh("pdftotext", "-layout", pdf, "-")
    flat = re.sub(r"\s+", " ", sh("pdftotext", pdf, "-"))      # reading order: columns stay apart
    sha = hashlib.sha256(jb).hexdigest()
    rid = f"APIC-{r['end_utc'][:10].replace('-', '')}-{sha[:8].upper()}"
    ok(r["report"] == "v2" and r["end_utc"], f"{fx}: patient.json report v2 and end_utc")
    ok(txt.count(rid) >= 4, f"{fx}: report ID on letterhead, header and footers ({txt.count(rid)})")
    st, other = r["apic_status"], {"APIC-Negative": "APIC-Positive", "APIC-Positive": "APIC-Negative"}[r["apic_status"]]
    ok(S["head"][st] in flat, f"{fx}: headline")
    ok(S["body"][st] in flat, f"{fx}: interpretation body")
    for k in SET:                                 # no headline or body of the other group or the other setting
        for g in SET[k]["head"]:
            if (k, g) != (r["setting"], st):
                ok(SET[k]["head"][g] not in flat and SET[k]["body"][g] not in flat, f"{fx}: foreign text {k} {g}")
    ok(("APIC-POSITIVE" if st == "APIC-Positive" else "APIC-NEGATIVE") in txt, f"{fx}: result chip")
    ok(f3(r["risk_score"]) in txt and "threshold 0.950" in flat and "Threshold 0.950" in flat, f"{fx}: risk and threshold")
    ok(S["os_title"] in flat and S["os_note"] in flat and all(c in flat for c in S["cards"]), f"{fx}: outcome cards")
    ok(S["use"] in flat, f"{fx}: intended use")
    ok(f"Setting {S['label']}" in flat or S["label"] in txt, f"{fx}: setting field")
    ok((NOTE_X in flat) == bool(r["extrapolating"]), f"{fx}: magnification note iff extrapolating")
    ok((NOTE_RP in flat) == (S["rp_note"] and r["specimen"] == "prostatectomy"), f"{fx}: prostatectomy note rule")
    ok(("Prostatectomy" if r["specimen"] == "prostatectomy" else "Biopsy") in txt, f"{fx}: specimen field")
    ok("Sample analyzed" in txt and "Core-needle" not in txt, f"{fx}: 'Sample analyzed', 'Core-needle' absent")
    for i, ref in enumerate(S["refs"], 1):
        ok(f"{i}. {ref}" in flat, f"{fx}: reference {i}")
    for ref in S["absent_refs"]:
        ok(ref not in flat, f"{fx}: reference outside this setting")
    used = sum(s["status"] == "scored" for s in r["slides"])
    ok(f"{used} of {r['n_slides']}" in flat, f"{fx}: slides analyzed")
    hexes = re.findall(r"\b[0-9a-f]{64}\b", txt)
    want = {DIG.split(":")[1], r["model_sha256"], sha, *[s["sha256"] for s in r["slides"]]}
    ok(set(hexes) == want, f"{fx}: traceability hashes")
    low = txt.lower()
    for w in ("not", "no", "never", "without", "each"):
        ok(not re.search(rf"\b{w}\b", low), f"{fx}: banned word '{w}'")
    ok("—" not in txt and "unavailable" not in low, f"{fx}: no em dash, every thumbnail drawn")
    for bad in ("triplet", "doublet", "ARPI", *S["banned"]):
        ok(bad.lower() not in low, f"{fx}: text outside this setting '{bad}'")
    # every number has a source
    rest = flat
    for ref in S["refs"]:
        rest = rest.replace(ref, " ")
    for s in sorted({rid, r["patient"], *[s["file"] for s in r["slides"]], "NRG/RTOG 0521", r["end_utc"][:10],
                     "2026-11-10", r["version"], "SHA-256", "H&E", *hexes}, key=len, reverse=True):
        rest = rest.replace(s, " ")
    rest = re.sub(r"\S*…\S*", " ", rest)
    allowed = set(S["nums"]) | COMMON_NUMS | {str(i) for i in range(1, r["n_slides"] + 1)} | {str(r["n_slides"]), str(used)}
    allowed |= {"0.950", f3(r["risk_score"])}
    for s in r["slides"]:
        if s["status"] == "scored":
            allowed |= {f3(s["risk_score"]), str(s["n_tiles_tissue"]), f"{int(s['n_nuclei']):,}",
                        f"{s['objective_power']:.0f}"}
    nums = re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?", rest)
    bad = sorted({x.rstrip(",.") for x in nums if x.rstrip(",.") not in allowed})
    ok(not bad, f"{fx}: numbers with no source {bad}")
    res = "PASS" if len(fails) == n0 else "FAIL"
    rows.append((fx, S["label"], st, r["specimen"], f"{used} of {r['n_slides']}", r["extrapolating"], len(nums), res))
    print(f"{fx}: {S['label']}, {st}, {r['specimen']}, slides {used} of {r['n_slides']}, extrapolating "
          f"{r['extrapolating']}, numbers checked {len(nums)}, {res}")

# --report v1: the v1 report, same as v2.0.3 for the same input
a, b = os.path.join(W, "eq_mhspc", "v205"), os.path.join(W, "eq_mhspc", "v203")
ja, jb_ = json.load(open(os.path.join(a, "patient.json"))), json.load(open(os.path.join(b, "patient.json")))
ok(ja["setting"] == "mhspc" and ja["report"] == "v1", "v1: patient.json fields")
drop = {"setting", "specimen", "report", "end_utc", "seconds", "version"}
ok({k: v for k, v in ja.items() if k not in drop} == {k: v for k, v in jb_.items() if k not in drop},
   "v1: patient.json equals v2.0.3 apart from the new fields")
ta, tb = sh("pdftotext", "-layout", os.path.join(a, "report.pdf"), "-"), sh("pdftotext", "-layout", os.path.join(b, "report.pdf"), "-")
ok(ta == tb and len(ta) > 200, "v1: report text equals v2.0.3")
for p in (a, b):
    subprocess.run(["pdftoppm", "-r", "72", "-png", os.path.join(p, "report.pdf"), os.path.join(p, "px")], check=True)
pa = sorted(f for f in os.listdir(a) if f.startswith("px")); pb = sorted(f for f in os.listdir(b) if f.startswith("px"))
same = len(pa) == len(pb) == 2 and all(open(os.path.join(a, x), "rb").read() == open(os.path.join(b, y), "rb").read()
                                        for x, y in zip(pa, pb))
ok(same, "v1: rendered pages equal v2.0.3 pixel for pixel")
print(f"eq_mhspc (--report v1): text equal {ta == tb}, pages equal {same}")
ref = open(os.path.join(W, "refused.txt")).read()
ok("refused for --setting localized" in ref and not ref.strip().endswith("exit 0"), "localized with --report v1 refused")
print("localized --report v1: refused" if "refused" in ref else "localized --report v1: ACCEPTED")

print("\n".join(["FAIL " + f for f in fails]) or "ALL PASS")
sys.exit(1 if fails else 0)
