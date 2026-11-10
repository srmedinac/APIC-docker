"""Check the v2.0.4 localized APIC reports against their patient.json and the published NRG/RTOG 0521 constants,
and the mHSPC report against v2.0.3.

    python3 check_report.py WORKDIR IMAGE_DIGEST

Every number in the PDF text must come from patient.json, from the fixed published figures, or be an axis, page,
slide or count number. Exit 1 on any failure.
"""
import hashlib
import json
import os
import re
import subprocess
import sys

W, DIG = sys.argv[1], sys.argv[2]
CONST = {"79.1", "66.9", "91.4", "1.17", "0.59", "2.30", "0.49", "0.26", "0.92", "74.4", "52.7", "35.7", "69.8",
         "10", "95", "20", "6", "0", "2"}
HEAD = {"APIC-Negative": "Favorable prognosis with radiotherapy and ADT alone",
        "APIC-Positive": "May benefit from treatment intensification"}
BODY = {"APIC-Negative": "APIC-negative patients had a 10-year overall survival of 79.1% (95% CI 66.9 to 91.4) with "
                         "radiotherapy and ADT alone. Adding docetaxel: hazard ratio 1.17 (95% CI 0.59 to 2.30).",
        "APIC-Positive": "adding docetaxel to radiotherapy and ADT improved overall survival in APIC-positive patients: "
                         "hazard ratio 0.49 (95% CI 0.26 to 0.92). 10-year overall survival: 74.4% with docetaxel, "
                         "52.7% with radiotherapy and ADT alone."}
NOTE_X = "Scan magnification outside the 20x training range."
NOTE_RP = "Prostatectomy specimen: APIC validation on prostatectomies comes from mHSPC (ENZAMET)."
fails = []


def ok(cond, msg):
    if not cond:
        fails.append(msg)


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True).stdout


def f3(v):
    return f"{v:.3f}"


for fx in ("neg_bx_1", "pos_bx_3", "neg_rp_3", "pos_rp_1", "pos_bx_8"):
    d = os.path.join(W, fx)
    jb = open(os.path.join(d, "patient.json"), "rb").read()
    r = json.loads(jb)
    pdf = os.path.join(d, "report.pdf")
    n0 = len(fails)
    info = sh("pdfinfo", pdf)
    ok(re.search(r"Pages:\s+2\b", info), f"{fx}: 2 pages")
    ok("595.276 x 841.89" in info, f"{fx}: A4")
    txt = sh("pdftotext", "-layout", pdf, "-")
    flat = re.sub(r"\s+", " ", txt)
    sha = hashlib.sha256(jb).hexdigest()
    rid = f"APIC-{r['end_utc'][:10].replace('-', '')}-{sha[:8].upper()}"
    ok(r["setting"] == "localized" and r["report"] == "localized" and r["end_utc"], f"{fx}: patient.json fields")
    ok(txt.count(rid) >= 4, f"{fx}: report ID on letterhead, header and footers ({txt.count(rid)})")
    st = r["apic_status"]
    ok(HEAD[st] in flat and HEAD["APIC-Positive" if st == "APIC-Negative" else "APIC-Negative"] not in flat,
       f"{fx}: headline for {st}")
    ok(BODY[st] in flat, f"{fx}: interpretation text")
    ok(("APIC-POSITIVE" if st == "APIC-Positive" else "APIC-NEGATIVE") in txt, f"{fx}: result chip")
    ok(f3(r["risk_score"]) in txt and "threshold 0.950" in flat and "Threshold 0.950" in flat, f"{fx}: risk and threshold")
    ok((NOTE_X in flat) == bool(r["extrapolating"]), f"{fx}: magnification note iff extrapolating")
    ok((NOTE_RP in flat) == (r["specimen"] == "prostatectomy"), f"{fx}: prostatectomy note iff prostatectomy")
    ok(("Prostatectomy" if r["specimen"] == "prostatectomy" else "Biopsy") in txt, f"{fx}: specimen field")
    ok("Localized" in txt, f"{fx}: setting field")
    used = sum(s["status"] == "scored" for s in r["slides"])
    ok(f"{used} of {r['n_slides']}" in flat, f"{fx}: slides analyzed")
    hexes = re.findall(r"\b[0-9a-f]{64}\b", txt)
    want = {DIG.split(":")[1], r["model_sha256"], sha, *[s["sha256"] for s in r["slides"]]}
    ok(set(hexes) == want, f"{fx}: traceability hashes")
    low = txt.lower()
    for w in ("not", "no", "never", "without", "each"):
        ok(not re.search(rf"\b{w}\b", low), f"{fx}: banned word '{w}'")
    ok("—" not in txt and "unavailable" not in low, f"{fx}: no em dash, every thumbnail drawn")
    for bad in ("docetaxel benefit from", "triplet", "doublet", "ARPI", "enzalutamide", "CHAARTED (n"):
        ok(bad not in txt, f"{fx}: treatment selection text '{bad}'")
    # every number has a source
    rest = txt
    for s in sorted({rid, r["patient"], *[s["file"] for s in r["slides"]], "NRG/RTOG 0521", r["end_utc"][:10],
                     "2026-11-10", r["version"], "SHA-256", "H&E", *hexes}, key=len, reverse=True):
        rest = rest.replace(s, " ")
    rest = re.sub(r"\S*…\S*", " ", rest)                     # shortened file names in the traceability list
    allowed = set(CONST) | {"1", "2"} | {str(i) for i in range(1, r["n_slides"] + 1)} | {str(r["n_slides"]), str(used)}
    allowed |= {"0.950", f3(r["risk_score"])}
    for s in r["slides"]:
        if s["status"] == "scored":
            allowed |= {f3(s["risk_score"]), str(s["n_tiles_tissue"]), f"{int(s['n_nuclei']):,}",
                        f"{s['objective_power']:.0f}"}
    nums = re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?", rest)
    bad = sorted({x.rstrip(",") for x in nums if x.rstrip(",") not in allowed})
    ok(not bad, f"{fx}: numbers with no source {bad}")
    print(f"{fx}: {st}, {r['specimen']}, slides {used} of {r['n_slides']}, extrapolating {r['extrapolating']}, "
          f"numbers checked {len(nums)}, {'PASS' if len(fails) == n0 else 'FAIL'}")

# mHSPC: v2.0.4 with --setting mhspc writes the same v1 report as v2.0.3
a, b = os.path.join(W, "eq_mhspc", "v204"), os.path.join(W, "eq_mhspc", "v203")
ja, jb_ = json.load(open(os.path.join(a, "patient.json"))), json.load(open(os.path.join(b, "patient.json")))
ok(ja["setting"] == "mhspc" and ja["report"] == "v1", "mhspc: patient.json fields")
drop = {"setting", "specimen", "report", "end_utc", "seconds", "version"}
ok({k: v for k, v in ja.items() if k not in drop} == {k: v for k, v in jb_.items() if k not in drop},
   "mhspc: patient.json equals v2.0.3 apart from the new fields")
ta, tb = sh("pdftotext", "-layout", os.path.join(a, "report.pdf"), "-"), sh("pdftotext", "-layout", os.path.join(b, "report.pdf"), "-")
ok(ta == tb and len(ta) > 200, "mhspc: report text equals v2.0.3")
for p in (a, b):
    subprocess.run(["pdftoppm", "-r", "72", "-png", os.path.join(p, "report.pdf"), os.path.join(p, "px")], check=True)
pa = sorted(f for f in os.listdir(a) if f.startswith("px")); pb = sorted(f for f in os.listdir(b) if f.startswith("px"))
same = len(pa) == len(pb) == 2 and all(open(os.path.join(a, x), "rb").read() == open(os.path.join(b, y), "rb").read()
                                        for x, y in zip(pa, pb))
ok(same, "mhspc: rendered pages equal v2.0.3 pixel for pixel")
print(f"eq_mhspc: v1 report, text equal {ta == tb}, pages equal {same}")

print("\n".join(["FAIL " + f for f in fails]) or "ALL PASS")
sys.exit(1 if fails else 0)
