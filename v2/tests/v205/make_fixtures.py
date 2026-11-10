"""Fixtures for the v2.0.5 report checks: per-slide apic.json files and tissue masks copied from real v2 runs,
with the extrapolating flag set per fixture. No GPU. The patient step itself (apic_patient.py) runs inside the image.

    python3 make_fixtures.py WORKDIR      -> WORKDIR/<fixture>/<slide dir>/{apic.json,tissue_mask.png}, WORKDIR/fixtures.json
"""
import json
import os
import shutil
import sys

# Source runs are local and not in the repository. APIC_TEST_SOURCES names a JSON file:
#   {"A": [slide file, run folder], "B": [...], "P": [...]}
# A and B: two slides of one patient. P: one slide of another patient. Each run folder holds the
# apic.json and tissue_mask.png of a v2 run on that slide.
SRC = {k: tuple(v) for k, v in json.load(open(os.environ["APIC_TEST_SOURCES"])).items()}
# fixture: setting, specimen, [(slide source, feature source, extrapolating, failed)]
FIX = {
    "neg_bx_1": ("localized", "biopsy", [("A", "A", False, False)]),
    "pos_bx_3": ("localized", "biopsy", [("P", "P", True, False), ("A", "P", True, False), ("B", "P", True, False)]),
    "neg_rp_3": ("localized", "prostatectomy", [("A", "A", True, False), ("B", "B", True, False), ("A", "A", True, False)]),
    "pos_rp_1": ("localized", "prostatectomy", [("P", "P", False, False)]),
    "pos_bx_8": ("localized", "biopsy", [("P", "P", i % 2 == 0, i == 5) for i in range(8)]),
    "m_neg_bx_1": ("mhspc", "biopsy", [("A", "A", False, False)]),
    "m_pos_bx_3": ("mhspc", "biopsy", [("P", "P", True, False), ("A", "P", True, False), ("B", "P", True, False)]),
    "m_neg_rp_3": ("mhspc", "prostatectomy", [("A", "A", True, False), ("B", "B", True, False), ("A", "A", True, False)]),
    "m_pos_rp_1": ("mhspc", "prostatectomy", [("P", "P", False, False)]),
    "eq_mhspc": ("mhspc", "biopsy", [("A", "A", True, False), ("B", "B", True, False)]),
}

w = sys.argv[1]
spec = {}
for name, (setting, specimen, items) in FIX.items():
    d = os.path.join(w, name)
    shutil.rmtree(d, ignore_errors=True)
    rows = []
    for i, (sl, fs, extrap, failed) in enumerate(items, 1):
        slide, src = SRC[sl]
        sd = os.path.join(d, f"s{i}_{os.path.splitext(slide)[0]}")
        os.makedirs(sd)
        shutil.copy(os.path.join(SRC[sl][1], "tissue_mask.png"), sd)
        if failed:
            open(os.path.join(sd, "error.txt"), "w").write("score stage failed (fixture)")
        else:
            rec = json.load(open(os.path.join(SRC[fs][1], "apic.json")))
            rec["extrapolating"] = extrap
            json.dump(rec, open(os.path.join(sd, "apic.json"), "w"), indent=2)
        rows.append([slide, os.path.basename(sd)])
    spec[name] = {"setting": setting, "specimen": specimen, "items": rows}
json.dump(spec, open(os.path.join(w, "fixtures.json"), "w"), indent=1)
print(" ".join(spec))
