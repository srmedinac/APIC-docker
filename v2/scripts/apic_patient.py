"""APIC for ONE PATIENT with one or more slides: the APIC v1 rule, then the patient report.

APIC v1 scores a patient, not a slide. Its multi-slide pipeline (feature_extraction_prediction.py,
aggregate_patient_features) takes the slide-level spaTIL and nucdiv features, averages them over the
patient's slides with np.nanmean, and feeds the 6 model features of that average to the Cox model.
v1's stored ENZAMET batch-3 outputs follow that rule exactly: on the 12 patients with more than one
slide, the stored patient features equal the nanmean of the stored slide features to 1.4e-13.

This script does the same with the per-slide apic.json files that apic_one_slide.py writes:

    patient feature_i = nanmean over scored slides of slide feature_i     (6 model features)
    patient risk      = frozen v1 Cox (scripts/apic_score.py) on those 6 values

A slide whose score stage failed has no apic.json and is listed as failed; the patient value uses
the scored slides. apic_one_slide.py refuses to write a JSON with a non-finite feature, so every
scored slide carries 6 finite values and the nanmean over them is a plain mean. (v1 kept a slide
whose single feature was NaN and averaged its other features; v2 drops that slide whole. Both give
the same patient value whenever every slide scores.)

Writes <out>/patient.json, and <out>/report.pdf in the v1.0.5 report format with the patient ID
and the patient score. The tissue panel and example tiles come from the first scored slide, as
v1's patient pipeline took its overlay from the first slide that had one.

Usage:
    python3 apic_patient.py --patient P1 --out /out \\
        --item /in/a.svs /out/a/apic.json --item /in/b.svs /out/b/apic.json
Exit 0 when patient.json holds a finite risk_score and report.pdf exists; exit 1 when no slide
scored (patient.json is still written, with null results and the error).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import apic_score  # noqa: E402

MODEL = os.path.join(_HERE, "..", "models", "apic_cox_frozen.json")
VERSION = os.environ.get("APIC_VERSION", "unknown")
AGGREGATION = "nanmean of the 6 model features over scored slides (APIC v1 rule)"
SLIDE_KEYS = ("risk_score", "risk_group", "apic_status", "features", "n_tiles_tissue", "n_tiles_with_nuclei",
              "n_nuclei", "objective_power", "mpp_x", "extrapolating", "warning", "seconds")


def sha256_file(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def slide_error(json_path):
    """The failure text the wrapper saved beside the missing apic.json, or a plain default."""
    p = os.path.join(os.path.dirname(json_path), "error.txt")
    try:
        t = open(p, encoding="utf-8", errors="replace").read().strip()
    except OSError:
        t = ""
    return t[-2000:] or "score stage failed"


def slide_record(slide_path, json_path):
    rec = {"stem": os.path.basename(os.path.dirname(json_path)), "file": os.path.basename(slide_path),
           "sha256": sha256_file(slide_path), "size_bytes": os.path.getsize(slide_path),
           "status": "failed", "error": None}
    rec.update({k: None for k in SLIDE_KEYS})
    try:
        d = json.load(open(json_path))
    except (OSError, ValueError):
        rec["error"] = slide_error(json_path)
        return rec, None
    score = d.get("risk_score")
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
        rec["error"] = "apic.json has no usable risk_score"
        return rec, None
    rec.update({k: d.get(k) for k in SLIDE_KEYS})
    rec["status"] = "scored"
    return rec, d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patient", required=True)
    ap.add_argument("--out", required=True, help="output directory: patient.json and report.pdf")
    ap.add_argument("--item", nargs=2, action="append", required=True, metavar=("SLIDE", "APIC_JSON"),
                    help="one slide and the apic.json its score stage wrote (repeat per slide, in order)")
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    t0 = time.time()
    model = apic_score.load_model(a.model)
    names = list(model["features"])

    slides, scored = [], []
    for slide_path, json_path in a.item:
        rec, d = slide_record(slide_path, json_path)
        slides.append(rec)
        if d is not None:
            scored.append((slide_path, json_path, d))

    out = {"version": VERSION, "patient": a.patient, "aggregation": AGGREGATION,
           "model": model.get("model"), "model_sha256": sha256_file(a.model),
           "features": None, "risk_score": None, "risk_group": None, "apic_status": None,
           "threshold": model["threshold"], "n_slides": len(slides), "n_slides_scored": len(scored),
           "extrapolating": any(bool(s.get("extrapolating")) for s in slides if s["status"] == "scored"),
           "slides": slides, "error": None, "seconds": None}

    if scored:
        import numpy as np
        feats = {}
        for f in names:                           # v1: np.nanmean(np.vstack(slide vectors), axis=0)
            col = np.array([float((d.get("features") or {}).get(f, float("nan"))) for _, _, d in scored])
            feats[f] = float(np.nanmean(col)) if np.isfinite(col).any() else float("nan")
        try:
            res = apic_score.score(model, feats)
            out.update(features=feats, risk_score=res["risk_score"], risk_group=res["risk_group"],
                       apic_status=res["apic_status"])
        except ValueError as e:
            out["error"] = str(e)
            out["features"] = {k: (v if math.isfinite(v) else None) for k, v in feats.items()}
    else:
        out["error"] = "no slide scored"

    os.makedirs(a.out, exist_ok=True)
    pj = os.path.join(a.out, "patient.json")

    def write():
        out["seconds"] = round(time.time() - t0, 3)
        tmp = pj + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=2, allow_nan=False)
        os.replace(tmp, pj)

    if out["risk_score"] is None:
        write()
        print(f"apic: {a.patient}: {out['error']}", file=sys.stderr)
        sys.exit(1)

    print("APIC:stage report", flush=True)
    import apic_report
    first_slide, first_json, _ = scored[0]
    rep = {"patient": a.patient, "stem": a.patient, "slide": first_slide, "_json_path": os.path.abspath(first_json),
           "_mtime": time.time(), "risk_score": out["risk_score"], "risk_group": out["risk_group"],
           "threshold": out["threshold"], "features": out["features"]}
    pdf = os.path.join(a.out, "report.pdf")
    write()                                       # the result stands even if the report step fails
    apic_report.build(rep, pdf, template_dir=os.environ.get("APIC_TEMPLATE_DIR", apic_report.TEMPLATE_DIR))
    if not os.path.isfile(pdf) or os.path.getsize(pdf) == 0:
        print("apic: no report written", file=sys.stderr)
        sys.exit(1)
    print(f"APIC:check patient risk_score={out['risk_score']} {out['risk_group']} "
          f"slides {len(scored)} of {len(slides)}", flush=True)


if __name__ == "__main__":
    main()
