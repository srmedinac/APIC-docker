"""Score the faithful v1 APIC Cox model from a slide's 6 features — no training data needed.
Loads the frozen artifact (models/apic_cox_frozen.json, built by freeze_apic_cox.R) and applies
    risk = exp( sum_i coef_i * (minmax(x_i) - center_i) )
which is bit-identical to the v1 coxph predict(type="risk"). This is the inference half of APIC
that the viewer `apic` job will call once the 6 features are computed for a slide.
"""
import json
import math
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(_HERE, "..", "models", "apic_cox_frozen.json")


def load_model(path=DEFAULT_MODEL):
    with open(path) as f:
        return json.load(f)


def score(model, feats):
    """feats: {feature_name: raw_value} for the 6 model features. Returns risk + APIC status.

    A non-finite feature is a HARD ERROR, never a score. nucdiv legitimately returns NaN when a slide
    is too sparse for a valid GLCM, and `nan > threshold` is False — so without this guard a FAILED
    computation would be reported as a confident clinical 'Low Risk / APIC-Negative', which is the
    worst possible failure mode for a biomarker."""
    lp = 0.0
    for f in model["features"]:
        x = float(feats[f])
        if not math.isfinite(x):
            raise ValueError("feature %r is %r — cannot score (slide too sparse or a stage failed)" % (f, x))
        mn, mx = model["feat_min"][f], model["feat_max"][f]
        nx = 0.0 if mx == mn else (x - mn) / (mx - mn)      # v1 min-max, frozen train params
        lp += model["coef"][f] * (nx - model["center"][f])
    risk = math.exp(lp)
    if not math.isfinite(risk):
        raise ValueError("risk score is %r — refusing to emit a group" % risk)
    high = risk > model["threshold"]
    return {
        "risk_score": risk,
        "risk_group": "High Risk" if high else "Low Risk",
        # Labels: High Risk (top 67% of the CHAARTED arm-B fit) = APIC-Positive; Low Risk = Negative.
        # WHO BENEFITS DEPENDS ON THE DRUG — the two validated axes point OPPOSITE ways, so never
        # write a drug-agnostic "benefiter" sentence (a report that does is wrong for one of them):
        #   docetaxel  (CHAARTED): APIC-POSITIVE benefits  -- scripts/seed_discovered.R:28
        #                          ("APIC-pos doce HR <1 = pos benefits, correct"), km_chaarted.R
        #   enzalutamide (ENZAMET): APIC-NEGATIVE benefits -- scripts/validate_enz.R:7 pre-registered
        #                          direction; reference/validation_recipe.md:43 (Neg HR 0.50 vs Pos 1.04)
        "apic_status": "APIC-Positive" if high else "APIC-Negative",
        "threshold": model["threshold"],
    }


if __name__ == "__main__":
    import sys
    m = load_model()
    feats = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {f: 0.0 for f in m["features"]}
    print(json.dumps(score(m, feats), indent=2))
