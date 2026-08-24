#!/usr/bin/env bash
# One container invocation: WSI in, apic.json AND report.pdf out.
#
# The published v2.0.0 ENTRYPOINT is `apic_one_slide.py`, which writes only the JSON. The PDF
# needs a second call to `apic_report.py`. Every caller therefore had to reimplement the same two
# steps. This wrapper is the only behavior ADDED to the image; both original scripts stay in place
# and are still reachable with `--entrypoint python3`.
#
#   docker run --gpus all -v /slides:/in:ro -v /out:/out <image> --slide /in/x.svs --out /out
#
#   --slide PATH   whole-slide image (required)
#   --out   DIR    output directory (required); writes <out>/<stem>/apic.json and report.pdf
#   --stem  NAME   output basename (default: slide filename without its extension)
# Any other argument is passed through to apic_one_slide.py unchanged.
set -euo pipefail

SLIDE=""; OUT=""; STEM=""; PASS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --slide) SLIDE="$2"; shift 2 ;;
    --out)   OUT="$2";   shift 2 ;;
    --stem)  STEM="$2";  shift 2 ;;
    -h|--help) exec python3 /opt/apic/scripts/apic_one_slide.py --help ;;
    *) PASS+=("$1"); shift ;;
  esac
done
[ -n "$SLIDE" ] || { echo "apic: --slide is required" >&2; exit 2; }
[ -n "$OUT" ]   || { echo "apic: --out is required (a directory)" >&2; exit 2; }
[ -f "$SLIDE" ] || { echo "apic: no such slide: $SLIDE" >&2; exit 2; }
mkdir -p "$OUT"
[ -n "$STEM" ] || { STEM="$(basename "$SLIDE")"; STEM="${STEM%.*}"; }

# One directory PER SLIDE, and this is not cosmetic. apic_one_slide.py caches the HistoQC tissue
# mask at dirname(--out)/tissue_mask.png with NO slide key, and reuses it whenever it is there.
# HistoQC is not reproducible (the docstring measures ~1.2-1.5% of mask pixels moving run to run),
# so caching is how a re-score stays bit-identical. But point two slides at one directory and the
# second is silently scored with the first slide's tissue. Writing <out>/<stem>/ makes the cache
# per slide, which is what the upstream `--out /out/foo/apic.json` invocation assumed all along.
SLIDE_OUT="$OUT/$STEM"
mkdir -p "$SLIDE_OUT"
JSON="$SLIDE_OUT/apic.json"
PDF="$SLIDE_OUT/report.pdf"

echo "APIC:stage score"
python3 /opt/apic/scripts/apic_one_slide.py --slide "$SLIDE" --stem "$STEM" --out "$JSON" ${PASS[@]+"${PASS[@]}"}

# The score stage has been seen to exit 0 with the real failure downstream, so exit 0 alone does
# not mean the JSON is usable. Check it before building a report on it, and before this wrapper
# returns 0 to a caller who will trust that.
python3 - "$JSON" <<'CHECK'
import json, math, sys
path = sys.argv[1]
try:
    d = json.load(open(path))
except Exception as e:
    sys.exit(f"apic: {path} did not parse: {e}")
score = d.get("risk_score")
if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
    sys.exit(f"apic: {path} has no usable risk_score: {score!r}")
if d.get("risk_group") not in ("High Risk", "Low Risk"):
    sys.exit(f"apic: {path} has no risk_group: {d.get('risk_group')!r}")
if not d.get("n_nuclei"):
    sys.exit(f"apic: {path} counted no nuclei")
print(f"APIC:check risk_score={score} {d['risk_group']} nuclei={d['n_nuclei']}")
CHECK

echo "APIC:stage report"
python3 /opt/apic/scripts/apic_report.py --apic "$JSON" --out "$PDF"
[ -s "$PDF" ] || { echo "apic: no report written at $PDF" >&2; exit 1; }
echo "APIC:done $JSON $PDF"
