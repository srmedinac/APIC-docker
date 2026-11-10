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
#   --slide PATH   whole-slide image (required; repeat it for a patient with several slides)
#   --out   DIR    output directory (required); writes <out>/<stem>/apic.json and report.pdf
#   --stem  NAME   output basename (default: slide filename without its extension; one slide only)
#   --patient ID   patient mode: score every slide, then the patient (apic_patient.py) writes
#                  <out>/patient.json and the patient report <out>/report.pdf. Given several
#                  --slide without --patient, the patient label is the first slide's stem.
#   --setting S    mhspc (default) or localized; the report text follows it (apic_report_patient.py):
#                  mhspc gives the CHAARTED docetaxel and ENZAMET enzalutamide results, localized gives
#                  the prognosis on radiotherapy and ADT alone and the NRG/RTOG 0521 docetaxel result
#   --specimen S   biopsy (default) or prostatectomy, recorded in patient.json and on the report
#   --report R     v2 (default): the MAP-style patient report, always in patient mode (one slide without
#                  --patient is a patient named by the slide stem). v1: the v1.0.5 BioRender report
#                  (apic_report.py), mHSPC only; one slide without --patient then keeps the v2.0.2
#                  single-slide behavior, unchanged
#   --image-digest sha256:...   printed in the report traceability (v2)
#   --report-date  YYYY-MM-DD   printed on the report (v2; default: today UTC)
# Any other argument is passed through to apic_one_slide.py unchanged.
#
# One slide, no --patient and --report v1 is the v2.0.2 behavior, unchanged: same files, same output lines.
set -euo pipefail

SLIDES=(); OUT=""; STEM=""; PATIENT=""; PASS=(); SETTING="mhspc"; SPECIMEN="biopsy"; REPORT=(); STYLE="v2"
while [ $# -gt 0 ]; do
  case "$1" in
    --slide)   SLIDES+=("$2"); shift 2 ;;
    --out)     OUT="$2";     shift 2 ;;
    --stem)    STEM="$2";    shift 2 ;;
    --patient) PATIENT="$2"; shift 2 ;;
    --setting) SETTING="$2"; shift 2 ;;
    --specimen) SPECIMEN="$2"; shift 2 ;;
    --report)  STYLE="$2"; shift 2 ;;
    --image-digest|--report-date) REPORT+=("$1" "$2"); shift 2 ;;
    -h|--help) exec python3 /opt/apic/scripts/apic_one_slide.py --help ;;
    *) PASS+=("$1"); shift ;;
  esac
done
[ ${#SLIDES[@]} -gt 0 ] || { echo "apic: --slide is required" >&2; exit 2; }
[ -n "$OUT" ]   || { echo "apic: --out is required (a directory)" >&2; exit 2; }
case "$SETTING" in mhspc|localized) ;; *) echo "apic: --setting must be mhspc or localized" >&2; exit 2 ;; esac
case "$SPECIMEN" in biopsy|prostatectomy) ;; *) echo "apic: --specimen must be biopsy or prostatectomy" >&2; exit 2 ;; esac
case "$STYLE" in v2|v1) ;; *) echo "apic: --report must be v2 or v1" >&2; exit 2 ;; esac
[ "$STYLE" = "v2" ] || [ "$SETTING" = "mhspc" ] || {
  echo "apic: --report v1 carries mHSPC treatment considerations and is refused for --setting localized" >&2; exit 2; }

if [ -n "$PATIENT" ] || [ ${#SLIDES[@]} -gt 1 ] || [ "$STYLE" = "v2" ]; then
  # ---- patient mode ---------------------------------------------------------------------------
  [ -z "$STEM" ] || { echo "apic: --stem works with one slide and no --patient" >&2; exit 2; }
  for s in "${SLIDES[@]}"; do [ -f "$s" ] || { echo "apic: no such slide: $s" >&2; exit 2; }; done
  mkdir -p "$OUT"
  if [ -z "$PATIENT" ]; then PATIENT="$(basename "${SLIDES[0]}")"; PATIENT="${PATIENT%.*}"; fi
  N=${#SLIDES[@]}; I=0; ITEMS=(); declare -A USED=()
  for s in "${SLIDES[@]}"; do
    I=$((I + 1))
    st="$(basename "$s")"; st="${st%.*}"
    # One directory per slide (the tissue-mask cache is keyed by directory, see below); two inputs
    # with one stem, such as a.svs and a.ndpi, must not share it.
    if [ -n "${USED[$st]+x}" ]; then st="${st}_$I"; fi
    USED[$st]=1
    d="$OUT/$st"; mkdir -p "$d"; rm -f "$d/apic.json" "$d/error.txt"
    echo "APIC:stage score $I/$N $st"
    if python3 /opt/apic/scripts/apic_one_slide.py --slide "$s" --stem "$st" --out "$d/apic.json" \
         ${PASS[@]+"${PASS[@]}"} 2> >(tee "$d/score.log" >&2); then
      python3 - "$d/apic.json" <<'SLIDECHECK' || true
import json, math, os, sys
p = sys.argv[1]
try:
    d = json.load(open(p)); s = d.get("risk_score")
    ok = isinstance(s, (int, float)) and not isinstance(s, bool) and math.isfinite(s) and d.get("n_nuclei")
except Exception as e:
    ok, why = False, str(e)
else:
    why = "apic.json has no usable risk_score or no nuclei"
if not ok:
    open(os.path.join(os.path.dirname(p), "error.txt"), "w").write(why)
    os.replace(p, p + ".rejected")
SLIDECHECK
    else
      sleep 0.2                                     # let tee flush the score log
      tail -n 20 "$d/score.log" > "$d/error.txt" 2>/dev/null || echo "score stage failed" > "$d/error.txt"
      rm -f "$d/apic.json"
    fi
    ITEMS+=(--item "$s" "$d/apic.json")
  done
  echo "APIC:stage patient"
  python3 /opt/apic/scripts/apic_patient.py --patient "$PATIENT" --out "$OUT" --setting "$SETTING" \
    --specimen "$SPECIMEN" --report "$STYLE" ${REPORT[@]+"${REPORT[@]}"} "${ITEMS[@]}"
  [ -s "$OUT/report.pdf" ] || { echo "apic: no report written at $OUT/report.pdf" >&2; exit 1; }
  echo "APIC:done $OUT/patient.json $OUT/report.pdf"
  exit 0
fi

SLIDE="${SLIDES[0]}"
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
