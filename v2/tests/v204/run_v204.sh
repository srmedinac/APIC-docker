#!/usr/bin/env bash
# APIC v2.0.4 report checks, no GPU. Patient step and reports run inside the image from fixture apic.json files.
#   APIC_TEST_SLIDES=... APIC_TEST_SOURCES=... bash v2/tests/v204/run_v204.sh WORKDIR       (after docker build -t apic:v2.0.4)
set -euo pipefail
T=$(cd "$(dirname "$0")" && pwd)
W=${1:?usage: run_v204.sh WORKDIR}
SL=${APIC_TEST_SLIDES:?set APIC_TEST_SLIDES to the slide folder}; : "${APIC_TEST_SOURCES:?set APIC_TEST_SOURCES (see make_fixtures.py)}"
IMG=apic:v2.0.4; OLD=apic:v2.0.3; DATE=2026-11-10
DIG=$(docker image inspect --platform linux/amd64 $IMG --format '{{.Descriptor.digest}}')
mkdir -p "$W"; python3 "$T/make_fixtures.py" "$W" > /dev/null
U="$(id -u):$(id -g)"
patient() {   # image fixture outdir extra-args...
  local img=$1 fx=$2 od=$3; shift 3
  local items=() ; while read -r f d; do items+=(--item "/in/$f" "/w/$fx/$d/apic.json"); done < <(python3 -c "
import json,sys; [print(f, d) for f, d in json.load(open('$W/fixtures.json'))['$fx']['items']]")
  docker run --rm --network none --user $U -v $SL:/in:ro -v $W:/w --entrypoint python3 $img \
    /opt/apic/scripts/apic_patient.py --patient "$fx" --out "/w/$od" "$@" "${items[@]}" | grep '^APIC:' || true
}
for fx in neg_bx_1 pos_bx_3 neg_rp_3 pos_rp_1 pos_bx_8; do
  sp=$(python3 -c "import json; print(json.load(open('$W/fixtures.json'))['$fx']['specimen'])")
  patient $IMG $fx $fx --setting localized --specimen $sp --image-digest "$DIG" --report-date $DATE
  pdftoppm -r 100 -png "$W/$fx/report.pdf" "$W/$fx/page"
done
# mHSPC: the v1 report, same as v2.0.3 for the same input
patient $IMG eq_mhspc eq_mhspc/v204 --setting mhspc
patient $OLD eq_mhspc eq_mhspc/v203
python3 "$T/check_report.py" "$W" "$DIG"
