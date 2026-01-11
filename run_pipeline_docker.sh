#!/usr/bin/env bash
set -euo pipefail

INPUT_SLIDE=""
OUTPUT_DIR=""
MULTI_SLIDE=false
RESUME=false
PATIENT_ID=""
RESEARCH_MODE=false

# Timing variables for ETA calculation
declare -a SLIDE_TIMES=()
BATCH_START=0

# Format seconds to human readable time
format_time() {
  local seconds=$1
  if (( seconds < 60 )); then
    echo "${seconds}s"
  elif (( seconds < 3600 )); then
    echo "$((seconds / 60))m $((seconds % 60))s"
  else
    echo "$((seconds / 3600))h $((seconds % 3600 / 60))m"
  fi
}

# Calculate average of array
calc_avg() {
  local sum=0
  local count=0
  for val in "$@"; do
    sum=$((sum + val))
    ((count++))
  done
  if (( count > 0 )); then
    echo $((sum / count))
  else
    echo 0
  fi
}

# Print progress bar
print_progress() {
  local current=$1
  local total=$2
  local width=30
  local percent=$((current * 100 / total))
  local filled=$((current * width / total))
  local empty=$((width - filled))

  printf "  ["
  printf "%${filled}s" | tr ' ' '#'
  printf "%${empty}s" | tr ' ' '-'
  printf "] %d/%d (%d%%)" "$current" "$total" "$percent"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input)
      INPUT_SLIDE="$2"
      shift 2
      ;;
    -o|--output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --multi-slide)
      MULTI_SLIDE=true
      shift
      ;;
    --resume)
      RESUME=true
      shift
      ;;
    --patient-id)
      PATIENT_ID="$2"
      shift 2
      ;;
    --research-mode)
      RESEARCH_MODE=true
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "${INPUT_SLIDE}" || -z "${OUTPUT_DIR}" ]]; then
  echo "Usage: $0 -i <input_slide> -o <output_dir> [--multi-slide] [--patient-id <id>] [--resume] [--research-mode]"
  echo ""
  echo "Options:"
  echo "  --resume            Skip already-processed slides"
  echo "  --patient-id <id>   Patient identifier for multi-slide mode (required for single-patient multi-slide)"
  echo "  --research-mode     Research mode: auto-group slides by patient ID from filename"
  echo ""
  echo "Modes:"
  echo "  Single slide:       -i /path/to/slide.svs -o /output"
  echo "  Batch mode:         -i /path/to/slides_dir/ -o /output"
  echo "                      (Each slide = different patient)"
  echo "  Multi-slide:        -i /path/to/patient_dir/ -o /output --multi-slide --patient-id PT001"
  echo "                      (All slides in folder = one patient, features averaged)"
  echo "  Batch multi-slide:  -i /path/to/patients_dir/ -o /output --multi-slide"
  echo "                      (Each subfolder = one patient with multiple slides)"
  echo "  Research mode:      -i /path/to/slides_dir/ -o /output --research-mode"
  echo "                      (Slides auto-grouped by patient ID from filename, outputs CSV)"
  exit 1
fi

echo ""
echo "============================================================"
echo "  APIC PIPELINE - Predictive Biomarker for Prostate Cancer"
echo "============================================================"
echo ""
echo "  Input:  ${INPUT_SLIDE}"
echo "  Output: ${OUTPUT_DIR}"
[[ "$RESUME" == "true" ]] && echo "  Mode:   Resume (skipping completed slides)"
echo ""

mkdir -p "${OUTPUT_DIR}"

SUPPORTED_EXTS=("svs" "tif" "tiff" "ndpi" "mrxs" "scn")

run_one () {
  local SLIDE="$1"
  local STEPS_ARG="${2:-}"
  local SLIDE_NAME=$(basename "$SLIDE" | sed 's/\.[^.]*$//')
  local COMPLETION_MARKER="${OUTPUT_DIR}/${SLIDE_NAME}/.complete"
  local SLIDE_START=$(date +%s)

  # Check if already processed (resume mode)
  if [[ "$RESUME" == "true" && -f "$COMPLETION_MARKER" ]]; then
    echo "  [SKIP] Already processed: $SLIDE_NAME"
    return 0
  fi

  echo ""
  echo "  Processing: $SLIDE_NAME"
  echo "  ------------------------------------------------------------"

  echo "  [1/2] Tissue segmentation & patch extraction..."
  conda run --no-capture-output -n histoqc_env python -u /app/tissue_segmentation_patches.py \
    -i "$SLIDE" \
    -o "$OUTPUT_DIR"

  echo "  [2/2] Feature extraction & biomarker prediction..."
  if [[ -n "$STEPS_ARG" ]]; then
    conda run --no-capture-output -n apic_env python -u /app/feature_extraction_prediction.py \
      -i "$SLIDE" \
      -o "$OUTPUT_DIR" \
      --steps $STEPS_ARG
  else
    conda run --no-capture-output -n apic_env python -u /app/feature_extraction_prediction.py \
      -i "$SLIDE" \
      -o "$OUTPUT_DIR"
  fi

  # Mark as complete
  touch "$COMPLETION_MARKER"

  local SLIDE_END=$(date +%s)
  local SLIDE_DURATION=$((SLIDE_END - SLIDE_START))
  SLIDE_TIMES+=($SLIDE_DURATION)
  echo ""
  echo "  Completed in $(format_time $SLIDE_DURATION)"
}

process_patient_folder () {
  local PATIENT_FOLDER="$1"
  local PATIENT_ID_OVERRIDE="${2:-}"

  # Use override if provided, otherwise use folder basename
  local PATIENT_ID
  if [[ -n "$PATIENT_ID_OVERRIDE" ]]; then
    PATIENT_ID="$PATIENT_ID_OVERRIDE"
  else
    PATIENT_ID=$(basename "${PATIENT_FOLDER}")
  fi

  echo ""
  echo "  Patient: $PATIENT_ID"
  echo "  ============================================================"

  # Count slides for this patient
  local SLIDE_COUNT=0
  shopt -s nullglob
  for ext in "${SUPPORTED_EXTS[@]}"; do
    for f in "${PATIENT_FOLDER}"/*.${ext}; do
      ((SLIDE_COUNT++)) || true
    done
  done
  shopt -u nullglob

  if [[ $SLIDE_COUNT -eq 0 ]]; then
    echo "  [SKIP] No supported WSI files found"
    return
  fi

  echo "  Found $SLIDE_COUNT slide(s)"

  # Process each slide (include overlay step for patient-level report)
  local CURRENT=0
  shopt -s nullglob
  for ext in "${SUPPORTED_EXTS[@]}"; do
    for f in "${PATIENT_FOLDER}"/*.${ext}; do
      ((CURRENT++)) || true
      echo ""
      echo "    Slide $CURRENT/$SLIDE_COUNT: $(basename "$f")"
      run_one "$f" "nuclei spatil nucdiv aggregate overlay"
    done
  done
  shopt -u nullglob

  # Run patient-level aggregation
  echo ""
  echo "  Patient-Level Aggregation & Prediction"
  echo "  ------------------------------------------------------------"
  conda run --no-capture-output -n apic_env python -u /app/feature_extraction_prediction.py \
    --patient-aggregate \
    --patient-id "$PATIENT_ID" \
    --patient-folder "${PATIENT_FOLDER}" \
    -o "$OUTPUT_DIR"
}

# Print ETA after processing a slide
print_eta() {
  local current=$1
  local total=$2
  local remaining=$((total - current))

  if (( ${#SLIDE_TIMES[@]} > 0 && remaining > 0 )); then
    local avg_time=$(calc_avg "${SLIDE_TIMES[@]}")
    local eta_seconds=$((avg_time * remaining))
    echo ""
    print_progress "$current" "$total"
    echo " | ETA: $(format_time $eta_seconds)"
  fi
}

BATCH_START=$(date +%s)

if [[ "${MULTI_SLIDE}" == "true" ]] && [[ -d "${INPUT_SLIDE}" ]]; then
  # Check if input contains slides directly or subdirectories
  DIRECT_SLIDE_COUNT=0
  shopt -s nullglob
  for ext in "${SUPPORTED_EXTS[@]}"; do
    for f in "${INPUT_SLIDE}"/*.${ext}; do
      ((DIRECT_SLIDE_COUNT++)) || true
    done
  done
  shopt -u nullglob

  if [[ $DIRECT_SLIDE_COUNT -gt 0 ]]; then
    # Single patient with multiple slides - require --patient-id
    if [[ -z "$PATIENT_ID" ]]; then
      echo "ERROR: --patient-id is required for single-patient multi-slide mode"
      echo ""
      echo "Example: $0 -i /data/input_slides/ -o /output --multi-slide --patient-id PT001"
      exit 1
    fi
    echo "Mode: Multi-slide (single patient)"
    echo "Patient ID: $PATIENT_ID"
    process_patient_folder "${INPUT_SLIDE}" "$PATIENT_ID"
  else
    echo "Mode: Batch Multi-slide"

    # Count patient folders
    PATIENT_COUNT=0
    for patient_dir in "${INPUT_SLIDE}"/*/; do
      if [[ -d "$patient_dir" ]]; then
        ((PATIENT_COUNT++)) || true
      fi
    done

    if [[ $PATIENT_COUNT -eq 0 ]]; then
      echo "ERROR: No patient subdirectories found in: ${INPUT_SLIDE}"
      exit 1
    fi

    echo "Found $PATIENT_COUNT patient folder(s)"

    CURRENT_PATIENT=0
    for patient_dir in "${INPUT_SLIDE}"/*/; do
      if [[ -d "$patient_dir" ]]; then
        ((CURRENT_PATIENT++)) || true
        echo ""
        echo "============================================================"
        echo "  Patient $CURRENT_PATIENT of $PATIENT_COUNT"
        echo "============================================================"
        process_patient_folder "${patient_dir%/}"
        print_eta "$CURRENT_PATIENT" "$PATIENT_COUNT"
      fi
    done
  fi

elif [[ "${RESEARCH_MODE}" == "true" ]] && [[ -d "${INPUT_SLIDE}" ]]; then
  echo "Mode: Research (batch with auto-grouping by patient ID)"
  echo "Patient IDs will be extracted from filenames (number before first underscore)"

  # Count slides
  COUNT=0
  shopt -s nullglob
  for ext in "${SUPPORTED_EXTS[@]}"; do
    for f in "${INPUT_SLIDE}"/*.${ext}; do
      ((COUNT++)) || true
    done
  done
  shopt -u nullglob

  if [[ $COUNT -eq 0 ]]; then
    echo "ERROR: No supported WSI files found in: ${INPUT_SLIDE}"
    echo "Supported formats: ${SUPPORTED_EXTS[*]}"
    exit 1
  fi

  echo "Found $COUNT slide(s) to process"
  echo ""

  # Phase 1: Process all slides without report generation
  CURRENT=0
  shopt -s nullglob
  for ext in "${SUPPORTED_EXTS[@]}"; do
    for f in "${INPUT_SLIDE}"/*.${ext}; do
      ((CURRENT++)) || true
      echo ""
      echo "============================================================"
      echo "  Slide $CURRENT of $COUNT"
      echo "============================================================"
      run_one "$f" "nuclei spatil nucdiv aggregate predict overlay"
      print_eta "$CURRENT" "$COUNT"
    done
  done
  shopt -u nullglob

  # Phase 2: Research aggregation - group by patient ID and generate summary CSV
  echo ""
  echo "============================================================"
  echo "  Research Mode: Patient Aggregation"
  echo "============================================================"
  conda run --no-capture-output -n apic_env python -u /app/feature_extraction_prediction.py \
    --research-aggregate \
    --input-dir "${INPUT_SLIDE}" \
    -o "$OUTPUT_DIR"

elif [[ -d "${INPUT_SLIDE}" ]]; then
  echo "Mode: Batch (directory)"

  COUNT=0
  shopt -s nullglob
  for ext in "${SUPPORTED_EXTS[@]}"; do
    for f in "${INPUT_SLIDE}"/*.${ext}; do
      ((COUNT++)) || true
    done
  done
  shopt -u nullglob

  if [[ $COUNT -eq 0 ]]; then
    echo "ERROR: No supported WSI files found in: ${INPUT_SLIDE}"
    echo "Supported formats: ${SUPPORTED_EXTS[*]}"
    exit 1
  fi

  echo "Found $COUNT slide(s) to process"

  CURRENT=0
  shopt -s nullglob
  for ext in "${SUPPORTED_EXTS[@]}"; do
    for f in "${INPUT_SLIDE}"/*.${ext}; do
      ((CURRENT++)) || true
      echo ""
      echo "============================================================"
      echo "  Slide $CURRENT of $COUNT"
      echo "============================================================"
      run_one "$f"
      print_eta "$CURRENT" "$COUNT"
    done
  done
  shopt -u nullglob

else
  echo "Mode: Single slide"
  run_one "${INPUT_SLIDE}"
fi

BATCH_END=$(date +%s)
TOTAL_TIME=$((BATCH_END - BATCH_START))

echo ""
echo "============================================================"
echo "  PIPELINE COMPLETE"
echo "  Total time: $(format_time $TOTAL_TIME)"
echo "============================================================"
echo ""
