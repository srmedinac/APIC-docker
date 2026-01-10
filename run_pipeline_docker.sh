#!/usr/bin/env bash
set -euo pipefail

INPUT_SLIDE=""
OUTPUT_DIR=""
MULTI_SLIDE=false

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
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "${INPUT_SLIDE}" || -z "${OUTPUT_DIR}" ]]; then
  echo "Usage: $0 -i <input_slide> -o <output_dir> [--multi-slide]"
  echo ""
  echo "Modes:"
  echo "  Single slide:       -i /path/to/slide.svs -o /output"
  echo "  Batch mode:         -i /path/to/slides_dir/ -o /output"
  echo "                      (Each slide = different patient)"
  echo "  Multi-slide:        -i /path/to/patient_dir/ -o /output --multi-slide"
  echo "                      (All slides in folder = one patient, features averaged)"
  echo "  Batch multi-slide:  -i /path/to/patients_dir/ -o /output --multi-slide"
  echo "                      (Each subfolder = one patient with multiple slides)"
  echo "                      e.g., /patients/PT1/slide1.svs, /patients/PT2/slide1.svs..."
  exit 1
fi

echo ""
echo "============================================================"
echo "  APIC PIPELINE - Predictive Biomarker for Prostate Cancer"
echo "============================================================"
echo ""
echo "Input:  ${INPUT_SLIDE}"
echo "Output: ${OUTPUT_DIR}"
echo ""

mkdir -p "${OUTPUT_DIR}"

SUPPORTED_EXTS=("svs" "tif" "tiff" "ndpi" "mrxs" "scn")

run_one () {
  local SLIDE="$1"
  local STEPS_ARG="${2:-}"  # Optional: specific steps to run (e.g., "nuclei spatil nucdiv aggregate")
  local SLIDE_NAME=$(basename "$SLIDE" | sed 's/\.[^.]*$//')

  echo "============================================================"
  echo "Processing: $SLIDE_NAME"
  echo "============================================================"
  echo ""

  echo "[1/2] Tissue segmentation & patch extraction..."
  conda run --no-capture-output -n histoqc_env python -u /app/tissue_segmentation_patches.py \
    -i "$SLIDE" \
    -o "$OUTPUT_DIR"
  echo ""

  echo "[2/2] Feature extraction & biomarker prediction..."
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
  echo ""
}

process_patient_folder () {
  local PATIENT_FOLDER="$1"
  local PATIENT_ID=$(basename "${PATIENT_FOLDER}")

  echo ""
  echo "============================================================"
  echo "  Patient: $PATIENT_ID"
  echo "============================================================"

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
    echo "  [SKIP] No supported WSI files found for patient $PATIENT_ID"
    return
  fi

  echo "  Found $SLIDE_COUNT slide(s)"
  echo ""

  # Process each slide through steps 1-6 only
  local CURRENT=0
  shopt -s nullglob
  for ext in "${SUPPORTED_EXTS[@]}"; do
    for f in "${PATIENT_FOLDER}"/*.${ext}; do
      ((CURRENT++)) || true
      echo ""
      echo "  --------------------------------------------------------"
      echo "    Slide $CURRENT of $SLIDE_COUNT: $(basename "$f")"
      echo "  --------------------------------------------------------"
      run_one "$f" "nuclei spatil nucdiv aggregate"
    done
  done
  shopt -u nullglob

  # Run patient-level aggregation, prediction, and report
  echo ""
  echo "  --------------------------------------------------------"
  echo "    Patient-Level Aggregation & Prediction"
  echo "  --------------------------------------------------------"
  echo ""
  conda run --no-capture-output -n apic_env python -u /app/feature_extraction_prediction.py \
    --patient-aggregate \
    --patient-id "$PATIENT_ID" \
    --patient-folder "${PATIENT_FOLDER}" \
    -o "$OUTPUT_DIR"
}

if [[ "${MULTI_SLIDE}" == "true" ]] && [[ -d "${INPUT_SLIDE}" ]]; then
  # Check if input contains slides directly (single patient) or subdirectories (batch)
  DIRECT_SLIDE_COUNT=0
  shopt -s nullglob
  for ext in "${SUPPORTED_EXTS[@]}"; do
    for f in "${INPUT_SLIDE}"/*.${ext}; do
      ((DIRECT_SLIDE_COUNT++)) || true
    done
  done
  shopt -u nullglob

  if [[ $DIRECT_SLIDE_COUNT -gt 0 ]]; then
    # Single patient multi-slide mode (slides directly in input folder)
    echo "Mode: Multi-slide (single patient with multiple slides)"
    echo ""
    process_patient_folder "${INPUT_SLIDE}"

  else
    # Batch multi-slide mode (subdirectories are patient folders)
    echo "Mode: Batch Multi-slide (multiple patients, each with multiple slides)"
    echo ""

    # Count patient folders
    PATIENT_COUNT=0
    for patient_dir in "${INPUT_SLIDE}"/*/; do
      if [[ -d "$patient_dir" ]]; then
        ((PATIENT_COUNT++)) || true
      fi
    done

    if [[ $PATIENT_COUNT -eq 0 ]]; then
      echo "ERROR: No patient subdirectories found in: ${INPUT_SLIDE}"
      echo "Expected structure: /input/PT1/slide1.svs, /input/PT2/slide1.svs, ..."
      exit 1
    fi

    echo "Found $PATIENT_COUNT patient folder(s)"
    echo ""

    # Process each patient folder
    CURRENT_PATIENT=0
    for patient_dir in "${INPUT_SLIDE}"/*/; do
      if [[ -d "$patient_dir" ]]; then
        ((CURRENT_PATIENT++)) || true
        echo ""
        echo "============================================================"
        echo "  Patient $CURRENT_PATIENT of $PATIENT_COUNT"
        echo "============================================================"
        process_patient_folder "${patient_dir%/}"
      fi
    done
  fi

elif [[ -d "${INPUT_SLIDE}" ]]; then
  echo "Mode: Batch (directory)"
  echo ""

  FOUND=false
  COUNT=0

  # Count slides first
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
    done
  done
  shopt -u nullglob

else
  echo "Mode: Single slide"
  echo ""
  run_one "${INPUT_SLIDE}"
fi

echo ""
echo "============================================================"
echo "  PIPELINE COMPLETE"
echo "============================================================"
echo ""
