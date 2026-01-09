#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# APIC Pipeline Runner
# =============================================================================

INPUT_SLIDE=""
OUTPUT_DIR=""

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
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "${INPUT_SLIDE}" || -z "${OUTPUT_DIR}" ]]; then
  echo "Usage: $0 -i <input_slide> -o <output_dir>"
  exit 1
fi

# =============================================================================
# Header
# =============================================================================
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

# =============================================================================
# Process single slide
# =============================================================================
run_one () {
  local SLIDE="$1"
  local SLIDE_NAME=$(basename "$SLIDE" | sed 's/\.[^.]*$//')

  echo "============================================================"
  echo "Processing: $SLIDE_NAME"
  echo "============================================================"
  echo ""

  # -------------------------------------------------------------------------
  # Steps 1-2: Tissue Segmentation & Patch Extraction (HistoQC environment)
  # -------------------------------------------------------------------------
  echo "[1/2] Tissue segmentation & patch extraction..."
  conda run --no-capture-output -n histoqc_env python -u /app/tissue_segmentation_patches.py \
    -i "$SLIDE" \
    -o "$OUTPUT_DIR"
  echo ""

  # -------------------------------------------------------------------------
  # Steps 3-8: Feature Extraction & Prediction (APIC environment)
  # -------------------------------------------------------------------------
  echo "[2/2] Feature extraction & biomarker prediction..."
  conda run --no-capture-output -n apic_env python -u /app/feature_extraction_prediction.py \
    -i "$SLIDE" \
    -o "$OUTPUT_DIR"
  echo ""
}

# =============================================================================
# Main: Batch vs Single mode
# =============================================================================
if [[ -d "${INPUT_SLIDE}" ]]; then
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

# =============================================================================
# Complete
# =============================================================================
echo ""
echo "============================================================"
echo "  PIPELINE COMPLETE"
echo "============================================================"
echo ""
