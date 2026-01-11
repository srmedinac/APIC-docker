#!/usr/bin/env bash
set -euo pipefail

REPO="madabhushilabapic/apic"

# Ask for version
read -p "Enter version (e.g., 1.0.5): v" VERSION

if [[ -z "$VERSION" ]]; then
  echo "Error: Version cannot be empty"
  exit 1
fi

VERSION_TAG="${REPO}:v${VERSION}"
LATEST_TAG="${REPO}:latest"

echo ""
echo "============================================================"
echo "  Building and pushing Docker image"
echo "============================================================"
echo "  Version tag: ${VERSION_TAG}"
echo "  Latest tag:  ${LATEST_TAG}"
echo "============================================================"
echo ""

# Build
echo "[1/4] Building image..."
docker build --platform linux/amd64 -t "${VERSION_TAG}" .

# Tag as latest
echo ""
echo "[2/4] Tagging as latest..."
docker tag "${VERSION_TAG}" "${LATEST_TAG}"

# Push version tag
echo ""
echo "[3/4] Pushing ${VERSION_TAG}..."
docker push "${VERSION_TAG}"

# Push latest tag
echo ""
echo "[4/4] Pushing ${LATEST_TAG}..."
docker push "${LATEST_TAG}"

echo ""
echo "============================================================"
echo "  Done! Pushed:"
echo "    - ${VERSION_TAG}"
echo "    - ${LATEST_TAG}"
echo "============================================================"
