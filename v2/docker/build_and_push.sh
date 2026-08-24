#!/usr/bin/env bash
# Build + publish the APIC v2 image (mirrors APIC-docker/build_and_push.sh so the workflow is familiar).
#
#   ./docker/build_and_push.sh              # prompts, defaults to the version below
#   ./docker/build_and_push.sh 2.0.1        # non-interactive
#   PUSH_LATEST=1 ./docker/build_and_push.sh 2.0.2   # ALSO move :latest to this build
#
# Publishes madabhushilabapic/apic:vX.Y.Z only. :latest is left alone.
# NOTE ON :latest. v2 is a different pipeline from v1. It streams, it drops MATLAB, it vendors
# HistoQC, and its CLI arguments differ. :latest holds the stable v1 release, and everything that
# pulls it expects v1, so moving it is opt-in and deliberate. Publish the version tag, verify it,
# then move :latest by hand if that is what you want.
set -euo pipefail

REPO="${APIC_REPO:-madabhushilabapic/apic}"
DEFAULT_VERSION="2.0.0"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root (APIC-docker)

VERSION="${1:-}"
if [[ -z "$VERSION" ]]; then
  read -r -p "Enter version (default ${DEFAULT_VERSION}): v" VERSION
  VERSION="${VERSION:-$DEFAULT_VERSION}"
fi
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "Version must look like 2.0.0 (got '$VERSION')"; exit 1; }

VERSION_TAG="${REPO}:v${VERSION}"
LATEST_TAG="${REPO}:latest"
PUSH_LATEST="${PUSH_LATEST:-0}"

echo "============================================================"
echo "  APIC v2 image"
echo "    version tag : ${VERSION_TAG}"
echo "    latest tag  : ${LATEST_TAG} $([[ "$PUSH_LATEST" == "1" ]] && echo '(will MOVE to this build)' || echo '(untouched, stays on v1)')"
echo "    context     : ${ROOT}"
echo "============================================================"

# The build COPYs v1's HistoQC env + HoVer-Net weights out of the v1 image, so it must be present.
if ! docker image inspect madabhushilabapic/apic:v1.0.4 >/dev/null 2>&1; then
  echo "[0/4] pulling madabhushilabapic/apic:v1.0.4 (source of HistoQC + HoVer-Net weights)..."
  docker pull madabhushilabapic/apic:v1.0.4
fi

echo "[1/4] building..."
docker build --platform linux/amd64 -f "${ROOT}/v2/docker/Dockerfile" -t "${VERSION_TAG}" "${ROOT}"

echo "[2/4] smoke test (frozen model loads, HistoQC present, torch sees CUDA if a GPU is attached)..."
docker run --rm --entrypoint python3 "${VERSION_TAG}" -c "
import json, subprocess, sys
m = json.load(open('/opt/apic/models/apic_cox_frozen.json'))
assert len(m['features']) == 6, m['features']
print('  frozen Cox:', m['model'], '| threshold %.6f' % m['threshold'])
subprocess.run(['/opt/conda/envs/histoqc_env/bin/python','-m','histoqc','--help'],
               check=True, capture_output=True); print('  histoqc: OK')
import torch; print('  torch', torch.__version__, '| cuda build', torch.version.cuda)
"
# Retag :latest only when it is being published. An unconditional tag moves the local
# :latest off the v1 image, which is what the viewer runs.
if [[ "$PUSH_LATEST" == "1" ]]; then
  docker tag "${VERSION_TAG}" "${LATEST_TAG}"
fi
echo "  size: $(docker image inspect "${VERSION_TAG}" --format '{{.Size}}' | awk '{printf "%.1f GB", $1/1e9}')"

echo "[3/4] pushing ${VERSION_TAG}..."
docker push "${VERSION_TAG}"

if [[ "$PUSH_LATEST" == "1" ]]; then
  echo "[4/4] pushing ${LATEST_TAG}..."
  docker push "${LATEST_TAG}"
else
  echo "[4/4] :latest untouched, still the stable v1 release (set PUSH_LATEST=1 to move it)"
fi

echo "============================================================"
echo "  Done: ${VERSION_TAG}$([[ "$PUSH_LATEST" == "1" ]] && echo " + ${LATEST_TAG}")"
echo "  Run:  docker run --rm --gpus all -v /slides:/slides:ro -v /out:/out \\"
echo "          ${VERSION_TAG} --slide /slides/foo.svs --out /out/foo/apic.json"
echo "============================================================"
