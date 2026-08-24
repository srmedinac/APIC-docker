# APIC v2 — the published biomarker, rebuilt streaming.
#
# Same science as v1 (same HoVer-Net weights, same HistoQC + config_v3.ini, same nucdiv bins, v1's
# verbatim spaTIL, and a Cox model frozen from the published CHAARTED fit), but ~30x faster and a
# fraction of the size:
#   * v1 wrote every patch as JPEG and every mask as PNG and read them back three times; we stream
#     tiles and masks in memory and write only apic.json (+ a cached tissue mask).
#   * v1's spaTIL re-ran connected-component labelling ONCE PER NUCLEUS and Macenko-normalised each
#     tile only to discard the result; we label once and skip the unused normalisation.
#   * v1 shelled out to the MATLAB Runtime for the full 3264-feature nucdiv bank; we compute the 4
#     model features in Python. MATLAB is gone, which is most of the 39-55 GB.
# Result: ~1-2 GB image, ~75-350 s/slide vs ~1.4 h.
#
# HistoQC ships as v1's OWN env, copied wholesale from the v1 image. It is py3.8 / numpy 1.23 /
# skimage 0.19 while our runtime is py3.10 / numpy 2.2 — they cannot share an environment, and
# porting HistoQC to modern pins would change the tissue mask, which is the one artifact measured as
# fidelity-critical (the wrong-era HistoQC image recovered only 49 of v1's 80 tiles on a test slide).
# So it is vendored as a separate interpreter and invoked as a subprocess: no docker-in-docker, no
# mounted docker socket, byte-identical binary.
#
# Build/publish with v2/docker/build_and_push.sh (context = repo root). Run:
#   docker run --rm --gpus all -v /data/slides:/slides:ro -v /data/out:/out \
#     madabhushilabapic/apic:v2.0.0 --slide /slides/foo.svs --out /out/foo/apic.json

# ---- stage 1: v1's HistoQC environment, taken verbatim ---------------------------------------
# Pinned by digest, not by tag. The v1.0.4 tag MOVED between 2026-07-27 and 2026-08-24: the image
# this build originally used is sha256:475687400540..., which no longer resolves on the registry,
# and the tag now points at sha256:7685d691b9a2... with different contents. A moving tag under a
# build that copies a tissue-segmentation environment out of it is how a silent fidelity change
# gets in, and the tissue mask is the one artifact measured as fidelity-critical.
FROM madabhushilabapic/apic:v1.0.4@sha256:7685d691b9a29d2dbc4656ef238b745f64d1cc8cb438d304504b498f68500efa AS v1

# ---- stage 2: the runtime --------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        libopenslide0 openslide-tools libglib2.0-0 libgomp1 libgl1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# v1's HistoQC env. The destination path MUST stay /opt/conda/envs/histoqc_env — conda hardcodes its
# own prefix inside the env's scripts, so relocating it breaks the interpreter.
COPY --from=v1 /opt/conda/envs/histoqc_env /opt/conda/envs/histoqc_env
# v1 assets the pipeline calls directly: HoVer-Net weights + net definition, PanNuke types, QC config.
COPY --from=v1 /app/models/hovernet_fast_pannuke_type_tf2pytorch.tar /opt/apic/v1/models/
COPY --from=v1 /app/types/type_info_pannuke.json                     /opt/apic/v1/types/
# nucleusSegmentationTiles.py comes from THIS repo, not from the v1 image. Every v1 image on the
# registry carries a copy that predates render_pred_info_mask, which hovernet_stream.py calls:
# v1.0.4 as it was in July is sha256 9868087b..., v1.0.4 as the tag stands now is f1a378a7..., and
# neither defines the function. The copy that does is this repo's src/, sha256 0060103d..., which
# is also what apic:latest carries and what the viewer runs.
COPY src/nucleusSegmentationTiles.py                                 /opt/apic/v1/src/
COPY --from=v1 /app/src/config_v3.ini                                /opt/apic/v1/src/
COPY --from=v1 /app/data/HUG-REPORT-EMPTY-POSITIVE.pdf /app/data/HUG-REPORT-EMPTY-NEGATIVE.pdf /opt/apic/v1/data/
RUN /opt/conda/envs/histoqc_env/bin/python -m histoqc --help >/dev/null   # fail the build, not a run

# ---- our pipeline ----------------------------------------------------------------------------
COPY v2/docker/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

COPY v2/src     /opt/apic/src
COPY v2/scripts /opt/apic/scripts
COPY v2/models  /opt/apic/models

# apic_one_slide.py resolves its imports and the frozen Cox model relative to its own location, so
# this layout needs no flags. APIC_DOCKER_DIR is the only hardcoded host path in the runtime.
ENV APIC_DOCKER_DIR=/opt/apic/v1 \
    APIC_HISTOQC_PYTHON=/opt/conda/envs/histoqc_env/bin/python \
    APIC_HISTOQC_CONFIG=/opt/apic/v1/src/config_v3.ini \
    APIC_TEMPLATE_DIR=/opt/apic/v1/data \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /work
ENTRYPOINT ["python3", "/opt/apic/scripts/apic_one_slide.py"]
CMD ["--help"]
