FROM containers.mathworks.com/matlab-runtime:r2025a

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# -------------------------------------------------------------------
# MATLAB Runtime (already present in base image) + baked env vars
# -------------------------------------------------------------------
ENV AGREE_TO_MATLAB_RUNTIME_LICENSE=yes
ENV MATLAB_RUNTIME_ROOT=/opt/matlabruntime/R2025a
ENV LD_LIBRARY_PATH=/opt/matlabruntime/R2025a/runtime/glnxa64:/opt/matlabruntime/R2025a/bin/glnxa64:/opt/matlabruntime/R2025a/sys/os/glnxa64:/opt/matlabruntime/R2025a/sys/opengl/lib/glnxa64

# -------------------------------------------------------------------
# System packages + R + precompiled R packages via apt (speed)
# -------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    openslide-tools \
    libglib2.0-0 \
    build-essential \
    curl \
    ca-certificates \
    r-base \
    r-cran-survival \
    r-cran-tidyverse \
    r-cran-caret \
    r-cran-glmnet \
    && rm -rf /var/lib/apt/lists/*
# -------------------------------------------------------------------
# Miniconda
# -------------------------------------------------------------------
ENV CONDA_DIR=/opt/conda
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm -f /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# -------------------------------------------------------------------
# Conda config: flexible priority (avoids strict-priority solve failures)
# + ToS acceptance for non-interactive builds
# -------------------------------------------------------------------
RUN conda config --set channel_priority flexible && \
    conda config --remove-key channels || true && \
    conda config --add channels conda-forge && \
    conda config --add channels defaults && \
    conda config --set always_yes yes && \
    conda config --set auto_activate_base false && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# -------------------------------------------------------------------
# Workdir
# -------------------------------------------------------------------
WORKDIR /app

# -------------------------------------------------------------------
# Cache-friendly: copy env YAMLs first, build envs, then copy rest
# -------------------------------------------------------------------
COPY envs/ /app/envs/

RUN conda env create -f envs/histoqc_env.yml && \
    conda env create -f envs/apic_env.yml && \
    conda clean -afy

RUN conda run -n histoqc_env pip install --no-deps git+https://github.com/choosehappy/HistoQC.git

# Now copy the remaining project files (code changes won't rebuild envs)
COPY . /app

# Optional debug
RUN conda env list

# Make run script executable
RUN chmod +x /app/run_pipeline_docker.sh
RUN chmod +x /app/src/nucdiv_executable_pipeline
ENTRYPOINT ["/bin/bash", "/app/run_pipeline_docker.sh"]
