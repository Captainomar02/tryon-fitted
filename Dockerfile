# syntax=docker/dockerfile:1

# Environment-only image for Vast.ai. The project source is cloned at rental
# startup so code, inputs, outputs, and checkpoints all live in one repo folder.
FROM vastai/pytorch:2.6.0-cuda-12.6.3-py312

ENV DEBIAN_FRONTEND=noninteractive \
    APP_DIR=/workspace/tryon-fitted \
    APP_REPO_URL=https://github.com/Captainomar02/tryon-fitted.git \
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch \
    SAM3D_SEGMENTOR=sam2 \
    SAM3D_SEGMENTOR_PATH=/workspace/tryon-fitted/external/sam2 \
    SAM2_DIR=/workspace/tryon-fitted/external/sam2 \
    SAM2_REF=main \
    SAM2_BUILD_CUDA=0 \
    FUSION_SIDE_SDF_CHEST_MODE=row_sdf \
    FUSION_SIDE_SDF_CHEST_LOBE_GAIN=2.4 \
    PYOPENGL_PLATFORM=egl \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/venv/main/bin:${PATH}"

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    libegl1 \
    libglib2.0-0 \
    libgl1 \
    libglvnd0 \
    libxext6 \
    libxrender1 \
    unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-vast.txt /tmp/requirements-vast.txt

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -r /tmp/requirements-vast.txt \
    && rm /tmp/requirements-vast.txt

RUN python -c 'import torch, torchvision; import pymomentum.geometry; import mhr; print(f"CLAD deps ok: torch {torch.__version__}, torchvision {torchvision.__version__}")'

RUN mkdir -p \
    /workspace/input \
    /workspace/output \
    /workspace/checkpoints \
    /workspace/.cache/huggingface \
    /workspace/.cache/torch \
    /opt/workspace-internal

CMD ["/bin/bash", "-l"]
