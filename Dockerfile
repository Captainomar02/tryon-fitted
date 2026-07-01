# syntax=docker/dockerfile:1

# Vast.ai supports any public Docker image. This base keeps CUDA, PyTorch,
# Jupyter/SSH integration, and the /venv/main Python environment aligned with
# Vast's recommended PyTorch templates.
FROM vastai/pytorch:2.6.0-cuda-12.6.3-py312

ENV DEBIAN_FRONTEND=noninteractive \
    APP_DIR=/workspace/sam3d-clad \
    IMAGE_APP_DIR=/opt/workspace-internal/sam3d-clad \
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch \
    PYOPENGL_PLATFORM=egl \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/venv/main/bin:${PATH}"

WORKDIR ${IMAGE_APP_DIR}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    git \
    libegl1 \
    libglib2.0-0 \
    libgl1 \
    libglvnd0 \
    libxext6 \
    libxrender1 \
    ninja-build \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-vast.txt ./requirements-vast.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install numpy cython \
    && python -m pip install xtcocotools --no-build-isolation \
    && python -m pip install -r requirements-vast.txt

COPY sam_3d_body ./sam_3d_body
COPY tools ./tools
COPY local_fit_tester ./local_fit_tester
COPY clad-body ./clad-body
COPY scripts ./scripts
COPY run_front_side_fusion.py ./run_front_side_fusion.py

RUN python -m pip install -e "./clad-body[mhr,render]" --no-build-isolation \
    && chmod +x scripts/vast/*.sh

RUN mkdir -p \
    /workspace/input \
    /workspace/output \
    /workspace/checkpoints \
    /workspace/.cache/huggingface \
    /workspace/.cache/torch \
    /opt/workspace-internal

CMD ["/bin/bash", "-l"]
