# ─────────────────────────────────────────────────────────────
# Backend image — CUDA 12.8 runtime + Python 3.11 + FastAPI
# CUDA 12.8 is required for Blackwell GPUs (RTX 50-series, sm_120).
# ─────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System deps + Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    git \
    build-essential \
    cmake \
    libgomp1 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch (CUDA 12.8 build) first so the layer is cached.
# torch 2.7+ on the cu128 index ships sm_120 kernels for Blackwell GPUs.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel \
    && pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# ─────────────────────────────────────────────────────────────
# llama.cpp — for exporting trained models to GGUF (Ollama / llama.cpp).
# We need two things: the Python converter `convert_hf_to_gguf.py` (which uses
# the in-repo, always-in-sync gguf-py) and the `llama-quantize` binary. The
# conversion + quantization run on CPU, so a plain (non-CUDA) build is enough.
#
# NOTE: we intentionally do NOT `pip install` llama.cpp's own
# requirements-convert_hf_to_gguf.txt — it pins torch/transformers ranges that
# could clobber our cu128 GPU torch. The converter's runtime deps (torch,
# transformers, numpy, sentencepiece, protobuf) are already installed above;
# `gguf-py` (installed editable) provides the rest.
#
# GGML_NATIVE=ON: build with -march=native so `llama-quantize` uses the host
# CPU's SIMD (AVX2 / AVX-512 / FMA). Quantization is heavily vectorised; the old
# GGML_NATIVE=OFF build fell back to scalar code and was several times slower for
# an identical, bit-for-bit GGUF result. Because we `docker compose build` on the
# same machine we run on, -march=native is safe here. (If you ever build this
# image on one CPU and run it on an older one, switch to explicit portable flags
# instead, e.g. -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON.)
# ─────────────────────────────────────────────────────────────
ENV LLAMA_CPP_DIR=/opt/llama.cpp
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR" \
    && cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" \
    -DLLAMA_CURL=OFF -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release \
    && cmake --build "$LLAMA_CPP_DIR/build" --config Release -j --target llama-quantize \
    && pip install -e "$LLAMA_CPP_DIR/gguf-py"

# Install the remaining Python dependencies
COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /app/requirements.txt

# Application source is mounted as a volume in docker-compose,
# but we copy it here too so the image is runnable standalone.
COPY . /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
