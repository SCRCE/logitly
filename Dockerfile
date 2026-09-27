FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/logitly-venv \
    PATH=/opt/logitly-venv/bin:$PATH \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/workspace/.cache/huggingface \
    HF_HUB_CACHE=/workspace/.cache/huggingface/hub \
    TORCH_HOME=/workspace/.cache/torch \
    MPLCONFIGDIR=/workspace/.cache/matplotlib \
    LOGITLY_PROJECT_ROOT=/workspace

WORKDIR /workspace
RUN apt-get update \
    && apt-get install --yes --no-install-recommends python3.12-venv \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml requirements.lock README.md README.pypi.md ./
RUN /usr/bin/python -m venv --system-site-packages "$VIRTUAL_ENV" \
    && python -m pip install --no-cache-dir -r requirements.lock
COPY src ./src
RUN python -m pip install --no-cache-dir --no-deps -e .

ENTRYPOINT ["python", "-m", "logitly"]
CMD ["--help"]
