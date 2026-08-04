FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY data/synthetic/seeds ./data/synthetic/seeds

# Plain `pip install` (no special CUDA index) resolves to CPU-only torch/
# onnxruntime here deliberately - this image is meant to run anywhere,
# not just on a machine with a matching NVIDIA driver. See docs/model.md
# for the GPU-training setup, which is a separate (non-containerized)
# workflow in this MVP.
RUN pip install --no-cache-dir ".[train,outbreak,api]"

ENV SPAMDET_MODEL_DIR=/app/models/spamdet-mdeberta-onnx
ENV SPAMDET_REDIS_URL=redis://redis:6379/0
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "spamdet.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
