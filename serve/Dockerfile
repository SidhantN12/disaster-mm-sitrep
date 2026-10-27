# CPU inference image for the multimodal SITREP service.
#
# Weights are NOT baked in: they are 1.2 GB and change every time the model is
# retrained. Mount them at /weights instead (see compose.yaml).
#
#   docker build -f serve/Dockerfile -t sitrep-serve .
#   docker run -p 8000:8000 -v $(pwd)/runs/base_fusion:/weights:ro sitrep-serve
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/hf \
    TORCH_HOME=/cache/torch \
    MODEL_DIR=/weights \
    CONFIG=configs/base.yaml \
    FUSION=hybrid \
    DEVICE=cpu

WORKDIR /app

# Dependencies first so a code change does not re-resolve the whole stack.
COPY serve/requirements.txt /app/serve/requirements.txt
RUN pip install -r /app/serve/requirements.txt

# Then the package and the configs it builds from.
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
COPY configs /app/configs
COPY serve /app/serve
RUN pip install -e . --no-deps

# The pretrained encoders are pulled once at build time so a cold start does
# not wait on a 1.1 GB download. This layer is what makes the image ~3 GB.
RUN python -c "import timm; from transformers import BertModel; \
    timm.create_model('vit_base_patch16_224', pretrained=True, in_chans=6, img_size=256, num_classes=0); \
    BertModel.from_pretrained('bert-base-uncased'); print('encoders cached')"

RUN useradd -m -u 1000 serve && chown -R serve:serve /app /cache
USER serve

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
