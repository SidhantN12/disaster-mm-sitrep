# Inference service

Serves single-tile SITREP generation over HTTP. Loads the checkpoints once at
startup, not per request.

The response carries the **evidence** alongside the prediction: which report
each claim leaned on (MIL attention weights) and how each generated word split
its attention between imagery and text. In this domain a SITREP a responder
cannot check is not actionable, so the attribution is part of the contract.

## Run it

Unpack the trained weights first — `base_fusion.tar.gz` gives you
`runs/base_fusion/` with `part1_stage1.pt`, `model_hybrid.pt` and friends.

```bash
docker compose -f serve/compose.yaml up --build
```

Or directly:

```bash
docker build -f serve/Dockerfile -t sitrep-serve .
docker run -p 8000:8000 -v "$(pwd)/runs/base_fusion:/weights:ro" sitrep-serve
```

Without Docker, from the repo root:

```bash
MODEL_DIR=runs/base_fusion CONFIG=configs/base.yaml FUSION=hybrid \
  uvicorn serve.app:app --port 8000
```

## Endpoints

`GET /health` — readiness, which variant is loaded, parameter count.

`POST /sitrep`

```json
{
  "reports": [
    {"text": "people trapped under the warehouse on Mill Lane",
     "dx": 0.08, "dy": -0.04, "dt": 1.2}
  ],
  "image_npy_b64": null
}
```

`dx`/`dy` are km from the tile centre, `dt` is hours from the event.
`image_npy_b64` is a base64 `.npy` holding a `(6, 256, 256)` float array —
pre-event RGB stacked on post-event RGB. Omit it and a synthetic tile is
generated, so the endpoint is callable with no inputs at all.

Interactive docs at `/docs`.

## Configuration

| Variable | Default | |
| --- | --- | --- |
| `MODEL_DIR` | `/weights` | directory holding the checkpoints |
| `CONFIG` | `configs/base.yaml` | model config to build from |
| `FUSION` | `hybrid` | `early`, `late` or `hybrid` |
| `DEVICE` | `cpu` | `cpu` or `cuda` |

`FUSION` lets one image serve any of the three variants, which is how the
comparison is reproduced without rebuilding.

## Design notes

**Weights are mounted, not baked.** They are 1.2 GB and change on every
retrain. Baking them would mean rebuilding the image for each new checkpoint
and shipping 1.2 GB on every pull.

**The pretrained encoders *are* baked.** ViT-B/16 and BERT-base are pulled at
build time so a cold start does not wait on a 1.1 GB download. This is most of
the image size and the trade is deliberate: build once, start fast.

**CPU by default.** The CUDA torch wheel is ~2.5 GB and inference does not need
it — measured at 155 ms per tile on a GPU, and latency here is not the binding
constraint anyway (imagery arrives hours after an event). For GPU serving, swap
the base image for a CUDA runtime, drop the `--index-url` line in
`requirements.txt`, and set `DEVICE=cuda`.

## What this does not do

It returns predictions, not decisions. Two of the four output fields are not
accurate enough to act on: damage classification is a majority-class predictor
and structure counts are not learned. Report relevance and needs detection are
the fields that carry real signal. See `results/fusion_results.json`.
