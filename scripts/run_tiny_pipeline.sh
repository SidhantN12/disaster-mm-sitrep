#!/usr/bin/env bash
# Runs the full tiny (CPU, synthetic-data) pipeline: all three training
# stages, then evaluation with modality ablation. Takes a few minutes.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m disaster_mm.train --config configs/tiny.yaml --stage 1
python -m disaster_mm.train --config configs/tiny.yaml --stage 2
python -m disaster_mm.train --config configs/tiny.yaml --stage 3
python -m disaster_mm.evaluate --config configs/tiny.yaml --stage 3 --out runs/tiny/metrics.json
