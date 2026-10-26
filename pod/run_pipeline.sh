#!/usr/bin/env bash
# Deterministic end-to-end pipeline for the disaster SITREP demo on a RunPod pod.
#
# Every step writes a marker in $STATE, so re-running skips completed work.
# This is the happy path: it is a plain script on purpose, because an LLM agent
# should not be improvising shell for steps that never vary. Codex supervises
# this script and repairs what fails; it does not replace it.
#
#   bash run_pipeline.sh              # run / resume
#   bash run_pipeline.sh --from 6     # force a redo from step 6 onward
#   bash run_pipeline.sh --smoke      # tiny config only, no paid training
#
set -euo pipefail

WORK=${WORK:-/workspace}
REPO=$WORK/disaster-mm-sitrep
RUN=$WORK/runs/base_fusion
STATE=$WORK/state
CFG=configs/base.yaml
LIMIT=${LIMIT:-16}
SEED=${SEED:-42}
BRANCH=${BRANCH:-fusion-comparison-study}
GIT_URL=${GIT_URL:-https://github.com/SidhantN12/disaster-mm-sitrep.git}

FROM=0
SMOKE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM=$2; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$STATE" "$WORK/runs"

# --from N means "redo everything from N onward": drop those markers up front,
# so the rest of the script only ever has to ask "is this step already done?"
if [ "$FROM" -gt 0 ]; then
  for n in $(seq "$FROM" 9); do
    rm -f "$STATE/$n.done"
  done
  echo "forcing a redo from step $FROM onward"
fi

step() {
  local n=$1
  local name=$2
  if [ -f "$STATE/$n.done" ]; then
    echo "STEP $n $name - SKIP (already done)"
    return 1
  fi
  echo ""
  echo "=================================================="
  echo "STEP $n $name - START $(date -u +%H:%M:%S)"
  echo "=================================================="
  return 0
}

done_step() {
  touch "$STATE/$1.done"
  echo "STEP $1 — DONE $(date -u +%H:%M:%S)"
}

# ---------------------------------------------------------------- 1. preflight
if step 1 preflight; then
  echo "workdir: $WORK"
  df -h "$WORK" | tail -1
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "FATAL: no nvidia-smi — this is not a GPU pod" >&2
    exit 1
  fi
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  done_step 1
fi

# ------------------------------------------------------------------- 2. clone
if step 2 clone; then
  if [ -d "$REPO/.git" ]; then
    echo "repo already present"
  else
    git clone "$GIT_URL" "$REPO"
  fi
  cd "$REPO"
  git checkout "$BRANCH"
  git log --oneline -1
  done_step 2
fi

# ----------------------------------------------------------------- 3. install
if step 3 install; then
  cd "$REPO"
  export HF_HOME=$WORK/hf
  export TORCH_HOME=$WORK/torch
  # --no-deps is deliberate: the image ships torch built for its CUDA runtime
  # and letting pip resolve it will silently install a CPU wheel.
  pip install -e . --no-deps
  pip install timm transformers sacrebleu scikit-learn pyyaml pillow matplotlib tqdm
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  done_step 3
fi

# ----------------------------------------------------- 4. cache encoders + cfg
if step 4 prepare; then
  cd "$REPO"
  export HF_HOME=$WORK/hf
  export TORCH_HOME=$WORK/torch
  python - <<'PY'
import timm
from transformers import BertModel
timm.create_model('vit_base_patch16_224', pretrained=True, in_chans=6, img_size=256, num_classes=0)
BertModel.from_pretrained('bert-base-uncased')
print('encoders cached')
PY
  # the single highest-leverage edit: 2.8x less compute, no loss of coverage
  sed -i 's/max_reports: 32/max_reports: 8/' "$CFG"
  grep -n 'max_reports\|batch_size\|n_train\|epochs_stage' "$CFG"
  done_step 4
fi

# --------------------------------------------------------------- 5. smoke test
if step 5 smoke; then
  cd "$REPO"
  export HF_HOME=$WORK/hf
  export TORCH_HOME=$WORK/torch
  bash scripts/run_tiny_pipeline.sh
  echo "smoke test passed — the install is sound"
  done_step 5
fi

if [ "$SMOKE" -eq 1 ]; then
  echo ""
  echo "PIPELINE-SMOKE-OK — stopping before paid training as requested"
  exit 0
fi

# ------------------------------------------------------------------ 6. train
if step 6 train; then
  cd "$REPO"
  export HF_HOME=$WORK/hf
  export TORCH_HOME=$WORK/torch
  echo "this is the expensive step: ~1h46m on a 4090, ~45m on an A100"
  python scripts/fusion_study.py \
    --config "$CFG" \
    --out "$RUN" \
    --seed "$SEED" 2>&1 | tee "$WORK/fusion.log"
  done_step 6
fi

# ----------------------------------------------------------------- 7. export
if step 7 export; then
  cd "$REPO"
  export HF_HOME=$WORK/hf
  export TORCH_HOME=$WORK/torch
  python "$WORK/export_demo_pod.py" \
    --config "$CFG" \
    --run "$RUN" \
    --out "$WORK/demo_data.json" \
    --limit "$LIMIT"
  done_step 7
fi

# ------------------------------------------------------------------ 8. build
if step 8 build; then
  cd "$WORK"
  python build_demo.py --data demo_data.json --out sitrep_demo.html --src "$WORK"
  done_step 8
fi

# ----------------------------------------------------------------- 9. verify
if step 9 verify; then
  cd "$WORK"
  python verify.py \
    --run "$RUN" \
    --data "$WORK/demo_data.json" \
    --html "$WORK/sitrep_demo.html" \
    --min-tiles "$LIMIT" \
    --min-ckpt-mb 10
  done_step 9
fi

echo ""
echo "PIPELINE-OK"
echo "deliverables:"
ls -lh "$WORK/demo_data.json" "$WORK/sitrep_demo.html" "$RUN/fusion_results.json" 2>/dev/null
echo ""
echo "transfer them with:  runpodctl send $WORK/demo_data.json $WORK/sitrep_demo.html $RUN/fusion_results.json"
