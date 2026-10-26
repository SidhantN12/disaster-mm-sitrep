# AGENTS.md — disaster SITREP pipeline on a RunPod pod

You are operating a GPU pod to produce one deliverable: a working interactive
demo of a trained multimodal disaster-SITREP model, plus the numbers behind it.

## What this project is

Satellite imagery (pre/post, 6-channel 256×256 tiles) is fused with geotagged
field reports to generate a situation report, a damage class, and attention-based
evidence. Three fusion mechanisms are compared under identical training:

- **early** — concatenate both feature vectors, one MLP, feature level
- **late** — two independent networks, a learned scalar gate, decision level
- **hybrid** — bidirectional cross-attention at every layer

The hybrid variant is what the demo showcases. The other two exist to be
compared against it.

## The one-line objective

`bash /workspace/run_pipeline.sh` ends with `PIPELINE-OK`, and
`/workspace/demo_data.json` plus `/workspace/sitrep_demo.html` exist and pass
`verify.py`.

## Ground rules

**Never substitute a cheaper config to make something pass.** `configs/base.yaml`
with `n_train: 20000` is the target. Do not train `configs/tiny.yaml` and present
it as the result, do not lower `n_train`, and do not cut `epochs_stage2`. The
tiny config exists only for the step-5 smoke test. `verify.py --min-ckpt-mb 10`
catches this substitution; do not work around it by lowering the floor.

**Never fabricate a number, a metric, or a sample output.** If a step did not
run, say it did not run. Numbers in your final report must be copied from
`fusion.log`, `fusion_results.json` or `verify.py` output, never recalled or
estimated.

**These are findings, not bugs. Do not fix them:**

- Structure counts in generated SITREPs are wrong at every training budget.
  Measured across 1/2/3/6/12/24/48 epochs: count accuracy is 0.00 throughout
  while BLEU climbs to 64. The model learns the template and never learns to
  count. This is the project's headline result.
- Damage classification scores 0.837 weighted F1 under all three mechanisms —
  exactly the always-guess-"destroyed" baseline, because 83% of tiles carry
  that label. It is a label-definition problem, not an architecture problem.
- Early and late fusion collapse toward one sentence per tile. Expected.

If you find yourself tuning hyperparameters to improve any of these, stop. You
are destroying the result.

**Training is expensive and billed by the second.** One full run is ~1h46m on a
4090. You may launch step 6 at most **twice** total (once, plus one retry after
a diagnosed fix). If it fails twice, stop and report; do not keep trying.

## Known failure modes and their fixes

| Symptom | Fix |
|---|---|
| CUDA version mismatch after install | reinstall torch-free: `pip install -e . --no-deps` |
| `CUDA out of memory` in stage 1 | `batch_size: 16` in `configs/base.yaml`, rerun `--from 6` |
| GPU utilisation low, one CPU core pinned | the DataLoaders never pass `num_workers`, so the config key does nothing. Patch the `DataLoader(` calls in `scripts/fusion_study.py` to add `num_workers=8, pin_memory=True, persistent_workers=True` |
| `TypeError` on `in_chans` / `img_size` | `pip install -U timm` (needs ≥0.9) |
| `nan` loss in stage 3 | `lr_stage3: 0.00001` |
| `demo_data.json` enormous | lower `--limit` (default 16) |
| checkpoints vanished after restart | everything must live under `/workspace` |

### Known performance characteristics

This repo trains in **pure fp32**: there is no `autocast`, no `GradScaler`, and
TF32 is not enabled. Expect roughly 3-4× slower than a mixed-precision estimate
for the same card. Measured on an RTX-class card: about 9 minutes per stage-1
epoch at 20,000 tiles.

Two optional speedups, in order of likely impact. Apply them ONLY if the run has
not already started, or if you are restarting anyway. Never kill a healthy run
to apply them:

1. Pass `num_workers` (see the table above). Biggest win if GPU utilisation is
   below 50%, which means data generation is the bottleneck.
2. Enable TF32 by adding these two lines after `import torch` in
   `scripts/fusion_study.py`:
   `torch.backends.cuda.matmul.allow_tf32 = True`
   `torch.backends.cudnn.allow_tf32 = True`

A full fusion study at 9 min/epoch is about 6.9 hours: stage 1 is 20 epochs,
then stage 2 runs three times (once per variant) at roughly 0.43× the per-epoch
cost. That is slow but it completes overnight, so prefer letting a healthy run
finish over restarting it faster.

## Conventions

- All state lives under `/workspace`. The container disk is wiped on restart.
- `run_pipeline.sh` is resumable: markers in `/workspace/state/`. Prefer
  `bash run_pipeline.sh --from N` over editing the script.
- Do not rewrite `run_pipeline.sh`, `verify.py` or `export_demo_pod.py` to make
  a gate pass. Fix the underlying cause, or report that you could not.
- Long jobs run under `nohup ... &` with output tee'd to a log, never in the
  foreground of an interactive shell.
