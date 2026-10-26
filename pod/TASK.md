# Codex task prompt

Paste this as the argument to `codex exec`, or keep it as a file and use
`codex exec "$(cat /workspace/TASK.md)"`.

---

Operate this pod to completion. Read /workspace/AGENTS.md first and follow it;
it contains the ground rules, the known failure modes, and three measured
findings you must not try to "fix".

FIRST: ESTABLISH WHAT IS ALREADY RUNNING
Work may already be underway and you must not start a second training job on
the same GPU. Before anything else, determine the state and report it in one
short paragraph:

    ps aux | grep -E "fusion_study|disaster_mm.train|run_pipeline" | grep -v grep
    ls -la /workspace/state/ 2>/dev/null
    ls -la /workspace/runs/ 2>/dev/null
    tail -30 /workspace/pipeline.log 2>/dev/null
    tail -30 /workspace/fusion.log 2>/dev/null
    nvidia-smi

Read the epoch counter to tell the two configs apart: the tiny smoke-test
config prints `epoch N/3`, the real base config prints `epoch N/20`. A run
showing `/3` is the smoke test and is NOT the real training run.

Then branch:
- A base-config training process is running and healthy (loss falling, no OOM)
  -> DO NOT restart or relaunch it. Poll its log every few minutes until it
  finishes, then continue from step 7 (export) with
  `bash /workspace/run_pipeline.sh --from 7`. Expect roughly 6.9 hours total
  from its start; this is normal for this repo, which trains in fp32.
- Only the tiny smoke test is running or finished -> let it finish, then start
  the real run as described below.
- Nothing is running -> start from the beginning below.

OBJECTIVE
Run `bash /workspace/run_pipeline.sh` through to `PIPELINE-OK`, so that
/workspace/demo_data.json and /workspace/sitrep_demo.html exist and
/workspace/verify.py exits 0 against them.

HOW TO WORK
1. If no smoke test has passed yet, run `bash /workspace/run_pipeline.sh
   --smoke` first. It stops after the step-5 smoke test, before any paid GPU
   training. Do not proceed until it prints PIPELINE-SMOKE-OK. If a smoke test
   already passed, skip this.
2. Then run `bash /workspace/run_pipeline.sh` to completion. Step 6 is the
   expensive one (~1h46m on a 4090). Launch it detached and poll the log
   rather than blocking:
       nohup bash /workspace/run_pipeline.sh > /workspace/pipeline.log 2>&1 &
   Poll with `tail -40 /workspace/pipeline.log` every few minutes.
3. Within the first 3 minutes of step 6, confirm all three of:
   - `nvidia-smi` shows utilisation above 85%
   - the stage-1 loss is falling between epoch 1 and epoch 2
   - no CUDA OOM traceback
   If any fails, kill the job, apply the matching fix from AGENTS.md, and
   relaunch with `--from 6`. This counts as your one retry.
4. When a step fails, diagnose from the log, apply the smallest fix that
   addresses the cause, and resume with `--from N`. Do not restart from zero.

BUDGET AND STOP CONDITIONS
- Launch step 6 at most twice in total. If it fails twice, STOP and report.
- Do not spend more than 30 minutes of wall-clock on any single non-training
  step. If you are stuck that long, STOP and report what you tried.
- If a fix requires changing `verify.py`, `export_demo_pod.py` or
  `run_pipeline.sh` to make a gate pass, STOP. That is a signal the run is
  genuinely bad, not that the checker is wrong.

DO NOT
- Do not train `configs/tiny.yaml` and present it as the result.
- Do not reduce `n_train`, `epochs_stage2`, or `--min-ckpt-mb` to make
  something finish or pass.
- Do not tune hyperparameters to improve count accuracy, damage-class F1, or
  early/late fusion diversity. Those outcomes are the findings.
- Do not invent numbers. Every figure you report must be copied from
  fusion.log, fusion_results.json, or verify.py output.
- Do not kill a training run that is healthy just because it is slow. 9 minutes
  per stage-1 epoch is expected here. Slow is not broken.
- Do not delete or stop the pod.

WHEN DONE
Write /workspace/REPORT.md containing exactly these sections:

1. **Outcome** — PIPELINE-OK or the step number you stopped at.
2. **What ran** — each step, its wall-clock duration, from the log timestamps.
3. **Numbers** — the constant-predictor baselines and the per-variant results,
   copied verbatim from fusion_results.json. State the weighted F1 for damage
   classification under each of the three fusion mechanisms.
4. **Sample output** — the hybrid model's generated SITREP for the first three
   test tiles, copied from demo_data.json, each next to its ground truth.
5. **Verify output** — the full stdout of the final verify.py run.
6. **Deviations** — every change you made to any config or file, and why. If
   you made none, write "none".
7. **What I could not do** — anything you were asked for and did not achieve.
   Write "nothing" only if that is true.

Then print the exact `runpodctl send` command for demo_data.json,
sitrep_demo.html, fusion_results.json and REPORT.md, and stop.
