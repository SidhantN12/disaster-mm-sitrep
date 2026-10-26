# SITREP pipeline bundle — autonomous RunPod run

Everything needed to take a fresh GPU pod to a working demo, with Codex
supervising. Upload this archive to the pod, extract, and run two commands.

## Quick start

```bash
# on the pod
cd /workspace
tar xzf sitrep_codex_bundle.tar.gz
cd sitrep_codex_bundle

export OPENAI_API_KEY=sk-...
bash bootstrap_codex.sh          # stages files, installs codex, prints its real flags

bash /workspace/run_pipeline.sh --smoke   # ~20 min, no paid training, do this yourself
bash launch_codex.sh                      # hand the rest to the agent
tail -f /workspace/codex.log
```

## What is in here

| File | Role |
|---|---|
| `run_pipeline.sh` | the 9-step pipeline, resumable via markers in `/workspace/state/` |
| `verify.py` | hard gates; exits non-zero with a specific reason |
| `export_demo_pod.py` | runs the trained models over the test split, writes `demo_data.json` |
| `build_demo.py` | inlines that JSON into one self-contained HTML page |
| `page_head.html`, `page_body.html`, `page_script.js` | the demo page source |
| `AGENTS.md` | project context, ground rules, failure table — Codex reads this |
| `TASK.md` | the prompt handed to Codex |
| `bootstrap_codex.sh` | stages the bundle, installs Codex, writes its config |
| `launch_codex.sh` | runs Codex detached against `TASK.md` |

## The division of labour

`run_pipeline.sh` is a plain shell script, deliberately. Steps that never vary
should not be improvised by a language model every time. Codex's value is the
other half: diagnosing an OOM at 23:00, applying the right fix from the table
in `AGENTS.md`, and resuming with `--from 6` while you sleep.

If you are awake and watching, you do not need Codex at all — run
`bash run_pipeline.sh` and read the log. The agent earns its place only when
nobody is watching.

## Verify it yourself before you trust the report

The agent writes `/workspace/REPORT.md`. Do not take it at face value. Run:

```bash
python /workspace/verify.py \
  --run /workspace/runs/base_fusion \
  --data /workspace/demo_data.json \
  --html /workspace/sitrep_demo.html \
  --min-tiles 16 --min-ckpt-mb 10
```

`VERIFY-OK` is the only claim worth believing. In particular `--min-ckpt-mb 10`
catches the most likely silent failure: training the tiny config because it is
faster, then reporting success.

## Flags that may have moved

Codex's CLI changes between releases. `bootstrap_codex.sh` prints
`codex exec --help` at the end for exactly this reason. If the subcommand or
the bypass flag differs, override them without editing anything:

```bash
SUBCMD=exec BYPASS=--full-auto bash launch_codex.sh
```

`approval_policy = "never"` and `sandbox_mode = "danger-full-access"` are
appropriate on a disposable pod with nothing on it worth protecting. They are
not appropriate on your laptop.
