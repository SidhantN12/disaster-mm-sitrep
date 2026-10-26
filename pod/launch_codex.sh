#!/usr/bin/env bash
# Hand the pipeline to Codex and detach, so it survives your SSH session.
#
#   bash launch_codex.sh
#   tail -f /workspace/codex.log
#
# If `codex exec` or the bypass flag errored during bootstrap, fix the two
# variables below to match what `codex exec --help` actually prints.
set -euo pipefail

WORK=${WORK:-/workspace}
SUBCMD=${SUBCMD:-exec}
BYPASS=${BYPASS:---dangerously-bypass-approvals-and-sandbox}

cd "$WORK"

if [ ! -f "$WORK/TASK.md" ]; then
  echo "FATAL: $WORK/TASK.md missing - run bootstrap_codex.sh first" >&2
  exit 1
fi

# strip the markdown preamble: everything after the '---' line is the prompt
PROMPT=$(awk 'f{print} /^---$/{f=1}' "$WORK/TASK.md")
if [ -z "$PROMPT" ]; then
  echo "FATAL: could not extract the prompt from TASK.md" >&2
  exit 1
fi

echo "== launching codex =="
echo "prompt is ${#PROMPT} chars"
echo "log: $WORK/codex.log"

nohup codex "$SUBCMD" $BYPASS "$PROMPT" > "$WORK/codex.log" 2>&1 &
echo "codex pid $!"
echo ""
echo "watch it with:  tail -f $WORK/codex.log"
echo "stop it with:   kill $!"
