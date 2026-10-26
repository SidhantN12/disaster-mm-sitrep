#!/usr/bin/env bash
# Stage the pipeline bundle on a RunPod pod and install Codex to supervise it.
#
#   export OPENAI_API_KEY=sk-...
#   bash bootstrap_codex.sh
#
# Run this from the directory where you extracted sitrep_codex_bundle.tar.gz.
set -euo pipefail

WORK=${WORK:-/workspace}
HERE=$(cd "$(dirname "$0")" && pwd)

echo "== staging the bundle into $WORK =="
mkdir -p "$WORK"
for f in run_pipeline.sh verify.py export_demo_pod.py build_demo.py \
         page_head.html page_body.html page_script.js AGENTS.md TASK.md; do
  if [ ! -f "$HERE/$f" ]; then
    echo "FATAL: $f missing from the bundle" >&2
    exit 1
  fi
  cp "$HERE/$f" "$WORK/$f"
done
chmod +x "$WORK/run_pipeline.sh"
ls -1 "$WORK"

echo ""
echo "== installing codex =="
if ! command -v node >/dev/null 2>&1; then
  echo "node not found; installing 22.x"
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
  apt-get install -y nodejs
fi
node --version

if ! command -v codex >/dev/null 2>&1; then
  npm install -g @openai/codex
fi
codex --version

echo ""
echo "== auth =="
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "WARNING: OPENAI_API_KEY is not set."
  echo "Either export it, or run 'codex login' and follow the device flow."
else
  echo "OPENAI_API_KEY is set (${#OPENAI_API_KEY} chars)"
fi

echo ""
echo "== writing ~/.codex/config.toml =="
mkdir -p ~/.codex
cat > ~/.codex/config.toml <<'TOML'
# The pod is disposable and has nothing on it worth protecting, so full access
# is the right trade here. Never use these settings on your own machine.
approval_policy = "never"
sandbox_mode = "danger-full-access"

# Uncomment and set to whichever model your account has:
# model = "gpt-5-codex"
TOML
cat ~/.codex/config.toml

echo ""
echo "== CONFIRM THE FLAGS BEFORE YOU RELY ON THEM =="
echo "Codex's CLI flags change between releases. Check these against the help"
echo "text below and adjust launch_codex.sh if they differ:"
echo "  - subcommand for non-interactive runs (expected: 'codex exec')"
echo "  - the bypass flag (expected: --dangerously-bypass-approvals-and-sandbox)"
echo ""
codex exec --help 2>&1 | head -40 || codex --help 2>&1 | head -40

echo ""
echo "BOOTSTRAP-OK"
echo "next:  bash $WORK/run_pipeline.sh --smoke     # verify by hand first"
echo "then:  bash launch_codex.sh                   # hand it to the agent"
