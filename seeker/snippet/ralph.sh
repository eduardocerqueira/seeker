#date: 2026-01-09T17:15:20Z
#url: https://api.github.com/gists/bd4c2e5e378b25dadfc007bc46aa5c77
#owner: https://api.github.com/users/roninjin10

#!/bin/bash
#
# Ralph Wiggum Autonomous Loop
#
# This script runs Claude Code in an infinite loop, feeding it the ralph wiggum
# prompt each iteration. The agent will autonomously progress through the
# implementation tasks, write reports, and create commits.
#
# Usage:
#   ./scripts/ralph.sh
#
# To stop:
#   Ctrl+C
#
# Monitor progress:
#   - Watch reports/ directory for new files
#   - Check git log for commits
#   - Read the terminal output
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
PROMPT_FILE="$REPO_DIR/prompts/012_ralph_wiggum_loop.md"

# Ensure we're in the repo directory
cd "$REPO_DIR"

# Check prompt file exists
if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "Error: Prompt file not found at $PROMPT_FILE"
    exit 1
fi

# Create reports directory if it doesn't exist
mkdir -p "$REPO_DIR/reports"

echo "=========================================="
echo "  Ralph Wiggum Autonomous Loop Starting"
echo "=========================================="
echo ""
echo "Prompt: $PROMPT_FILE"
echo "Reports: $REPO_DIR/reports/"
echo ""
echo "Press Ctrl+C to stop"
echo ""
echo "=========================================="

iteration=1

while true; do
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting iteration $iteration"
    echo "-------------------------------------------"

    # Run claude with the prompt
    # Using -p for print mode
    # Using --dangerously-skip-permissions for autonomous operation
    # WARNING: Only use in trusted environments!
    claude -p "$(cat "$PROMPT_FILE")" --dangerously-skip-permissions

    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Iteration $iteration complete"

    iteration=$((iteration + 1))

    # Small delay between iterations to prevent runaway loops
    sleep 2
done
