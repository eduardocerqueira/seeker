#date: 2026-02-05T17:37:11Z
#url: https://api.github.com/gists/160fed9519ec745b1883c17353f6745b
#owner: https://api.github.com/users/xnorkl

#!/usr/bin/env bash
#
# add-rulesets.sh -- Add newtworthy/agent-rulesets as a git submodule.
#
# Usage:
#   # Standalone (defaults to .cursor/rules)
#   ./utils/add-rulesets.sh
#
#   # Custom path
#   ./utils/add-rulesets.sh .rules
#
#   # Gist / curl pipe
#   curl -fsSL <raw-url> | bash
#
#   # With custom path via curl
#   curl -fsSL <raw-url> | bash -s -- .rules
#
#   # Shell alias
#   alias add-rulesets='curl -fsSL <raw-url> | bash'

set -euo pipefail

REMOTE_URL="git@github.com:newtworthy/agent-rulesets.git"
TARGET_PATH="${1:-.cursor/rules}"

# --- Validate environment ---------------------------------------------------

if ! command -v git >/dev/null 2>&1; then
  printf "error: git is not installed or not in PATH\n" >&2
  exit 1
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  printf "error: current directory is not inside a git repository\n" >&2
  exit 1
fi

# --- Guard against duplicates ------------------------------------------------

if git config --file .gitmodules --get-regexp "submodule\..*\.path" 2>/dev/null \
   | awk '{print $2}' | grep -qx "$TARGET_PATH"; then
  printf "submodule already exists at '%s' -- nothing to do\n" "$TARGET_PATH"
  exit 0
fi

# --- Add submodule -----------------------------------------------------------

printf "adding agent-rulesets submodule at '%s'...\n" "$TARGET_PATH"
git submodule add "$REMOTE_URL" "$TARGET_PATH"

printf "initializing submodule...\n"
git submodule update --init --recursive

printf "done -- submodule installed at '%s'\n" "$TARGET_PATH"
printf "next: commit .gitmodules and the submodule entry\n"
