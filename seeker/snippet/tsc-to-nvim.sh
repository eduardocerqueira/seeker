#date: 2025-10-03T16:52:05Z
#url: https://api.github.com/gists/0281bc3564e8edfbffe451e19a2989a0
#owner: https://api.github.com/users/AlexBeauchemin

#!/usr/bin/env bash

# Send TypeScript compiler output directly to Neovim's quickfix list from the command line.

### Installation

# chmod +x tsc-to-nvim.sh
# sudo ln -s $(pwd)/tsc-to-nvim.sh /usr/local/bin/tsc-to-nvim

## Add to your shell config (~/.bashrc or ~/.zshrc):
# export NVIM_LISTEN_ADDRESS=/tmp/nvim.sock

##  And always start Neovim with nvim --listen /tmp/nvim.sock, or create an alias:
# alias nvim='nvim --listen /tmp/nvim.sock'

### Usage
# tsc-to-nvim tsc
# tsc-to-nvim pnpm build
# tsc-to-nvim npm run typecheck

set -euo pipefail

NVIM_SOCKET="${NVIM_LISTEN_ADDRESS:-${NVIM:-}}"

if [ -z "$NVIM_SOCKET" ]; then
  echo "Error: No Neovim instance found. Start Neovim with: nvim --listen /tmp/nvim.sock" >&2
  echo "Or set NVIM_LISTEN_ADDRESS environment variable" >&2
  exit 1
fi

if [ $# -eq 0 ]; then
  echo "Usage: tsc-to-nvim <command> [args...]" >&2
  echo "Example: tsc-to-nvim tsc" >&2
  echo "Example: tsc-to-nvim pnpm build" >&2
  echo "Example: tsc-to-nvim npm run typecheck" >&2
  exit 1
fi

echo "Running: $*"
COMMAND_OUTPUT=$("$@" 2>&1 || true)

echo "$COMMAND_OUTPUT"

if [ -z "$COMMAND_OUTPUT" ]; then
  nvim --server "$NVIM_SOCKET" --remote-expr "setqflist([], 'r') | echo 'No errors found'" >/dev/null
  exit 0
fi

TEMP_FILE=$(mktemp)
echo "$COMMAND_OUTPUT" | grep -E '^\S.*\([0-9]+,[0-9]+\):' >"$TEMP_FILE" || true

if [ -s "$TEMP_FILE" ]; then
  nvim --server "$NVIM_SOCKET" --remote-expr \
    "setqflist([], 'r', {'lines': readfile('$TEMP_FILE'), 'efm': '%+A %#%f %#(%l\,%c): %m,%C%m'}) | echo 'Errors loaded to quickfix'" >/dev/null
else
  nvim --server "$NVIM_SOCKET" --remote-expr "setqflist([], 'r') | echo 'No TypeScript errors found'" >/dev/null
fi

rm "$TEMP_FILE"

echo "Output sent to Neovim quickfix list"
