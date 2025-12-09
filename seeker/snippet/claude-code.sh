#date: 2025-12-09T17:10:47Z
#url: https://api.github.com/gists/51041d9555d40f753320ce31b96acf9c
#owner: https://api.github.com/users/clankerbot

#!/bin/bash
set -e

CLAUDE_DIR="$HOME/.claude/.devcontainer"
DUMPS_DIR="$HOME/claude/dumps"
CONFIG="$CLAUDE_DIR/devcontainer.json"

# Create dumps directory
mkdir -p "$DUMPS_DIR"

# Download devcontainer files if missing
if [ ! -f "$CONFIG" ]; then
    mkdir -p "$CLAUDE_DIR"
    curl -fsSL https://raw.githubusercontent.com/anthropics/claude-code/main/.devcontainer/devcontainer.json -o "$CONFIG"
    curl -fsSL https://raw.githubusercontent.com/anthropics/claude-code/main/.devcontainer/Dockerfile -o "$CLAUDE_DIR/Dockerfile"
    curl -fsSL https://raw.githubusercontent.com/anthropics/claude-code/main/.devcontainer/init-firewall.sh -o "$CLAUDE_DIR/init-firewall.sh"
fi

# Add dumps mount if not present
if ! grep -q 'claude/dumps' "$CONFIG"; then
    tmp=$(mktemp)
    sed 's/"mounts": \[/"mounts": [\n\t\t"source=${localEnv:HOME}\/claude\/dumps,target=\/home\/node\/claude\/dumps,type=bind",/' "$CONFIG" > "$tmp"
    mv "$tmp" "$CONFIG"
fi

# Run devcontainer with stdin attached
if ! pnpx @devcontainers/cli exec --workspace-folder . --config "$CONFIG" claude --dangerously-skip-permissions 2>/dev/null </dev/tty; then
    echo "Starting devcontainer..."
    pnpx @devcontainers/cli up --workspace-folder . --config "$CONFIG" \
    && pnpx @devcontainers/cli exec --workspace-folder . --config "$CONFIG" claude --dangerously-skip-permissions </dev/tty
fi
