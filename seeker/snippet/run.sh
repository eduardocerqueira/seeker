#date: 2026-01-05T17:11:12Z
#url: https://api.github.com/gists/ca83ba5417e3d9e25b68c7bdc644832c
#owner: https://api.github.com/users/soderlind

#!/usr/bin/env bash
set -euo pipefail

# Bootstrap runner for the Ralph loop.
# Tests are defined in prd.json and executed by ./copilot-ralph.py.

# If progress.txt does not exist, create it
if [ ! -f progress.txt ]; then
	echo "Creating progress.txt..."
	touch progress.txt
fi

# Initialize git repo if not already initialized
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    echo "Initializing git repository..."
    git init
    git add -A
    git commit -m "Initial commit"
fi

# Install dependencies if possible.
# If this is a brand-new empty repo, there may be no package.json yet.
# In that case, skip install so the agent can create the project files.
if [ -f package.json ]; then
    if [ ! -d node_modules ]; then
        echo "Installing dependencies..."
        if command -v pnpm &>/dev/null; then
            pnpm install
        else
            npm install
        fi
    fi
else
    echo "No package.json found; skipping dependency install (agent will create project files)."
fi

# Ensure clean working tree (Ralph refuses to start otherwise)
if [ -n "$(git status --porcelain)" ]; then
    echo "Working tree not clean; committing changes before Ralph run..."
    git add -A
    if [ "${RALPH_NO_VERIFY:-}" = "1" ]; then
        git commit -m "chore: prep for ralph run" --no-verify
    else
        git commit -m "chore: prep for ralph run"
    fi
fi
./copilot-ralph.py --prd prd.json --progress progress.txt --max-iterations 30 \
    --copilot-arg="--allow-all-tools" \
    --copilot-arg="--allow-all-paths"
