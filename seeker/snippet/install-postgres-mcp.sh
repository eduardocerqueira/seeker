#date: 2026-02-24T17:42:42Z
#url: https://api.github.com/gists/9dca16dfc23e8a870157a25b57e01efb
#owner: https://api.github.com/users/aseba

#!/bin/bash
# Install Postgres MCP for Cursor
# Usage: curl -fsSL https://gist.githubusercontent.com/aseba/9dca16dfc23e8a870157a25b57e01efb/raw/install-postgres-mcp.sh | bash
#
# Installs .cursor/mcp.json and .cursor/start-postgres-mcp.sh into the current directory.
# Installs missing dependencies: brew, gcloud, cloud-sql-proxy, uv/uvx

set -euo pipefail

echo "🔧 Installing Postgres MCP for Cursor..."

# --- Create .cursor directory ---
mkdir -p .cursor

# --- Install missing dependencies (runs in terminal, interactive OK) ---

# Homebrew
if ! command -v brew &>/dev/null; then
  echo "📦 Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  eval "$(/opt/homebrew/bin/brew shellenv)" 2>/dev/null || eval "$(/usr/local/bin/brew shellenv)" 2>/dev/null
fi

# gcloud
if ! command -v gcloud &>/dev/null; then
  echo "📦 Installing gcloud CLI..."
  brew install --cask google-cloud-sdk
fi

# cloud-sql-proxy
if ! command -v cloud-sql-proxy &>/dev/null; then
  echo "📦 Installing cloud-sql-proxy..."
  brew install cloud-sql-proxy
fi

# uv / uvx
if ! command -v uvx &>/dev/null; then
  export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uvx &>/dev/null; then
  echo "📦 Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# --- Write start-postgres-mcp.sh (check-only, no interactive installs) ---
cat > .cursor/start-postgres-mcp.sh << 'WRAPPER'
#!/bin/bash
# Wrapper script to ensure cloud-sql-proxy is running before starting postgres-mcp.
# Used by Cursor MCP integration (.cursor/mcp.json).
#
# Expects DATABASE_URI env var to be set by Cursor config.

set -euo pipefail

INSTANCE="remotely-platform:us-central1:platform-master-replica"
PROXY_PORT=5432

# --- Ensure PATH includes common install locations (Cursor launches with minimal env) ---
eval "$(/opt/homebrew/bin/brew shellenv)" 2>/dev/null || eval "$(/usr/local/bin/brew shellenv)" 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"

# Find and source gcloud PATH
GCLOUD_INC=$(find "$HOME" /usr/local /opt -maxdepth 5 -name "path.bash.inc" -path "*/google-cloud-sdk/*" 2>/dev/null | head -1)
if [ -n "$GCLOUD_INC" ]; then
  source "$GCLOUD_INC"
fi

# --- Check dependencies (no interactive installs — Cursor runs this non-interactively) ---

MISSING=()

if ! command -v brew &>/dev/null; then
  MISSING+=("brew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
fi

if ! command -v gcloud &>/dev/null; then
  MISSING+=("gcloud: brew install --cask google-cloud-sdk && gcloud auth login")
fi

if ! command -v cloud-sql-proxy &>/dev/null; then
  MISSING+=("cloud-sql-proxy: brew install cloud-sql-proxy")
fi

if ! command -v uvx &>/dev/null; then
  MISSING+=("uvx: curl -LsSf https://astral.sh/uv/install.sh | sh")
fi

if [ ${#MISSING[@]} -gt 0 ]; then
  echo "ERROR: Missing dependencies. Run these in your terminal:" >&2
  for m in "${MISSING[@]}"; do
    echo "  $m" >&2
  done
  echo "Then restart Cursor." >&2
  exit 1
fi

# Start cloud-sql-proxy with gcloud auth if not already running on our port
if ! lsof -i ":$PROXY_PORT" -sTCP:LISTEN &>/dev/null; then
  cloud-sql-proxy "$INSTANCE" --port "$PROXY_PORT" -g &
  PROXY_PID=$!

  # Wait up to 10 seconds for proxy to be ready
  for _ in $(seq 1 20); do
    if lsof -i ":$PROXY_PORT" -sTCP:LISTEN &>/dev/null; then
      break
    fi
    sleep 0.5
  done

  if ! lsof -i ":$PROXY_PORT" -sTCP:LISTEN &>/dev/null; then
    echo "ERROR: cloud-sql-proxy failed to start on port $PROXY_PORT" >&2
    echo "Try: gcloud auth login" >&2
    kill "$PROXY_PID" 2>/dev/null
    exit 1
  fi
fi

# Start postgres-mcp in restricted (read-only) mode via stdio
exec uvx postgres-mcp --access-mode=restricted
WRAPPER

chmod +x .cursor/start-postgres-mcp.sh

# --- Write mcp.json ---
cat > .cursor/mcp.json << 'MCPJSON'
{
  "mcpServers": {
    "postgres": {
      "command": "bash",
      "args": [
        "${workspaceFolder}/.cursor/start-postgres-mcp.sh"
      ],
      "env": {
        "DATABASE_URI": "**********"://ci:${env:PLATFORM_DB_PASSWORD}@127.0.0.1:5432/platform"
      }
    }
  }
}
MCPJSON

echo ""
echo "✅ Created .cursor/mcp.json"
echo "✅ Created .cursor/start-postgres-mcp.sh"
echo ""

# --- Check manual steps ---
NEEDS_ACTION=false

if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | grep -q .; then
  echo "⚠️  Run: gcloud auth login"
  NEEDS_ACTION=true
fi

if [ -z "${PLATFORM_DB_PASSWORD: "**********"
  echo  "**********"⚠️  Set your DB password: "**********"
  echo "   echo 'export PLATFORM_DB_PASSWORD= "**********"
  echo "   (Get the password from platform/database-env or a colleague)"
  NEEDS_ACTION=true
fi

if [ "$NEEDS_ACTION" = true ]; then
  echo ""
  echo "After completing the steps above, restart Cursor (Cmd+Q) to activate the MCP."
else
  echo ""
  echo "🚀 All set! Restart Cursor (Cmd+Q) to activate the Postgres MCP."
fi
P."
fi
