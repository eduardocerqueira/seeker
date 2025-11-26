#date: 2025-11-26T16:51:11Z
#url: https://api.github.com/gists/048511871e998bb77c19f10550828b13
#owner: https://api.github.com/users/boyarskiy

#!/usr/bin/env bash
set -eo pipefail

# =========================================
# Quick config scan with TruffleHog
#  - Home config files/dirs
#  - Config-ish files in WORKING_DIR
#  - Ignores node_modules, .git, .next, dist
#  - Runs trufflehog per file/path (no arg list explosion)
#  - Masks secret fields as "XXX"
#
# WORKING_DIR default: $HOME
# USER ARG: path under $HOME, e.g.:
#   ./scan-configs.sh /work
#   ./scan-configs.sh /Development/smg
#   ./scan-configs.sh
# =========================================

# ---- Resolve working directory under $HOME ----
if [ -z "${1:-}" ]; then
  WORKING_DIR="$HOME"
else
  CLEANED="${1#/}"        # "/work" -> "work"
  WORKING_DIR="$HOME/$CLEANED"
fi

if [ ! -d "$WORKING_DIR" ]; then
  echo "❌ Working directory does not exist: $WORKING_DIR" >&2
  exit 1
fi

OUTPUT_FILE="$WORKING_DIR/masked-quick.json"

echo "🗂  Working directory: $WORKING_DIR"
echo "📄 Output will be:     $OUTPUT_FILE"
echo

# ---- Required tools ----
if ! command -v trufflehog >/dev/null 2>&1; then
  echo "❌ trufflehog not found in PATH" >&2
  exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "❌ jq not found in PATH" >&2
  exit 1
fi

# ---- Home config paths (using ~/ for readability) ----
HOME_PATHS=(
  "~/.ssh"
  "~/.aws"
  "~/.kube"
  "~/.docker"
  "~/.config"
  "~/.npmrc"
  "~/.gitconfig"
  "~/.git-credentials"
  "~/.bash_history"
  "~/.zsh_history"
  "~/.bashrc"
  "~/.zshrc"
  "~/.profile"
  "~/.bash_profile"
)

expand_home_path() {
  local p="$1"
  echo "${p/#\~/$HOME}"
}

# ---- Helper: run trufflehog on a single path and append masked JSON ----
run_trufflehog_on_path() {
  local target="$1"

  # Just in case
  if [ ! -e "$target" ]; then
    return
  fi

  echo "▶️  Scanning: $target"

  # We append (>>) so we build one big NDJSON file
  trufflehog filesystem \
    --json \
    --only-verified \
    "$target" \
  | jq -c '
      if has("Raw")   then .Raw   = "XXX" else . end |
      if has("RawV2") then .RawV2 = "XXX" else . end |
      if has("raw")   then .raw   = "XXX" else . end |
      if has("rawV2") then .rawV2 = "XXX" else . end
    ' >> "$OUTPUT_FILE" || true
}

# ---- Start fresh output file ----
: > "$OUTPUT_FILE"

########################################
# 1) Scan home config paths
########################################

echo "🔍 Scanning home config locations..."
for p in "${HOME_PATHS[@]}"; do
  FULL=$(expand_home_path "$p")
  if [ -e "$FULL" ]; then
    run_trufflehog_on_path "$FULL"
  fi
done

########################################
# 2) Find config files in WORKING_DIR
########################################

echo
echo "🔍 Finding config-like files in working directory (excluding node_modules, .git, .next, dist)..."

# We only care about *files* that look like configs/ envs
# and we PRUNE heavy dirs.
find "$WORKING_DIR" \
  -type d \( \
    -name "node_modules" -o \
    -name ".git" -o \
    -name ".next" -o \
    -name "dist" \
  \) -prune -o \
  -type f \( \
    -name '.env'      -o \
    -name '*.env'     -o \
    -name '.env.*'    -o \
    -path "*/.env/*"  -o \
    -name '*config*'  -o \
    -name '*.yaml'    -o \
    -name '*.yml'     -o \
    -name '*.toml'    -o \
    -name '*.ini'     -o \
    -name '*.properties' \
  \) -print | while IFS= read -r file; do
    run_trufflehog_on_path "$file"
  done

########################################
# 3) Summary
########################################

echo
if [ ! -s "$OUTPUT_FILE" ]; then
  echo "✅ No verified secrets found in home configs or working directory config files."
  echo "   (Output file is empty: $OUTPUT_FILE)"
else
  echo "✅ Scan complete. Masked results in: $OUTPUT_FILE"
  echo "   (NDJSON, one JSON object per line, secrets masked as \"XXX\")"
fi
