#date: 2025-09-04T16:54:33Z
#url: https://api.github.com/gists/b2df1d13b1e2f8c0b0ff182ec4135c26
#owner: https://api.github.com/users/rigerc

#!/usr/bin/env bash
# Brief: Run uvx init (unless .roo exists), rename .claude -> .roo, replace ".claude" -> ".roo" in files,
# and replace "CLAUDE.MD" (case-insensitive) -> "AGENTS.MD". Prints a colored summary of changes.
# Usage: bash scripts/spec-kit_for_roo.sh
# Notes:
#  - Skips running uvx if a .roo directory already exists.
#  - Performs in-place edits; temporary .bak files removed after edit.
#  - Excludes the .git directory from searches.

set -euo pipefail

# Colors
NC='\033[0m'        # No Color / reset
BOLD='\033[1m'
RED='\033[31m'
GREEN='\033[32m'
YELLOW='\033[33m'
BLUE='\033[34m'

# Helpers
info()   { printf "${BLUE}%s${NC}\n" "$1"; }
ok()     { printf "${GREEN}%s${NC}\n" "$1"; }
warn()   { printf "${YELLOW}%s${NC}\n" "$1"; }
err()    { printf "${RED}%s${NC}\n" "$1" >&2; }
title()  { printf "${BOLD}${BLUE}%s${NC}\n" "$1"; }
hr()     { printf "%s\n" "----------------------------------------"; }

# Ensure cleanup of any stray .bak files if the script exits unexpectedly
cleanup_baks() {
  # only remove .bak files created by this script pattern
  # Be conservative: only remove *.bak in files we touched. We'll rely on rm -f after edits too.
  true
}
trap cleanup_baks EXIT

title "Spec-kit: Convert .claude -> .roo"
hr

# Fail-safe: abort if a .claude directory already exists to avoid accidental overwrite
if [ -d ".claude" ]; then
  err "ERROR: .claude directory already exists. Aborting to avoid overwriting or unexpected changes."
  err "If you intended to convert Spec-Kit's .claude to .roo, please remove or rename your own .claude and re-run this script."
  exit 1
fi

# Decide whether to run uvx:
# - Skip if a .roo directory already exists (preserve existing conversion)
# - Skip if repository already contains 'memory', 'scripts', and 'templates' folders
if [ -d ".roo" ] || { [ -d "memory" ] && [ -d "scripts" ] && [ -d "templates" ]; }; then
  info "Found .roo or existing Spec-kit project structure (memory, scripts, templates); skipping uvx command."
else
  UVX_CMD='uvx --from git+https://github.com/github/spec-kit.git specify init --here --ai claude --ignore-agent-tools'
  info "Running: ${BOLD}$UVX_CMD${NC}"
  if $UVX_CMD; then
    ok "uvx command completed successfully."
  else
    err "ERROR: uvx command failed with exit code $?."
    exit 1
  fi
fi

hr
# Rename .claude directory to .roo if present
if [ -d ".claude" ]; then
  if [ -e ".roo" ]; then
    warn "Target .roo already exists. Backing up existing .roo to .roo.bak"
    mv -v ".roo" ".roo.bak"
  fi
  mv -v ".claude" ".roo"
  ok "RENAMED: .claude -> .roo"
else
  info "No .claude directory present to rename."
fi

hr
# Replace occurrences of '.claude' with '.roo'
info "Searching for occurrences of '.claude'..."
CLAUDE_FILES_FOUND=0
CLAUDE_TOTAL_REPLACED=0

# Find files containing literal ".claude"
while IFS= read -r -d '' file; do
  count=$(grep -o '\.claude' "$file" | wc -l || true)
  if [ "$count" -gt 0 ]; then
    sed -i.bak 's/\.claude/\.roo/g' "$file"
    rm -f "${file}.bak"
    printf "${YELLOW}REPLACED${NC}: %s occurrences in '%s' ('.claude' -> '.roo')\n" "$count" "$file"
    CLAUDE_FILES_FOUND=$((CLAUDE_FILES_FOUND+1))
    CLAUDE_TOTAL_REPLACED=$((CLAUDE_TOTAL_REPLACED+count))
  fi
done < <(grep -RIlZ --exclude-dir=.git --binary-files=without-match '\.claude' . || true)

if [ "$CLAUDE_FILES_FOUND" -eq 0 ]; then
  info "No files contained '.claude'."
else
  ok "Replaced ${CLAUDE_TOTAL_REPLACED} total occurrences across ${CLAUDE_FILES_FOUND} files."
fi

hr
# Replace 'CLAUDE.MD' with 'AGENTS.MD' (case-insensitive)
info "Searching for occurrences of 'CLAUDE.MD' (case-insensitive)..."
CLAUDE_MD_FILES_FOUND=0
CLAUDE_MD_TOTAL_REPLACED=0

while IFS= read -r -d '' file; do
  count=$(grep -o -i 'CLAUDE\.MD' "$file" | wc -l || true)
  if [ "$count" -gt 0 ]; then
    # Use perl for a safe case-insensitive in-place replacement and create a .bak which we remove
    perl -pi.bak -e 's/CLAUDE\.MD/AGENTS.MD/ig' "$file"
    rm -f "${file}.bak"
    printf "${YELLOW}REPLACED${NC}: %s occurrences in '%s' ('CLAUDE.MD' -> 'AGENTS.MD')\n" "$count" "$file"
    CLAUDE_MD_FILES_FOUND=$((CLAUDE_MD_FILES_FOUND+1))
    CLAUDE_MD_TOTAL_REPLACED=$((CLAUDE_MD_TOTAL_REPLACED+count))
  fi
done < <(grep -RIlZ -i --exclude-dir=.git --binary-files=without-match 'CLAUDE.MD' . || true)

if [ "$CLAUDE_MD_FILES_FOUND" -eq 0 ]; then
  info "No files contained 'CLAUDE.MD'."
else
  ok "Replaced ${CLAUDE_MD_TOTAL_REPLACED} total occurrences across ${CLAUDE_MD_FILES_FOUND} files."
fi

hr
title "Summary"
if [ "$CLAUDE_FILES_FOUND" -eq 0 ] && [ "$CLAUDE_MD_FILES_FOUND" -eq 0 ]; then
  ok "No replacements were necessary."
else
  printf "${BOLD}Files updated:${NC} %d ('.claude' -> '.roo'), %d ('CLAUDE.MD' -> 'AGENTS.MD')\n" "${CLAUDE_FILES_FOUND}" "${CLAUDE_MD_FILES_FOUND}"
  printf "${BOLD}Total replacements:${NC} %d ('.claude' -> '.roo'), %d ('CLAUDE.MD' -> 'AGENTS.MD')\n" "${CLAUDE_TOTAL_REPLACED}" "${CLAUDE_MD_TOTAL_REPLACED}"
fi

ok "Done. You can now run /specify in Roo Code to start the Spec-kit process."