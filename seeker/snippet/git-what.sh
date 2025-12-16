#date: 2025-12-16T17:10:16Z
#url: https://api.github.com/gists/2629551a9d02f1ea9427e23a0cceb83d
#owner: https://api.github.com/users/eskil

#!/usr/bin/env bash
set -euo pipefail

# ------------------------------
# CONFIGURATION
# ------------------------------
PROTECTED_PATTERNS=(
  "*.env"
  "*.pem"
  "*.key"
  "*.crt"
  "*.p12"
  "*secrets*"
)

EDITOR="${EDITOR:-vim}"

# ------------------------------
# HELPERS
# ------------------------------
is_protected() {
  local f="$1"
  for pat in "${PROTECTED_PATTERNS[@]}"; do
    [[ "$f" == $pat ]] && return 0
  done
  return 1
}

# Portable move-to-trash:
move_to_trash() {
  local f="$1"

  # Try trash-put (trash-cli)
  if command -v trash-put >/dev/null 2>&1; then
    trash-put "$f" && return 0
  fi

  # Try gio (GNOME)
  if command -v gio >/dev/null 2>&1; then
    gio trash "$f" && return 0
  fi

  # Try gvfs-trash (older systems)
  if command -v gvfs-trash >/dev/null 2>&1; then
    gvfs-trash "$f" && return 0
  fi

  # macOS: use AppleScript to move to Trash
  # macOS Finder
  if [[ "$(uname -s)" == "Darwin" ]]; then
    for f in "$@"; do
      # Resolve absolute path
      if command -v realpath >/dev/null 2>&1; then
        absf=$(realpath "$f")
      else
        absf="$PWD/$f"
      fi
      /usr/bin/osascript -e "tell application \"Finder\" to move (POSIX file \"$absf\") to trash"
    done
    return
  fi  

  # # Fallback: move into $XDG_DATA_HOME/Trash/files or ~/.local/share/Trash/files
  # echo "\t try fallback"
  # local trash_dir="${XDG_DATA_HOME:-$HOME/.local/share}/Trash/files"
  # mkdir -p "$trash_dir"
  # # If file name collision, append timestamp
  # local base
  # base=$(basename -- "$f")
  # if [[ -e "$trash_dir/$base" ]]; then
  #   mv -- "$f" "$trash_dir/${base}.$(date +%s)" && return 0
  # else
  #   mv -- "$f" "$trash_dir/" && return 0
  # fi

  return 1
}

# Permanent delete
permanent_delete() {
  local f="$1"
  rm -f -- "$f"
}

# ------------------------------
# BUILD A TEMP PREVIEW SCRIPT (robust)
# ------------------------------
PREVIEW_SCRIPT=$(mktemp)
chmod +x "$PREVIEW_SCRIPT"
cat > "$PREVIEW_SCRIPT" <<'PREVIEW_EOF'
#!/usr/bin/env bash
# arg1 -> path to file to preview
f="$1"

# If not a regular file, show message
if [[ ! -f "$f" ]]; then
  echo "Not a regular file"
  exit 0
fi

# Protected check: we can't access PROTECTED_PATTERNS array from parent here, so
# replicate a small subset of patterns or read them from env var if exported.
# We'll accept PROTECTED list in PROTECT_PATTERNS env if present.
if [[ -n "${PROTECT_PATTERNS:-}" ]]; then
  IFS=$'\n' read -r -d '' -a pats <<< "$PROTECT_PATTERNS" || true
  for pat in "${pats[@]}"; do
    if [[ "$f" == $pat ]]; then
      echo "[ PROTECTED FILE ]"
      echo "Matches protected pattern: $pat"
      echo
      break
    fi
  done
fi

# Show text preview if file command says text
if file "$f" | grep -qi text; then
  if command -v bat >/dev/null 2>&1; then
    bat --style=plain --color=always --line-range=1:200 "$f"
  else
    head -n 200 "$f"
  fi
else
  echo "[binary file]"
fi
PREVIEW_EOF

# Ensure temp preview script is removed on exit
cleanup() {
  rm -f "$PREVIEW_SCRIPT"
}
trap cleanup EXIT

# Export protected patterns into a single-line env var for preview script
# Use null-separated to be safe
PROTECT_PATTERNS=$(printf "%s\n" "${PROTECTED_PATTERNS[@]}")
export PROTECT_PATTERNS

# ------------------------------
# MAIN
# ------------------------------
files=$(git ls-files --others --exclude-standard)

if [[ -z "$files" ]]; then
  echo "No untracked files."
  exit 0
fi

# keep original fzf options the user requested
selected=$(printf "%s\n" "$files" | fzf \
  --multi \
  --prompt="Untracked Files › " \
  --border \
  --ansi \
  --reverse \
  --marker="x " \
  --pointer="- " \
  --preview="$PREVIEW_SCRIPT {}" \
  --color='pointer:161,marker:168' \
  --preview-window=right:50% \
  --bind="ctrl-o:execute($EDITOR {+})" \
  --bind="ctrl-e:execute($EDITOR {+})" 
  # --bind="ctrl-d:execute(
  #     echo 'Force deleting:' {+};
  #     for f in {+}; do rm -f \"\$f\"; done;
  #     echo 'Done';
  #     sleep 1
  #   )+abort"
)

# If user aborted / nothing selected
if [[ -z "${selected:-}" ]]; then
  exit 0
fi

# ------------------------------
# NORMAL DELETE (Enter key): move to trash unless protected
# ------------------------------
# iterate selected lines (could be multiple)
printf "%s\n" "$selected" | while IFS= read -r f; do
  if is_protected "$f"; then
    echo "Skipping PROTECTED file: $f"
    continue
  fi

  echo "Moving to trash: $f"
  if move_to_trash "$f"; then
    echo " → Moved to trash"
  else
    echo " → Failed to move to trash, attempting permanent delete"
    permanent_delete "$f"
  fi
done
