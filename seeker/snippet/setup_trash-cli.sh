#date: 2025-08-12T17:01:22Z
#url: https://api.github.com/gists/497f27d318fba1d03909dd1592157843
#owner: https://api.github.com/users/6MA-606

#!/bin/bash

declare -A aliases=(
  [t]="trash-put"
  [tl]="trash-list"
  [trs]="trash-restore"
)

# Detect shell and rc file
shell_name=$(basename "$SHELL")

case "$shell_name" in
  bash)
    rc_file="$HOME/.bashrc"
    ;;
  zsh)
    rc_file="$HOME/.zshrc"
    ;;
  ksh)
    rc_file="$HOME/.kshrc"
    ;;
  *)
    echo "Unsupported shell: $shell_name"
    exit 1
    ;;
esac

# Install trash-cli if not installed
if ! command -v trash-put >/dev/null 2>&1; then
  echo "trash-cli not found. Installing..."
  sudo apt update && sudo apt install -y trash-cli || {
    echo "Failed to install trash-cli. Exiting."
    exit 1
  }
else
  echo "trash-cli is already installed."
fi

# Add aliases if missing
added_any=false
for name in "${!aliases[@]}"; do
  if grep -q -E "alias\s+$name=" "$rc_file"; then
    echo "Alias '$name' already exists in $rc_file"
  else
    echo "alias $name='${aliases[$name]}'" >> "$rc_file"
    echo "Alias '$name' added to $rc_file"
    added_any=true
  fi
done

# Source rc file if added new alias, to apply immediately
if [ "$added_any" = true ]; then
  # Use the current shell to source, fallback to bash
  if [ -n "$BASH_VERSION" ]; then
    source "$rc_file"
  elif [ -n "$ZSH_VERSION" ]; then
    source "$rc_file"
  else
    echo "Please source your rc file manually: source $rc_file"
  fi
fi
