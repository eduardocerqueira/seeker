#date: 2026-03-03T17:24:38Z
#url: https://api.github.com/gists/82faf1d263cc86c267f94016d33528ee
#owner: https://api.github.com/users/edelah

#!/bin/bash
set -euo pipefail

CLAWSSH_VERSION="2"
CLAWSSH_INSTALL_BASE_URL="${CLAWSSH_INSTALL_BASE_URL:-https://gist.githubusercontent.com/edelah/82faf1d263cc86c267f94016d33528ee/raw}"
CLAWSSH_INSTALL_ARTIFACT_URL="${CLAWSSH_INSTALL_ARTIFACT_URL:-https://gist.githubusercontent.com/edelah/82faf1d263cc86c267f94016d33528ee/raw/clawssh-backend-2.py}"
CLAWSSH_STATE_DIR="${CLAWSSH_STATE_DIR:-$HOME/.clawssh}"
CLAWSSH_ARTIFACT="clawssh-backend-${CLAWSSH_VERSION}.py"

detect_package_manager() {
  local manager
  for manager in apt-get dnf yum pacman zypper apk brew; do
    if command -v "$manager" >/dev/null 2>&1; then
      printf '%s\n' "$manager"
      return 0
    fi
  done
  return 1
}

install_missing_system_packages() {
  local manager
  local missing=()
  command -v python3 >/dev/null 2>&1 || missing+=("python3")
  command -v tmux >/dev/null 2>&1 || missing+=("tmux")
  command -v openssl >/dev/null 2>&1 || missing+=("openssl")

  if [ "${#missing[@]}" -eq 0 ]; then
    return 0
  fi

  echo "Missing system packages: ${missing[*]}"

  manager="$(detect_package_manager || true)"
  if [ -z "$manager" ]; then
    echo "ERROR: Could not detect a supported package manager." >&2
    echo "Install these packages manually, then rerun:" >&2
    printf '  %s\n' "${missing[@]}" >&2
    exit 1
  fi

  case "$manager" in
    apt-get)
      echo "About to run: sudo apt-get update"
      echo "About to run: sudo apt-get install -y ${missing[*]}"
      sudo apt-get update
      sudo apt-get install -y "${missing[@]}"
      ;;
    dnf)
      echo "About to run: sudo dnf install -y ${missing[*]}"
      sudo dnf install -y "${missing[@]}"
      ;;
    yum)
      echo "About to run: sudo yum install -y ${missing[*]}"
      sudo yum install -y "${missing[@]}"
      ;;
    pacman)
      echo "About to run: sudo pacman -Sy --noconfirm ${missing[*]}"
      sudo pacman -Sy --noconfirm "${missing[@]}"
      ;;
    zypper)
      echo "About to run: sudo zypper --non-interactive install ${missing[*]}"
      sudo zypper --non-interactive install "${missing[@]}"
      ;;
    apk)
      echo "About to run: sudo apk add ${missing[*]}"
      sudo apk add "${missing[@]}"
      ;;
    brew)
      echo "About to run: brew install ${missing[*]}"
      brew install "${missing[@]}"
      ;;
  esac
}

download_artifact() {
  local url="$CLAWSSH_INSTALL_ARTIFACT_URL"
  local dest="$CLAWSSH_STATE_DIR/clawssh.py"

  echo "Downloading $url"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url" -o "$dest"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$dest" "$url"
  else
    python3 - "$url" "$dest" <<'PY'
import pathlib
import sys
import urllib.request

url = sys.argv[1]
output = pathlib.Path(sys.argv[2])
with urllib.request.urlopen(url) as response:
    output.write_bytes(response.read())
PY
  fi
  chmod 700 "$dest"
}

main() {
  install_missing_system_packages

  mkdir -p "$CLAWSSH_STATE_DIR"
  chmod 700 "$CLAWSSH_STATE_DIR"

  download_artifact

  echo "Installed ClawSSH monolithic to: $CLAWSSH_STATE_DIR/clawssh.py"
  echo

  if [ -r /dev/tty ] && [ -t 1 ]; then
    exec </dev/tty >/dev/tty 2>/dev/tty "$CLAWSSH_STATE_DIR/clawssh.py"
  fi

  echo "Start ClawSSH setup with:"
  echo "  $CLAWSSH_STATE_DIR/clawssh.py"
}

main "$@"
