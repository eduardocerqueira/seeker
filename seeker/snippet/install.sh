#date: 2026-03-02T17:32:20Z
#url: https://api.github.com/gists/81d0b3faad8328d3c95fa02b5b22de33
#owner: https://api.github.com/users/edelah

#!/bin/bash
set -euo pipefail

CLAWSSH_VERSION="2"
CLAWSSH_INSTALL_BASE_URL="${CLAWSSH_INSTALL_BASE_URL:-https://gist.githubusercontent.com/edelah/81d0b3faad8328d3c95fa02b5b22de33/raw}"
CLAWSSH_INSTALL_ARTIFACT_URL="${CLAWSSH_INSTALL_ARTIFACT_URL:-https://gist.githubusercontent.com/edelah/81d0b3faad8328d3c95fa02b5b22de33/raw/clawssh-backend-2.py}"
CLAWSSH_STATE_DIR="${CLAWSSH_STATE_DIR:-$HOME/.clawssh}"
CLAWSSH_ARTIFACT="clawssh-backend-${CLAWSSH_VERSION}.py"

detect_package_manager() {
  for manager in apt-get dnf yum pacman zypper apk brew; do
    if command -v "$manager" >/dev/null 2>&1; then
      printf '%s\n' "$manager"
      return 0
    fi
  done
  return 1
}

need_python_pip() {
  if ! command -v python3 >/dev/null 2>&1; then
    return 0
  fi
  python3 -m pip --version >/dev/null 2>&1 || return 0
  return 1
}

install_missing_system_packages() {
  local manager missing=()
  command -v python3 >/dev/null 2>&1 || missing+=("python3")
  need_python_pip && missing+=("python3-pip")
  command -v tmux >/dev/null 2>&1 || missing+=("tmux")
  command -v openssl >/dev/null 2>&1 || missing+=("openssl")
  command -v qrencode >/dev/null 2>&1 || missing+=("qrencode")

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
  local output="$1"

  echo "Downloading $url"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url" -o "$output"
    return 0
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -qO "$output" "$url"
    return 0
  fi

  python3 - "$url" "$output" <<'PY'
import pathlib
import sys
import urllib.request

url = sys.argv[1]
output = pathlib.Path(sys.argv[2])
with urllib.request.urlopen(url) as response:
    output.write_bytes(response.read())
PY
}

main() {
  local work_dir artifact_path answer
  install_missing_system_packages

  work_dir="$(mktemp -d)"
  cleanup() {
    rm -rf "${work_dir:-}"
  }
  trap cleanup EXIT
  artifact_path="$work_dir/clawssh.py"
  download_artifact "$artifact_path"

  mkdir -p "$CLAWSSH_STATE_DIR"
  chmod 700 "$CLAWSSH_STATE_DIR"
  cp "$artifact_path" "$CLAWSSH_STATE_DIR/clawssh.py"
  chmod 700 "$CLAWSSH_STATE_DIR/clawssh.py"

  echo "Installed ClawSSH backend runtime:"
  echo "  $CLAWSSH_STATE_DIR/clawssh.py"
  echo

  if [ -r /dev/tty ] && [ -t 1 ]; then
    exec </dev/tty >/dev/tty 2>/dev/tty "$CLAWSSH_STATE_DIR/clawssh.py"
  fi

  echo "Start ClawSSH setup with:"
  echo "  $CLAWSSH_STATE_DIR/clawssh.py"
}

main "$@"
