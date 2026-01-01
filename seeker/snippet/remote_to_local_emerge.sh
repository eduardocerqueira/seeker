#date: 2026-01-01T17:06:53Z
#url: https://api.github.com/gists/ae7e2bb986637b52fe09929674c2e544
#owner: https://api.github.com/users/csepulveda

#!/usr/bin/env bash
set -euo pipefail

REMOTE_USER_HOST="root@192.168.100.251"
CHROOT_PATH="/mnt/gentoo"

usage() {
  cat <<'EOF'
Usage:
  remote_to_local_emerge.sh -- <command> [args...]

Example:
  remote_to_local_emerge.sh -- emerge -av sys-kernel/gentoo-sources

What it does:
  1) SSH to root@192.168.100.251
  2) Mounts necessary filesystems and enters chroot
  3) Executes the provided command INSIDE the chroot
  4) Streams output live
  5) Executes the same command locally, prefixed with sudo
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" != "--" ]]; then
  echo "Error: you must use '--' before the command."
  usage
  exit 2
fi
shift

if [[ $# -lt 1 ]]; then
  echo "Error: you must provide a command to execute."
  usage
  exit 2
fi

# Quote the command safely for local sudo
printf -v CMD_Q '%q ' "$@"
CMD_Q="${CMD_Q% }"

# Encode command in base64 to avoid all quoting issues through SSH
CMD_B64=$(printf '%s' "$*" | base64)

echo "==> [REMOTE] Running inside chroot on ${REMOTE_USER_HOST}: $*"
echo

ssh "${REMOTE_USER_HOST}" bash -s -- "${CMD_B64}" <<'REMOTE_SCRIPT'
set -euo pipefail

CHROOT_PATH="/mnt/gentoo"
CMD_B64="$1"

# Decode command
CMD=$(echo "${CMD_B64}" | base64 -d)

# Mount filesystems (ignore errors if already mounted)
mount --rbind /dev "${CHROOT_PATH}/dev" 2>/dev/null || true
mount --make-rslave "${CHROOT_PATH}/dev" 2>/dev/null || true
mount -t proc /proc "${CHROOT_PATH}/proc" 2>/dev/null || true
mount --rbind /sys "${CHROOT_PATH}/sys" 2>/dev/null || true
mount --make-rslave "${CHROOT_PATH}/sys" 2>/dev/null || true
mount --rbind /tmp "${CHROOT_PATH}/tmp" 2>/dev/null || true
mount --bind /run "${CHROOT_PATH}/run" 2>/dev/null || true

cp /etc/resolv.conf "${CHROOT_PATH}/etc" 2>/dev/null || true

# Execute command inside chroot
chroot "${CHROOT_PATH}" /bin/bash -lc "${CMD}"
REMOTE_SCRIPT

remote_status=$?

if [[ $remote_status -ne 0 ]]; then
  echo "==> [REMOTE] Command failed with status ${remote_status}"
  exit $remote_status
fi

echo
echo "==> [LOCAL] Running locally with sudo: $*"
echo

sudo bash -lc "${CMD_Q}"