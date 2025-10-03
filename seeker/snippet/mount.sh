#date: 2025-10-03T16:43:01Z
#url: https://api.github.com/gists/3ba2f025525a692570b652b510f885a4
#owner: https://api.github.com/users/AdamGagorik

#!/usr/bin/env bash
set -euo pipefail

SOURCE="$(dirname "${BASH_SOURCE[0]}")"

declare -a R_PATHS=(
  "$1:$2"
)

declare -a L_PATHS=(
  "${SOURCE}/mounts/$3"
)

do_mount() {
  local l_path
  local r_path
  l_path="${1}"
  r_path="${2}"
  if [[ ! -d "${l_path}" ]]; then
     mkdir -p "${l_path}"
  fi
  sshfs -o sshfs_debug "${r_path}" "${l_path}"
}

do_unmount() {
  local l_path
  l_path="${1}"
  fusermount -u "${l_path}"
  rmdir "${l_path}"
}

loop_do_mount() {
  local i
  for i in "${!R_PATHS[@]}"; do
    do_mount "${L_PATHS[i]}" "${R_PATHS[i]}"
  done
}

loop_do_unmount() {
  local i
  for i in "${!R_PATHS[@]}"; do
    do_unmount "${L_PATHS[i]}"
  done
}

trap loop_do_unmount EXIT
loop_do_mount

