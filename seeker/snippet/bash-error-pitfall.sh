#date: 2022-10-12T17:21:41Z
#url: https://api.github.com/gists/810d5471c6617c4391d7f4fe07670f74
#owner: https://api.github.com/users/cardil

#!/usr/bin/env bash

set -Eeuo pipefail

function error_handler() {
  local code="${1:-${?}}"
  abort "🚨 Error (code: ${code}) occurred at ${BASH_SOURCE[1]}:${BASH_LINENO[0]}, with command: ${BASH_COMMAND}"
}

function abort() {
  echo "Error: $*" >&2
  exit 42
}

function sign() {
  echo 'Signing'

  false

  echo 'Signed'
}

function main() {
  trap error_handler ERR
  sign || abort "Failed to sign"
  # sign
}

main "$@"
