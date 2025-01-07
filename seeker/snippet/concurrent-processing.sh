#date: 2025-01-07T16:42:05Z
#url: https://api.github.com/gists/9229605d12f075e0751de3de4746d765
#owner: https://api.github.com/users/ThinaticSystem

#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

on_some_job_error() {
  kill "$(jobs -pr)"
  exit 1
}

trap 'on_some_job_error' ERR

command1 &
command2 &

wait

echo 'All commands success'
