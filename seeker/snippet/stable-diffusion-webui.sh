#date: 2024-04-15T16:42:58Z
#url: https://api.github.com/gists/3efa7c9383385c27a07fa6e599c11b88
#owner: https://api.github.com/users/Amb0s

#!/usr/bin/env bash

set -Eeuo pipefail
trap cleanup SIGINT SIGTERM ERR EXIT

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [-h] [-v] -f file

Start Stable Diffusion WebUI.

Available options:

-h, --help             Print this help and exit
-v, --verbose          Print script debug information
-f, --file             Dockerfile path
EOF
  exit
}

cleanup() {
  trap - SIGINT SIGTERM ERR EXIT
}

msg() {
  echo >&2 -e "${1-}"
}

die() {
  local msg=$1
  local code=${2-1} # Default exit status 1.
  msg "$msg"
  exit "$code"
}

parse_params() {
  file="$HOME/Download/stable-diffusion-webui-docker/docker-compose.yml"

  while :; do
    case "${1-}" in
    -h | --help) usage ;;
    -v | --verbose) set -x ;;
    -f | --file)
      file="${2-}"
      shift
      ;;
    -?*) die "Unknown option: $1" ;;
    *) break ;;
    esac
    shift
  done

  args=("$@")

  return 0
}

check_exploit() {
  # Source: https://pytorch.org/blog/compromised-nightly-dependency/
  
  docker exec -it $container_id python3 -c "import pathlib;import importlib.util;s=importlib.util.find_spec('triton'); affected=any(x.name == 'triton' for x in (pathlib.Path(s.submodule_search_locations[0] if s is not None else '/' ) / 'runtime').glob('*'));print('You are {}affected'.format('' if affected else 'not '))"
}

parse_params "$@"

if (! systemctl -q is-active docker); then
	sudo systemctl start docker	
fi

docker compose -f $file --profile auto up