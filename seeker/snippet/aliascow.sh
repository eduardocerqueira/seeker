#date: 2026-02-20T17:18:41Z
#url: https://api.github.com/gists/c75b37dca88848c344190b02e351d67d
#owner: https://api.github.com/users/andrewhaller

#!/usr/bin/env bash

# Include this file in ~/.bash_aliases
#
# Usage: aliascow [COWSAY_COMMAND] [MAX_LENGTH] [COWFILES]
# COWSAY_COMMAND: The command to use for cow output (default: cowthink)
# MAX_LENGTH: Maximum line length before adding a newline (default: 72)
# COWFILES: Space-separated list of cowfiles to choose from (default: "default three-eyes moose tux bunny bud-frogs")
#
# Examples:
#   . /path/to/aliascow.sh aliascow
#   . /path/to/aliascow.sh aliascow cowthink 80
#   . /path/to/aliascow.sh aliascow cowsay 60 "default three-eyes moose"

aliascow() {
  local com="${1:-cowthink}"
  local max_length="${2:-72}"
  local cowfiles="${3:-default three-eyes moose tux bunny bud-frogs}"
  if [ -n "${com-}" ] && type "$com" >/dev/null 2>&1; then
    shift
    while read -r line; do
      [ "$(echo "$line" | wc -L)" -lt "${max_length}" ] || {
        suffix="\n"
        break
      }
    done <<<"$(alias)"
    [ -z "$suffix" ] && opts+=("-n") || opts+=("-W${max_length}")
    set - "${opts[@]}"
    alias | while read -r line; do
      echo -e "${line}${suffix}"
    done | "$com" -f "$(echo "${cowfiles:-default}" | xargs -n 1 | shuf -n 1)" "$@"
  fi
}

printf "\e[2m" # Dim the output; comment out or modify as desired
aliascow "$@"
printf "\e[0m" # Reset text formatting; comment out as necessary