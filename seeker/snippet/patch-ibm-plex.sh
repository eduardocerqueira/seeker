#date: 2023-08-24T16:45:25Z
#url: https://api.github.com/gists/01cfe2f248dd858eb7e12ad77ae646a4
#owner: https://api.github.com/users/ryanccn

#! /usr/bin/env nix-shell
#! nix-shell --pure --keep DEBUG -i bash -p bash coreutils fd python311Packages.fonttools
#  shellcheck shell=bash

set -eo pipefail

ansi_dim="\033[2m"
ansi_blue="\033[34m"
ansi_green="\033[32m"
ansi_reset="\033[0m"

run() {
  echo -e "${ansi_dim}$ $*${ansi_reset}"

  if [ -z "$DEBUG" ]; then
    "$@" &> /dev/null
  else
    "$@"
  fi
}

medium_fonts=$(fd --extension=otf Medium)

for medium_font in $medium_fonts; do
  echo -e "${ansi_blue}Patching${ansi_reset} $medium_font (Medm -> Medium)"

  ttx_path="${medium_font%.*}.ttx"
  
  run ttx "$medium_font"
  run sed -i 's/Medm/Medium/g' "$ttx_path"
  run ttx -f "$ttx_path"
done

semibold_fonts=$(fd --extension=otf SemiBold)

for semibold_font in $semibold_fonts; do
  echo -e "${ansi_blue}Patching${ansi_reset} $semibold_font (SmBld -> Semibold)"

  ttx_path="${semibold_font%.*}.ttx"
  
  run ttx "$semibold_font"
  run sed -i 's/SmBld/Semibold/g' "$ttx_path"
  run ttx -f "$ttx_path"
done

echo -e "${ansi_green}Done!${ansi_reset}"
