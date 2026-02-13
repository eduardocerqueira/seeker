#date: 2026-02-13T17:34:32Z
#url: https://api.github.com/gists/0d827c8ea1964cebdfc9a0d907720dbf
#owner: https://api.github.com/users/rayschpp

#!/bin/bash
case "$1" in
  upgrade) sudo nixos-rebuild switch --upgrade ;;
  cleanup) szdo nix-env --delete-generations +5 ;;
  gc) sudo nix-collect-garbage -d ;;
  full) sysmaint upgrade && sysmaint cleanup && sysmaint gc ;;
esac