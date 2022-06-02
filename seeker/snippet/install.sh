#date: 2022-06-02T16:56:47Z
#url: https://api.github.com/gists/e553ec94c33c75bae125025dfd27a317
#owner: https://api.github.com/users/littleneko

#!/bin/bash

CYAN="$(tput bold; tput setaf 6)"
RESET="$(tput sgr0)"

curl https://gist.githubusercontent.com/thpryrchn/c0ea1b6793117b00494af5f05959d526/raw/ccdl.command -o "/Applications/Adobe Packager.command"
chmod +x "/Applications/Adobe Packager.command"

clear

echo "${CYAN}Done! You can now start /Applications/Adobe Packager.command to begin${RESET}"
exit