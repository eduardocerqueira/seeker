#date: 2022-01-17T17:10:31Z
#url: https://api.github.com/gists/5db2a8efb07c1fa666e740f8c0b72b6a
#owner: https://api.github.com/users/aziis98

#!/bin/bash

# Cleans unlinked dotfiles in all used folders
#
# Author: aziis98
# Date: 06/01/2021

set -euo pipefail

TRACKED_DOTFILES_FILE=~/.cache/dotfiles/tracked

if [[ ! -e "$TRACKED_DOTFILES_FILE" ]]; then
	echo -e "Run \"dotfiles_link\" at least once to start tracking symlinks!"
	exit 1
fi

DELETE_COUNT=0

while read tracked_file; do
	if [[ ! -e "$tracked_file" ]]; then
		DELETE_COUNT=$(($DELETE_COUNT + 1))
		sed -i "\|$tracked_file|d" $TRACKED_DOTFILES_FILE
		rm -v "$tracked_file"
	fi
done < "$TRACKED_DOTFILES_FILE"

echo "Deleted $DELETE_COUNT broken link"$([[ "$DELETE_COUNT" -ne 1 ]] && echo -e "s")
