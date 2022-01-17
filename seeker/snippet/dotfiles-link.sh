#date: 2022-01-17T17:10:31Z
#url: https://api.github.com/gists/5db2a8efb07c1fa666e740f8c0b72b6a
#owner: https://api.github.com/users/aziis98

#!/bin/bash

# Links all files marked with "TARGET_FILE=..." to their target location
#
# Author: aziis98
# Date: 06/01/2021

set -euo pipefail

TRACKED_DOTFILES_FILE=~/.cache/dotfiles/tracked
mkdir -p "$(dirname "$TRACKED_DOTFILES_FILE")"

# Extract a TSV table of SOURCE_FILE and TARGET_FILE
grep -e "^# TARGET_PATH=" -r . | sed -e 's/:# TARGET_PATH=/\t/g' | \
while read line; do
	SOURCE_FILE="$(echo "$line" | cut -f1)"
	TARGET_PATH="$(echo "$line" | cut -f2 | sed -e "s|~|$HOME|")"

	# Track directories with a symlink
	echo "$TARGET_PATH" >> $TRACKED_DOTFILES_FILE

	ln -sfv "$(realpath -s "$SOURCE_FILE")" "$TARGET_PATH"
done

# Dedup the cache file
cat $TRACKED_DOTFILES_FILE | sort | uniq > "$TRACKED_DOTFILES_FILE.tmp"
mv "$TRACKED_DOTFILES_FILE.tmp" "$TRACKED_DOTFILES_FILE"


