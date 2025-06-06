#date: 2025-06-06T16:50:26Z
#url: https://api.github.com/gists/df5fdcec99399f2ffc99608122393bf7
#owner: https://api.github.com/users/rawnly

#!/usr/bin/env bash

## Usage
# better_cat <file>
# better_cat package.json -r '[.name, .version] | join("@")'
# better_cat README.md

if [ -z "$1" ]; then
	echo "Usage: better_cat <file>"
	exit 1
fi

file="$1"
filename=$(basename "$file")
extension="${filename##*.}"

if [ -z "$filename" ]; then
	echo "Invalid file name: $filename"
	exit 1
fi
shift

if [[ $extension = "md" ]]; then
	glow <"$file"
elif [[ $extension = "json" ]]; then
	# Check if there are any arguments
	if [ $# -eq 0 ]; then
		# No arguments, just display the JSON
		jq '.' "$file"
	else
		# Pass the remaining arguments to jq
		jq "$@" "$file"
	fi
else
	bat --theme=1337 --style=header,grid,snip,changes "$file" "$@"
fi
