#date: 2024-12-06T17:07:18Z
#url: https://api.github.com/gists/f318b8dfcf2249d2637990c23600e50f
#owner: https://api.github.com/users/AlextheYounga

#!/bin/bash
# Switch PHP versions using Homebrew
# Example usage: `phps 8.1`
# Add to profile with `source phps.sh`

function phps() {
	local new_version=$1
	local php_output=$(php -v)
	local current_version=$(echo "$php_output" | awk '/^PHP/ {print $2}' | cut -d. -f1,2)
	echo "Switching PHP version from $current_version to $new_version"

	brew unlink php@$current_version
	brew link php@$new_version

	printf "Now using PHP version:\n"
	php -v
}