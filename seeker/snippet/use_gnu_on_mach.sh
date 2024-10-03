#date: 2024-10-03T17:12:01Z
#url: https://api.github.com/gists/d7a6b2d4d9e05f6c686e445af2c5a666
#owner: https://api.github.com/users/drewbrokke

#!/bin/bash

# First install the GNU utils with Homebrew
# Reference this article: https://apple.stackexchange.com/questions/69223/how-to-replace-mac-os-x-utilities-with-gnu-core-utilities
# They can actually configured to replace the built-in tools without a prefix, but we don't need to require that just for bundlebuster

# This is the command from the SO answer:
# brew install coreutils findutils gnu-tar gnu-sed gawk gnutls gnu-indent gnu-getopt grep

# echo g-prefixed command if found, otherwise use base command
function check_command() {
	_command="$1"

	if command -v "g${_command}" &>/dev/null ; then
		echo "g${_command}"
	else
		echo "${_command}"
	fi
}

# List the commands you want to substitue with GNU utils
cmds=(
	find
	grep
	sed
)

for cmd in "${cmds[@]}" ; do
	# Get the command that we want to use
	resolved_cmd="$(check_command "${cmd}")"

	# Declare a variable with a dynamic name
	declare "cmd_${cmd}"="${resolved_cmd}"

	# Just for debugging
	echo "cmd_${cmd} is assigned to ${resolved_cmd}"
done

# Example usages
git ls-files | $cmd_grep gradle | $cmd_sed 's/gradle/GRRREAT-le/g'
