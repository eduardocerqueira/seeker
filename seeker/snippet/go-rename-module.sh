#date: 2021-09-01T17:15:04Z
#url: https://api.github.com/gists/cddcc4a7afb033a0596904a3803e05d5
#owner: https://api.github.com/users/zakedy

#!/bin/bash

current_module_name() {
	awk '$1 == "module" { print $2 }' go.mod
}

cut_final_slash() {
	echo "$name_to" | sed 's/\/$//'
}

usage() {
	echo
	echo "Replace module name (or part of it) recursively"
	echo
	echo "usage:"
	echo "	$0  old_path new_path"
	echo
	echo "current module name:   $(current_module_name)"
	exit $1
}

[ $# -lt 2 ] && usage 1
name_from="$1"
name_to="$2"

[ -r go.mod ] || {
	echo "could not find file go.mod"
	exit 2
}

modname=$(current_module_name)

echo "current name:  $modname"
echo "replace"
echo "        from:  $name_from"
echo "          to:  $name_to"

if [[ $modname != $name_from* ]]; then
	echo "module name does not match current name"
	exit 3
fi

sed -i .bkp -e  "s,$name_from,$name_to,g" go.mod
find . -type f -name '*.go' -exec sed -i '' -e "s,$name_from,$name_to,g" {} \;