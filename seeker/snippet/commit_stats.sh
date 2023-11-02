#date: 2023-11-02T16:52:42Z
#url: https://api.github.com/gists/91f42c1d3357ced155afb99c3c41bacc
#owner: https://api.github.com/users/nagasivanath

#!/bin/bash

dir="$1"

# No directory has been provided, use current
if [ -z "$dir" ]
then
    dir="`pwd`"
fi

# Make sure directory ends with "/"
if [[ $dir != */ ]]
then
	dir="$dir/*"
else
	dir="$dir*"
fi

# Loop all sub-directories
for f in $dir
do
	# Only interested in directories
	[ -d "${f}" ] || continue

	echo -en "\033[0;35m"
	echo "${f}"
	echo -en "\033[0m"

	# Check if directory is a git repository
	if [ -d "$f/.git" ]
	then
		mod=0
		cd $f

		# Check for modified files
		# git shortlog -scne --all --no-merges  --since=6.month
		git log --since="6 month ago" --shortstat --pretty="%cE" | sed 's/\(.*\)@.*/\1/' | grep -v "^$" | awk 'BEGIN { line=""; } !/^ / { if (line=="" || !match(line, $0)) {line = $0 "," line }} /^ / { print line " # " $0; line=""}' | sort | sed -E 's/# //;s/ files? changed,//;s/([0-9]+) ([0-9]+ deletion)/\1 0 insertions\(+\), \2/;s/\(\+\)$/\(\+\), 0 deletions\(-\)/;s/insertions?\(\+\), //;s/ deletions?\(-\)//' | awk 'BEGIN {name=""; files=0; insertions=0; deletions=0;} {if ($1 != name && name != "") { print name ": " files " files changed, " insertions " insertions(+), " deletions " deletions(-), " insertions-deletions " net"; files=0; insertions=0; deletions=0; name=$1; } name=$1; files+=$2; insertions+=$3; deletions+=$4} END {print name ": " files " files changed, " insertions " insertions(+), " deletions " deletions(-), " insertions-deletions " net";}'
		cd ../
	else

		echo "Not a git repository"
	fi

	echo
done
