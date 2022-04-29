#date: 2022-04-29T16:50:47Z
#url: https://api.github.com/gists/82563373ab75649bfdc2f85b4cb372f3
#owner: https://api.github.com/users/Zren

#!/bin/bash
# To run anywhere with "git lsdir"
# sudo mv ./git-lsdir.sh /usr/local/bin/git-lsdir
# sudo chmod +x /usr/local/bin/git-lsdir

Red='\033[0;31m'
Orange='\033[0;33m'
LightRed='\033[91m'
LightGreen='\033[92m'
Yellow='\033[93m'
LightBlue='\033[94m'
ResetColors='\033[0m'

function printHelp ()
{
	echo "git lsdir -n [numCommits] -C [dirPath] --color"
	echo "Eg: git lsdir"
	echo "Eg: git lsdir -C ~/Code/"
	echo "Eg: git lsdir -n 5"
	echo "Eg: git lsdir --color | cat"
	echo "Note: [numCommits] defaults to 0"
}

# Options
isPipe=false
useColors=true
numCommits='0'
dirPath="$PWD"
logColorArg='--color'

if [ ! -t 1 ]; then
	isPipe=true
	useColors=false
fi

# https://stackoverflow.com/a/14203146/947742
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
	case $1 in
		-n)
			if [ ! -z "$2" ]; then
				# Check if input is an integer: https://stackoverflow.com/a/808740/947742
				[ "$2" -eq "$2" ] 2>/dev/null
				if [ $? -ne 0 ]; then
					printHelp
					exit 1
				else
					numCommits="$2"
				fi
			fi
			shift # past argument
			shift # past value
			;;
		--color)
			useColors=true
			shift # past argument
			;;
		-C)
			if [ -d "$2" ]; then
				dirPath="$2"
			else
				printHelp
				exit 1
			fi
			shift # past argument
			shift # past value
			;;
		-*|--*)
			echo "Unknown option $1"
			exit 1
			;;
		*)
			POSITIONAL_ARGS+=("$1") # save positional arg
			shift # past argument
			;;
	esac
done
set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters


# echo "numCommits: $numCommits"
# echo "dirPath: $dirPath"
# echo "useColors: $useColors"

if ! $useColors ; then
	Red=''
	Orange=''
	LightRed=''
	LightGreen=''
	Yellow=''
	LightBlue=''
	ResetColors=''
	logColorArg=''
fi

for f in `ls "$dirPath" | cat`; do
	if git -C "$dirPath/$f" rev-parse --git-dir > /dev/null 2>&1; then
		branchName=`git -C "$dirPath/$f" branch --show-current`
		gitDesc=`git -C "$dirPath/$f" describe --always --tags`
		if $useColors ; then
			gitDesc="$(echo "$gitDesc" | sed 's/-/, '"\\${LightBlue}"'+/' | sed 's/-g/, '"\\${Orange}"'/')"
		else
			gitDesc="$(echo "$gitDesc" | sed 's/-/, +/' | sed 's/-/, /')"
		fi
		echo -e "$f ${LightRed}(${LightRed}${branchName}, ${Yellow}${gitDesc}${LightRed})${ResetColors}"
		commitsSince=`git -C "$dirPath/$f" describe --always --tags | sed 's/-/ /g' | awk '{print $2;}'`
		if [ ! -z "$commitsSince" ]; then
			lastTag=`git -C "$dirPath/$f" describe --always --tags | sed 's/-/ /g' | awk '{print $1;}'`
			colorArg=''
			git -C "$dirPath/$f" log --oneline $logColorArg --max-count="${numCommits}" "${lastTag}..HEAD" -- | sed 's/^/    /'
		fi
	else
		echo "$f"
	fi
done
