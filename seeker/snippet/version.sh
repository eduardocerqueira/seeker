#date: 2022-03-28T17:12:05Z
#url: https://api.github.com/gists/aeba4a044e8551013d5e9d6320b52c1d
#owner: https://api.github.com/users/Shahnama

#!/bin/bash

version="$1"
major=0
minor=0
build=0

# break down the version number into it's components
regex="([0-9]+).([0-9]+).([0-9]+)"
if [[ $version =~ $regex ]]; then
  major="${BASH_REMATCH[1]}"
  minor="${BASH_REMATCH[2]}"
  build="${BASH_REMATCH[3]}"
fi

# check paramater to see which number to increment
if [[ "$2" == "feature" ]]; then
  minor=$(echo $minor + 1 | bc)
elif [[ "$2" == "bug" ]]; then
  build=$(echo $build + 1 | bc)
elif [[ "$2" == "major" ]]; then
  major=$(echo $major+1 | bc)
else
  echo "usage: ./version.sh version_number [major/feature/bug]"
  exit -1
fi

# echo the new version number
echo "new version: ${major}.${minor}.${build}"