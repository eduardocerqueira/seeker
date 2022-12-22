#date: 2022-12-22T16:39:45Z
#url: https://api.github.com/gists/dd33741d6dcb046f7c2e7f0a60d9cefc
#owner: https://api.github.com/users/natenho

#!/bin/bash -x

# Get the last tag
last_tag=$(git describe --tags 2> /dev/null)

# If the last tag does not exist, get the commit hash of the last commit
if [ $? -ne 0 ]; then
  last_tag="HEAD~1"
fi

# Get the list of modified directories
modified_dirs=$(git diff --dirstat=files,0 $last_tag HEAD | awk '{print $2}')

# Print the modified directories
echo " modified directories: $modified_dirs"