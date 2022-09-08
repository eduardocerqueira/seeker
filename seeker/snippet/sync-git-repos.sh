#date: 2022-09-08T17:05:45Z
#url: https://api.github.com/gists/be36d8e9d891824bd38cb001cac07093
#owner: https://api.github.com/users/danielbdias

#!/bin/sh

for code_directory in $(ls -d */)
do
    # go to directory
    cd $code_directory

    code_directory_default_branch=$(git rev-parse --abbrev-ref origin/HEAD | sed 's@^origin/@@')
    code_directory_current_branch=$(git rev-parse --abbrev-ref HEAD | sed 's@^origin/@@')

    echo "Updating $code_directory ..."
    echo "Current branch: $code_directory_current_branch"
    echo "Default branch: $code_directory_default_branch"
    
    echo ""
    echo "Updating current branch..."
    git pull origin $code_directory_current_branch

    if [ $code_directory_default_branch != $code_directory_current_branch ] 
    then
        echo ""
        echo "Updating default branch..."
        git checkout $code_directory_current_branch
        git pull origin $code_directory_current_branch
        git checkout -
    fi

    echo ""

    # go to root
    cd ..
done