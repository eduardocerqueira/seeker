#date: 2024-01-02T17:05:33Z
#url: https://api.github.com/gists/29e11f71fe76d75516f66f9df9e26b1b
#owner: https://api.github.com/users/heitor-galindo

#!/bin/bash

readonly bitbucket_app_password= "**********"
readonly bitbucket_user='xxxxxxxxxxxxxxxxxxx'
repository_foder="$HOME/Documents/xxxxxxxx"

if [[ -d "$repository_foder" ]]; then
    echo "Retrieving repositories list...wait"
    repositories=$(curl -s --user $bitbucket_user: "**********"://api.bitbucket.org/2.0/repositories/xxxxxxx?pagelen=100 | jq -r '.values | .[].links.clone | .[1].href')
    cd $repository_foder
    for repository in $repositories; do

        repository_name="${repository##*/}"
        repository_name="${repository_name%.git}"

        echo -e "---\n> Repository: $repository"
        if [[ -d "${repository_foder}/${repository_name}" ]]; then
            pushd "${repository_foder}/${repository_name}" > /dev/null
                git checkout master && git pull
            popd > /dev/null
        else
            git clone $repository
        fi
    done
else
    mkdir $supersim_directory
fi
upersim_directory
fi
