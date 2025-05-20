#date: 2025-05-20T17:10:19Z
#url: https://api.github.com/gists/41e185478ff9ca1891490eb4b7ebdd6b
#owner: https://api.github.com/users/simon-brooke

#!/bin/bash

# Script to set all origin URLs of all my projects in my workspace to the new 
# (forgejo) URL, replacing previous GitHub URLs. 

# Background: I keep all my active projects as subdirectories of the 
# directory `workspace`.

# Variables in capitals you can think of as configuration -- you will want
# to change these to account for the difference between your setup and mine

WORKSPACE=~/workspace

GHUSERNAME="simon-brooke"
GHURL="git@github.com:${GHUSERNAME}/"

FJUSERNAME="simon"
FJSSHPORT="1234"
FJURL="ssh://forgejo@git.journeyman.cc:${FJSSHPORT}/${FJUSERNAME}/"

pushd "$WORKSPACE"

for project in *
do
    if [ -d $project ]
    then
        pushd $project

        origin=`git remote -v | grep '^origin' | head -1 | awk '{print $2}'`

        echo "Project = ${project}; origin = ${origin}"

        echo $origin | grep "${GHURL}"

        if [ $? -eq 0 ]
        then
            # Is it one of mine or my fork of someone else's?
            git remote -v | grep '^upstream' 

            if [ $? -eq 0 ]
            then
                echo "...fork!"
            else
                echo "...mine!"

                # I really ought to check that the directory name and the 
                # repository name are identical, but all mine are.

                echo "Setting remote origin URL to ${FJURL}${project}"

                git remote set-url origin ${FJURL}${project}
                git fetch origin

                git remote -v | grep '^github' 

                if [ $? -eq 0 ]
                then 
                    echo "... github remote already set up for ${project}, not altering"
                else
                    git remote add github ${origin}
                fi
            fi
        fi

        popd
    fi
done

popd
