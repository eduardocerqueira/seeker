#date: 2025-07-11T17:14:10Z
#url: https://api.github.com/gists/633dec48faaab4e3af86ab0b9463eeea
#owner: https://api.github.com/users/agrif

#!/bin/bash
# -*- mode: shell-script -*-
set -e

# This hook shows the relevant commits and remote and confirms that you want
# to push this before allowing a push to complete.

# Comment to disable colors.
USE_COLOR="1"

remote="$1"
url="$2"

# zeros, same length as git hashes
zero=$(git hash-object --stdin < /dev/null | tr '[0-9a-f]' '0')

# check for color support
if [ ! -z "$USE_COLOR" ] && [ -t 1 ]; then
    ncolors=$(tput colors)
    if [ -n "$ncolors" ] && [ $ncolors -ge 8 ]; then
        crst="$(tput sgr0)"
        cbld="$(tput bold)"
        cblk="$(tput setaf 0)"
        cred="$(tput setaf 1)"
        cgrn="$(tput setaf 2)"
        cylw="$(tput setaf 3)"
        cblu="$(tput setaf 4)"
        cmag="$(tput setaf 5)"
        ccyn="$(tput setaf 6)"
        cwht="$(tput setaf 7)"
    fi
fi

found_changes=0
while read -r local_ref local_oid remote_ref remote_oid; do
    found_changes=1

    if [ "$local_oid" = "$zero" ]; then
        # delete ref
        echo "${cred}DELETE${crst} ${cbld}${remote_ref}${crst}"
    elif [ "$remote_oid" = "$zero" ]; then
        # create new ref
        echo "${cgrn}CREATE${crst} ${cbld}${remote_ref}${crst} (from local ${local_ref})"
    elif ! git merge-base --is-ancestor "$remote_oid" "$local_oid"; then
        # non-fastforward push
        echo "${cylw}MODIFY${crst} ${cbld}${remote_ref}${crst} (from local ${local_ref})"
        echo " (this is a ${cred}force push!${crst})"
        git log --pretty=oneline --left-right "${remote_oid}...${local_oid}"| while read -r mark commit msg; do
            commit=$(echo "$commit" | cut -c1-7)
            if [ "$mark" = ">" ]; then
                echo -n "$cgrn"
            elif [ "$mark" = "<" ]; then
                echo -n "$cred"
            fi
            echo " ${mark} ${commit}${crst} ${msg}"
        done
    else
        # fastforward push
        echo "${cylw}APPEND${crst} ${cbld}${remote_ref}${crst} (from local ${local_ref})"
        git log --pretty=oneline --cherry-mark --right-only "${remote_oid}...${local_oid}"| while read -r mark commit msg; do
            commit=$(echo "$commit" | cut -c1-7)
            if [ "$mark" = "+" ]; then
                echo -n "$cgrn"
            elif [ "$mark" = "-" ]; then
                echo -n "$cred"
            fi
            echo " ${mark} ${commit}${crst} ${msg}"
        done
    fi
done

if [ "$found_changes" -eq 0 ]; then
    # no changes to push
    exit 0
fi

echo
echo "You are about to push these changes to remote ${cbld}${remote}${crst}, at"
echo "${url}"
echo
echo -n "Push these changes? (y/N) "
read -n 1 -r reply < /dev/tty
echo

if echo "$reply" | grep -E '^[Yy]$' > /dev/null; then
    exit 0
else
    echo "Push canceled."
    exit 1
fi
