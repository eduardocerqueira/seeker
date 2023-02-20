#date: 2023-02-20T16:56:36Z
#url: https://api.github.com/gists/16e4c28c1b52a1d583fd40bc560189ec
#owner: https://api.github.com/users/carlosleonam

#!/usr/bin/env zsh
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2017-08-25 14:59:30 (UTC+0200)

usage ()
{
    echo "Usage"
    echo "autogit [-i, -a filename, -s, -d filename]"
    echo "    • -i: initialize the autogit repository"
    echo "    • -a <filename>: add the given file under versioning"
    echo "    • -A : add all the files in the current directory, excepted hidden files, under versioning"
    echo "    • -s: get the status of the current repository"
    echo "    • -l: get the log of the current repository"
    echo "    • -d <filename>: show the diff history for the given file"
    echo "    • -d --word-diff: show the diff history with a word based diff"
    echo "    • --diff <commit hash or tag name> <filename>: show the differences between the current version of the file and the given commit hash"
    echo "    • --ls: list all the file under versioning"
    echo "    • -f: force to backup all files under versioning"
    echo "    • -v <commit hash or tag name> <filename>: show the version given by the <commit hash> for the given file"
    echo "    • -t message: create a tag with as tagname the date in format +%Y%m%d_%H%M (e.g.: 20180829_1643) and the given message"
    echo "    • --lt list tags"
    echo "    • -r <filename>: restore (git-checkout) deleted file"
    echo "    • -x <filename>: remove the given filename from versionning"
    echo "    • -X: clean the current repository from any versionning"
    echo "    • --prompt: get the system status as a prompt status"
    echo "    • --sync hostname:/path/to/dir/: synchronize the given host"
    exit
}

function autogit_prompt () {
    if [ -d ".autogit" ]; then
        git  --git-dir=.autogit status -s |
        awk 'BEGIN{var=1};{if ($1=="M" || $1=="D"){var=0}};END{if (var){printf "\033[1;32m•\033[0m "}else{printf "\033[1;31m•\033[0m "}}'
    fi
}

if [ $# -lt 1 ]; then
    usage
fi

if [ "$1" = "--prompt" ]; then
    autogit_prompt
    exit
fi

if [ "$1" = "-i" ]; then
    if [ -d ".autogit" ]; then
        echo "AUTOGIT repository"
        exit
    else
        # create bare repository for distant access
        mkdir .bare
        cd .bare
            git --bare init
        cd -
        if [ -d ".git" ]; then # Already a git repo. Preserve the .git file
            TMPDIR=$(mktemp -d)
            mv .git $TMPDIR/.
            git init
            mv .git .autogit
            mv $TMPDIR/.git .
        else
            git init
            mv .git .autogit
        fi
        CWD=$(pwd)
        git --git-dir=.autogit remote add origin $CWD/.bare
        # Create post-commit hook
        echo "#!/bin/sh
        git --git-dir=.autogit push origin master" > .autogit/hooks/post-commit
        chmod +x .autogit/hooks/post-commit
        # Set the GIT_DIR environment variable to .autogit
        echo "#export GIT_DIR=.autogit" >> .envrc
        direnv allow
    fi
fi

if [ "$1" = "--sync" ]; then
    ADDR=$2
    A=("${(@s/:/)ADDR}")
    HOSTNAME=$A[1]
    HOSTPATH=$A[2]
    CWD=$(pwd)
    MYHOSTNAME=$(hostname)
    ssh $HOSTNAME "mkdir -p $HOSTPATH && cd $HOSTPATH && git init && git remote add origin $MYHOSTNAME:$CWD/.bare && git pull origin master"
    echo "ssh $HOSTNAME 'cd $HOSTPATH && git pull origin master'" >> .autogit/hooks/post-commit
fi

function add_file () {
    if git --git-dir=.autogit ls-files | grep $1 ; then
        echo "File $1 already under versionning..."
    else
        git --git-dir=.autogit add "$1"
        git --git-dir=.autogit commit $1 -m "Added: $1"
        # Create the script file to be executed by incron:
        SCRIPTNAME=".$(echo $1 | tr '/' '%').incron"
        echo "cd $(pwd) && git --git-dir=.autogit commit $1 -m 'Modified: $1' ; incrontab --reload" > $SCRIPTNAME
        # Update the incrontab
        INCRONTABENTRY="$1:A IN_CLOSE_WRITE sh $SCRIPTNAME:A"
        incrontab -l | cat - =(echo $INCRONTABENTRY) | incrontab -
    fi
}
if [ "$1" = "-a" ]; then
    add_file $2
    exit
fi

if [ "$1" = "-A" ]; then
    for x in $(find . -type f -not -path '*/\.*' -printf '%P\n'); do
        add_file $x
    done
    exit
fi

if [ "$1" = "-x" ]; then
    SCRIPTNAME=".$2.incron"
    SCRIPTNAME=".$SCRIPTNAME:t"
    print "Remove incrontab entry for $2"
    incrontab -l | grep -v $SCRIPTNAME | incrontab -
    print "Remove Incron script for $2"
    rm -v $SCRIPTNAME
    print "Remove $2 from the git index"
    git --git-dir=.autogit rm --cached $2
    print "Commit changes..."
    git --git-dir=.autogit commit -a -m "$2 removed from the git index"
    print "Reloading incrontab"
    incrontab --reload
    exit
fi

if [ "$1" = "-X" ]; then
    print "Remove incrontab entries ..."
    incrontab -l | grep -v -f =(ls .*.incron) | incrontab -
    print "Remove Incron scripts..."
    rm -v .*.incron
    print "Remove Git files..."
    rm -rf .autogit
    print "Remove bare repo..."
    rm -rf .bare
    print "Reloading incrontab"
    incrontab --reload
    exit
fi

if [ "$1" = "-s" ]; then
    git --git-dir=.autogit status -uno
    exit
fi

if [ "$1" = "-l" ]; then
    git --git-dir=.autogit log --reverse
    exit
fi

if [ "$1" = "-d" ]; then
    git --git-dir=.autogit log --reverse -p $2
    exit
fi

if [ "$1" = "--diff" ]; then
    git --git-dir=.autogit diff $2 $3
    exit
fi

if [ "$1" = "--ls" ]; then
    git --git-dir=.autogit ls-files
    exit
fi

if [ "$1" = "-v" ]; then
    git --git-dir=.autogit show $2:$3
    exit
fi

if [ "$1" = "-r" ]; then
    git --git-dir=.autogit checkout $2
    exit
fi

if [ "$1" = "-f" ]; then
    git --git-dir=.autogit commit -a -m "Backup forced!"
    for SCRIPTNAME in .*.incron; do
        if $(incrontab -l | grep -q "$SCRIPTNAME"); then
            echo "$SCRIPTNAME already presents in incrontab"
        else
            TRACKEDFILE=$(echo "$SCRIPTNAME:t" | sed 's/.//' | sed 's/.incron//' | sed 's,%,/,g')
            INCRONTABENTRY="$TRACKEDFILE:A IN_CLOSE_WRITE sh $SCRIPTNAME:A"
            echo "Adding:\n$INCRONTABENTRY to incrontab"
            incrontab -l | cat - =(echo $INCRONTABENTRY) | incrontab -
        fi
    done
    incrontab --reload
    exit
fi

if [ "$1" = "-t" ]; then
    git --git-dir=.autogit tag -a $(date +%Y%m%d_%H%M) -m $2
    git tag
    exit
fi

if [ "$1" = "--lt" ]; then
    git --git-dir=.autogit tag -n
    exit
fi
