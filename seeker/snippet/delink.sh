#date: 2023-04-25T16:50:29Z
#url: https://api.github.com/gists/dc88fac155fccfdf7ddba9497870ab6e
#owner: https://api.github.com/users/pirogoeth

#!/usr/bin/env bash
# vim: set ai et ts=4 sts=4 sw=4 syntax=bash:
# -*- coding: utf-8 -*-

export DEBUG=no
export REMOVE_AFTER_DELINK=no
export SOURCE=

set -euo pipefail

function usage() {
    echo "$0: [-v|-R|-h] -s <file>"
    echo
    echo "  delink.sh - given a path, check if it's a symlink. if it is a symlink, resolve"
    echo "              the path to the target file, unlink the source file, and replace the"
    echo "              source file with the target file."
    echo
    echo "  arguments:"
    echo "      -R      remove the target after copying it over where source was"
    echo "      -s      source file to delink"
    echo "      -v      turn on debug messages"
    echo "      -h      this message"
    echo
    exit 127
}

function debug_print() {
    test $DEBUG == "yes" && printf "$1\n" "$*" || return 0
}

while getopts "vs:Rh" arg
do
    case $arg in
        v)
            export DEBUG=yes
            ;;
        s)
            export SOURCE=$OPTARG
            ;;
        R)
            export REMOVE_AFTER_DELINK=yes
            ;;
        *)
            usage
            exit 127
            ;;
    esac
done

if test -z "$SOURCE" ; then
    echo "source file (-s) was unset"
    usage
fi

debug_print "attempting to delink ${SOURCE}"

if test -L "$SOURCE" ; then
    target="$(readlink -n $SOURCE)"
    debug_print "${SOURCE} is linked to ${target}"
    unlink "${SOURCE}"
    if test $? != 0 ; then
        echo "error unlinking ${SOURCE}"
        exit 1
    fi

    if test -d "${target}" ; then
        if test "$DEBUG" == "yes" ; then
            cp -rv "${target}" "${SOURCE}"
        else
            cp -r "${target}" "${SOURCE}"
        fi
    else
        if test "$DEBUG" == "yes" ; then
            cp -v "${target}" "${SOURCE}"
        else
            cp "${target}" "${SOURCE}"
        fi
    fi

    if test "$REMOVE_AFTER_DELINK" == "yes" ; then
        if test "$DEBUG" == "yes" ; then
            rm -v "${target}"
        else
            rm "${target}"
        fi
    fi
else
    echo "${SOURCE} is not a symlink"
    exit 2
fi
