#date: 2023-02-28T16:49:47Z
#url: https://api.github.com/gists/f1e2d081034d43d1140f3333e878da13
#owner: https://api.github.com/users/saileshbro

#!/bin/sh
PATH=/bin:/sbin:/usr/bin:/usr/sbin; export PATH

basedir="/Library/org.pqrs/PCKeyboardHack"
kextfile=''
uname=`uname -r`
case "${uname%%.*}" in
    10)
        kextfile="$basedir/PCKeyboardHack.10.6.kext"
        ;;
    11)
        kextfile="$basedir/PCKeyboardHack.10.7.kext"
        ;;
    12)
        kextfile="$basedir/PCKeyboardHack.10.7.kext" # Hack to use 10.7 kext in 10.8
        ;;
esac

if [ "x$kextfile" == 'x' ]; then
    exit 1
fi

if [ "$1" == 'unload' ]; then
    kextunload -b org.pqrs.driver.PCKeyboardHack
else
    kextload "$kextfile"
fi

exit 0