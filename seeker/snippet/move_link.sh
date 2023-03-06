#date: 2023-03-06T16:42:48Z
#url: https://api.github.com/gists/c7e43143d3a4f0982fb501469f54fb05
#owner: https://api.github.com/users/big-thousand

#!/bin/sh
set -e
for link; do
    test -h "$link" || continue

    dir=$(dirname "$link")
    reltarget=$(readlink "$link")
    case $reltarget in
        /*) abstarget=$reltarget;;
        *)  abstarget=$dir/$reltarget;;
    esac

    rm -fv "$link"
    cp -afv "$abstarget" "$link" || {
        # on failure, restore the symlink
        rm -rfv "$link"
        ln -sfv "$reltarget" "$link"
    }
done