#date: 2021-10-26T16:52:58Z
#url: https://api.github.com/gists/1e2797b14b9f28ceaf27d7b2214fe78f
#owner: https://api.github.com/users/onthedock

#!/bin/bash

# A simple test script to demonstrate how to find the
# "absolute path" at which a script is running. Used
# to avoid some of the pitfals of using 'pwd' or hard-
# coded paths when running scripts from cron or another
# directory.
#
# Try it out:
# run the script from the current directory, then 
# cd.. and run it again (using the file path).
#
# You can see that CURDIR changes depending on where
# you run the script from, but ABSDIR stays the same.

CURDIR=`/bin/pwd`
BASEDIR=$(dirname $0)
ABSPATH=$(readlink -f $0)
ABSDIR=$(dirname $ABSPATH)

echo "CURDIR is $CURDIR"
echo "BASEDIR is $BASEDIR"
echo "ABSPATH is $ABSPATH"
echo "ABSDIR is $ABSDIR"
