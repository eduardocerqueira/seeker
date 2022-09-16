#date: 2022-09-16T22:03:59Z
#url: https://api.github.com/gists/766a32dc57f9307a98f9b17a54fd0dfb
#owner: https://api.github.com/users/malkab

#!/bin/bash

# This script explains the shift technique to parse the command line. To
# see more command line parsing techniques, refer to the getopts.sh
# script example.

# Number of parameters

echo $#

# This represents the command itself

echo $0

# This loop pass through all the parameters. The shift function discards
# the parameter and moves to the next one. It can be used, for example,
# shift 2, which discards that number of parameters in a row. This is
# useful when parsing -o <value> parameters.

while [ "$1" ]
do
	echo $1
	shift 1
done

# This launches a command appending all passed command line arguments.
# It gently skips $0, which is the invoked script name itself. For
# example, "script_name -ls" will run "ls -ls".

ls $*
