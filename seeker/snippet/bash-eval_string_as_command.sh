#date: 2022-09-16T22:05:53Z
#url: https://api.github.com/gists/b721024078d4a2e7b30c9c05b2bc31b1
#owner: https://api.github.com/users/malkab

#!/bin/bash

# Evaluates a string as a command

COMMAND='ls'

COMMAND="${COMMAND} -lh"

eval $COMMAND "*.sh"
