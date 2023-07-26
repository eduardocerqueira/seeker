#date: 2023-07-26T16:55:02Z
#url: https://api.github.com/gists/662aa6034cc6c90ba387fe46b5018c53
#owner: https://api.github.com/users/jimdiroffii

#!/bin/bash

# Simulates job duration by sleeping for a random period.

sleep_time=$((1 + RANDOM % 10))
echo "Script $1 sleeping for $sleep_time seconds"
sleep $sleep_time
echo "Script $1 done"