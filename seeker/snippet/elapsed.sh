#date: 2025-07-28T16:56:20Z
#url: https://api.github.com/gists/0e947c5d633a8e8827cc9657a8e25102
#owner: https://api.github.com/users/tyholling

#!/bin/bash

start=$(date +%s)

watch -ct -n1 "
t=\$(( \$(date +%s) - $start ))" '
printf "%02d:%02d:%02d\n" $(( t / 3600 )) $(( (t % 3600) / 60 )) $(( t % 60 ))
'