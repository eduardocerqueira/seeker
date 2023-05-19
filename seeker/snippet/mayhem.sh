#date: 2023-05-19T17:01:48Z
#url: https://api.github.com/gists/8e1c6f81626a110f8a6e8df1285ae0e1
#owner: https://api.github.com/users/lbfalvy

#!/bin/bash
npm install `npm outdated | tail -n +2 | awk '{ print $1 "@latest" }' | tr '\n' ' '`
