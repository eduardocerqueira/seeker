#date: 2021-10-12T17:14:55Z
#url: https://api.github.com/gists/f2446fdf9026252179cea6a9b96bf71d
#owner: https://api.github.com/users/r0nk

#!/bin/bash

while true; do
        if [ "$(echo quoth the raven | netcat -l 80)" == "nevermore" ] ;then
                echo serving flag on port 1337
                cat flag | timeout 10 netcat -l 1337
        fi
done
