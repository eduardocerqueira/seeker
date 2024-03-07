#date: 2024-03-07T16:56:57Z
#url: https://api.github.com/gists/476330fb8135d6d13c7db08da9c5c370
#owner: https://api.github.com/users/prestonmcgowan

#!/bin/bash

FILE=/var/log/speedtest/speedtest-`date +"%Y%m"`.results
printf -- '-%.0s' {1..80} >> $FILE
echo >> $FILE
date >> $FILE
/usr/local/bin/speedtest --progress=no >> $FILE
printf -- '-%.0s' {1..80} >> $FILE
echo >> $FILE

## Speedtest CLI
# https://www.speedtest.net/apps/cli

##
## Crontab
## 0 */6 * * * /usr/local/bin/speedtest-log.sh