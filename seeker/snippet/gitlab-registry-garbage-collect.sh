#date: 2023-10-10T16:57:02Z
#url: https://api.github.com/gists/e89424586555169f338de68907d60fef
#owner: https://api.github.com/users/chrisblech

#!/bin/bash

LOGFILE="/tmp/gitlab-registry-garbage-collect.log"

REGEXOK='^ok: run: registry: \(pid [0-9]+\) [0-5]s$'

gitlab-ctl registry-garbage-collect 2>&1 | grep -Ev '^INFO\[0001\] .+ service=registry .+' | grep -v ' level=info ' >$LOGFILE

OK="$(cat $LOGFILE \
  | grep -v ' CPU quota undefined' \
  | grep -Ev '^Running garbage-collect .+ this might take a while\.\.\.$' \
  | grep -Ev '^ok: down: registry: [0-9]s, normally up$')"

[[ $OK =~ $REGEXOK ]] && exit 0

echo "Passt nicht - OK=$OK -"

cat $LOGFILE

exit 1
