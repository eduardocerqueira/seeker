#date: 2022-04-15T17:06:04Z
#url: https://api.github.com/gists/b7a2e5df97d54ce24a3c71299186ee9c
#owner: https://api.github.com/users/Gems

#!/usr/bin/env bash

LOG_FILE=/dev/null

exec > >(tee -a $LOG_FILE >&1)
exec 2> >(tee -a $LOG_FILE >&2)

if [ -n "$GIT_COMMITTER_DATE" ]; then
  FST="--faked-system-time $(date -j -f '%Y-%m-%dT%H:%M:%S' $GIT_COMMITTER_DATE +'%s')"
fi

echo $FST "$@" >>$LOG_FILE

gpg $FST "$@"
