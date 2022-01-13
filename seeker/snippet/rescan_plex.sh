#date: 2022-01-13T17:12:12Z
#url: https://api.github.com/gists/f62a57429ae3a346743462a0a06e0d7f
#owner: https://api.github.com/users/davromaniak

#!/bin/bash

# Rescan all Plex libraries
#
# Simple rescan :
## ./rescan_plex.sh
#
# if you want to run extra actions, use the var EXTRA_ACTIONS :
## EXTRA_ACTIONS="analyze generate" ./rescan_plex.sh
## analyze and generate can be time/ressources consuming BTW...
#

## Plex runs with this locale
LANG="en_US.UTF-8"
## Grab the env vars from Plex systemd unit and export them (needed to avoid Plex having a stroke)
eval $(systemctl cat plexmediaserver.service  | grep ^Environment | sed -e "s/^Environment=/export /")

IFS=':'
# stdbuf needed because Plex fucks up the output and it cannot be piped or used in a variable, may be better/cleaner ways to do it
stdbuf -oL -- ${PLEX_MEDIA_SERVER_HOME}/Plex\ Media\ Scanner --list |
  while read plexlibID plexlibName; do
    for action in scan refresh ${EXTRA_ACTIONS}; do
      echo -n "${action^} in progress for ${plexlibName/ /}... "
      ${PLEX_MEDIA_SERVER_HOME}/Plex\ Media\ Scanner --${action} --section $((plexlibID)) && echo "OK" || echo "ERROR" 1>&2
    done
  done