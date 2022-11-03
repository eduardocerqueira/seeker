#date: 2022-11-03T17:15:54Z
#url: https://api.github.com/gists/a71ee218a03d24f85d0e70257bee488d
#owner: https://api.github.com/users/NickNothom

#!/bin/bash

# Set defaults if not provided by environment
CHECK_DELAY=${CHECK_DELAY:-5}
CHECK_IP=${CHECK_IP:-8.8.8.8}
PRIMARY_IF=${PRIMARY_IF:-eth0}
PRIMARY_GW=${PRIMARY_GW:-10.0.0.1}
BACKUP_IF=${BACKUP_IF:-eth0}
BACKUP_GW=${BACKUP_GW:-10.0.0.6}

whoami

# Compare arg with current default gateway interface for route to healthcheck IP
gateway_add() {
  [[ "$1" = "$(ip r g "$CHECK_IP" | sed -rn 's/^.*via ([^ ]*).*$/\1/p')" ]]
}

# Cycle healthcheck continuously with specified delay
while sleep "$CHECK_DELAY"
do
  # If healthcheck succeeds from primary interface
  if ping -c1 rhea.local &>/dev/null
  then
    echo "Ping Success Primary"
    # Are we using the backup?
    if gateway_add "$BACKUP_GW"
    then # Switch to primary
      echo "Switch to PTP"
      ip r d default via "$BACKUP_GW" dev "$BACKUP_IF"
      ip r a default via "$PRIMARY_GW" dev "$PRIMARY_IF" metric 200
    fi
  else
    # Are we using the primary?
    if gateway_add "$PRIMARY_GW"
    then # Switch to backup
      echo "Switch to 4G"
      ip r d default via "$PRIMARY_GW" dev "$PRIMARY_IF"
      ip r a default via "$BACKUP_GW" dev "$BACKUP_IF" metric 200
    fi
  fi
done
