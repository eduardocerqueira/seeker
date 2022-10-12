#date: 2022-10-12T17:24:11Z
#url: https://api.github.com/gists/527bdd32c554e4a100cbf49dbdbb802e
#owner: https://api.github.com/users/sergiocampama

#!/usr/bin/env bash

function confirm {
  if [[ $FORCE == 1 ]]; then
    return 0
  fi

  local ok
  read -r ok

  [[ $ok == "$1" ]]
}

function remove_profile {
  profiles -R -p $1
}

function kill_loop {
  while true; do
    kill -9 $(ps aux | grep /opt/fleetsmith | awk '{print $2}')
    sleep 0.1
  done
}

function on_exit {
  [[ -z $pid ]] && return
  >&2 echo "Killing kill loop :)"
  kill -9 $pid
}

function main {
  [[ $UID != 0 ]] && >&2 echo "this script must be run as root! sudo it"
  [[ $UID == 0 ]] || exit 1

  FORCE=1

  trap on_exit EXIT


  fleet_profiles=$(profiles -Lv | grep "name: $4" -4 | awk -F": " '/attribute: profileIdentifier/{print $NF}' | grep fleetsmith)

  # Create a "disabled" folder. This means none of these services will run on
  # startup. They are still on the system, so you can check them out!

  mkdir -p /Library/LaunchAgents/disabled
  mkdir -p /Library/LaunchDaemons/disabled

  >&2 echo "Starting kill loop..."
  kill_loop &> /dev/null &
  pid=$!

  pushd /Library/LaunchAgents
    >&2 echo "Disabling fleetsmith LaunchAgents ($PWD/disabled)"
    mv *fleetsmith* disabled &> /dev/null
  popd

  pushd /Library/LaunchDaemons
    >&2 echo "Disabling fleetsmith LaunchDaemons ($PWD/disabled)"
    mv *fleetsmith* disabled &> /dev/null
  popd

  >&2 echo "Removing FleetSmith profiles:"
  for profile in $fleet_profiles; do
    >&2 printf "Remove: $profile ? n[y] "
    confirm "n" || remove_profile $profile
  done

  >&2 echo "Now it's a good idea to restart your computer"
  if [[ $FORCE == 0 ]]; then
    >&2 printf "Say \"yes\" to restart it: "
    confirm "yes" && reboot && wait
  fi

  >&2 echo "You should restart now"
}

pushd() { builtin pushd $1 > /dev/null; }
popd() { builtin popd > /dev/null; }

main "$@"