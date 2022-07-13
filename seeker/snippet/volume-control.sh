#date: 2022-07-13T17:11:33Z
#url: https://api.github.com/gists/baa36bf5d4b91b50da81313281dbc887
#owner: https://api.github.com/users/basilioss

#!/usr/bin/env bash

# You can call this script like this:
# $ ./volume-control.sh up
# $ ./volume-control.sh down
# $ ./volume-control.sh mute

# Script modified from these wonderful people:
# https://github.com/dastorm/volume-notification-dunst/blob/master/volume.sh
# https://gist.github.com/sebastiencs/5d7227f388d93374cebdf72e783fbd6a

iconSound="audio-volume-high"
iconMuted="audio-volume-muted"

function get_volume {
  pamixer --get-volume
}

function send_notification {
  if [ "$(pamixer --get-mute)" = true ]; then
    dunstify -i $iconMuted -r 2593 -u normal "mute"
  else
    dunstify -i $iconSound -r 2593 -u normal -h int:value:"$1" "Volume: $1%"
  fi
}

case $1 in
  up)
    pamixer --unmute
    pamixer --increase 5
    send_notification "$(get_volume)"
    ;;
  down)
    pamixer --unmute
    pamixer --decrease 5
    send_notification "$(get_volume)"
    ;;
  mute)
    pamixer --toggle-mute
    send_notification "$(get_volume)"
    ;;
esac