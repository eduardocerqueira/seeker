#date: 2022-07-13T17:11:33Z
#url: https://api.github.com/gists/baa36bf5d4b91b50da81313281dbc887
#owner: https://api.github.com/users/basilioss

#!/usr/bin/env bash

# You can call this script like this:
# $ ./brightness-control.sh up
# $ ./brightness-control.sh down

# Script inspired by these wonderful people:
# https://github.com/dastorm/volume-notification-dunst/blob/master/volume.sh
# https://gist.github.com/sebastiencs/5d7227f388d93374cebdf72e783fbd6a

# Icon path: /usr/share/icons/Papirus/16x16/apps
icon="display-brightness"

function get_brightness {
  xbacklight -get | cut -d '.' -f 1
}

function send_notification {
  dunstify -i "$icon" -r 5555 -u normal -h int:value:"$1" "Brightness: $1%"
}

case $1 in
  up)
    xbacklight -inc 5
    send_notification "$(get_brightness)"
    ;;
  down)
    xbacklight -dec 5
    send_notification "$(get_brightness)"
    ;;
esac