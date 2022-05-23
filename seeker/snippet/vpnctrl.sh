#date: 2022-05-23T17:03:09Z
#url: https://api.github.com/gists/09359662dca81466cd8357308971d371
#owner: https://api.github.com/users/vxsl

#!/bin/bash

VPN_NAME="my-vpn-name" # see output of "nmcli con show" to get this value
PASSWD_PATH="/this/could/be/any/path/.vpnpasswd.tmp" 
trap "rm -f $PASSWD_PATH" EXIT
if [[ ("$1" == "--up") ]]; then
  if [[ $(nmcli c s -a | grep $VPN_NAME) ]]; then
    notify-send "$VPN_NAME is already active."
    exit
  fi
  PASSWD=$(zenity --password)
  echo "vpn.secrets.password:$PASSWD" > $PASSWD_PATH 
  notify-send "$(nmcli c up $VPN_NAME passwd-file $PASSWD_PATH)"
elif [[ ("$1" == "--down") ]]; then
  if [[ -z $(nmcli c s -a | grep $VPN_NAME) ]]; then
    notify-send "$VPN_NAME is already inactive."
    exit
  fi
  notify-send $(nmcli c down $VPN_NAME)
fi
