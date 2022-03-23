#date: 2022-03-23T17:13:41Z
#url: https://api.github.com/gists/5a11536dee2908fd568b38005f6b3644
#owner: https://api.github.com/users/ninp0

#!/bin/bash --login
usage() {
  echo "${0} <gui app name> <desktop coordinates e.g. 0,0,1700,1200,185"
  exit 1
}

if [[ $# == 2 ]]; then
  gui_app="${1}"
  coordinates="${2}"
  wmctrl -l | grep -i $gui_app | awk '{print $1}' | while read win_id; do
    transset-df -i $win_id 0.9
    wmctrl -ir $win_id -e $coordinates
  done
else
  usage
fi