#date: 2023-08-15T16:48:35Z
#url: https://api.github.com/gists/c71d9c110a74844a41906337d22caff6
#owner: https://api.github.com/users/mierzejk

#!/usr/bin/env bash

args=$@

function str_strip() {
 xargs echo -n <<< $1
}

function has_arg() {
 [[ $args =~ (^|[[:blank:]])$1([[:blank:]]|$) ]]
}

DEVICE_NAME=$(str_strip "$DEVICE_NAME")
[[ ${DEVICE_NAME:=$(str_strip "$1")} ]] || exit 1
DEVICE_NUM=$(str_strip "$DEVICE_NUM")
[[ ${DEVICE_NUM:=$(str_strip "$2")} ]] || DEVICE_NUM=0

function xinput_pointer_ids() {
 pref="^[^[:alnum:]]*↳"
 xinput list --short | sed -n -E -e "0,/${pref/↳/}Virtual[[:blank:]]core[[:blank:]]pointer.*master[[:blank:]]pointer/Id" -e "/$pref/"'!q1' -e "s/$pref[[:blank:]]*//; p" | perl -ne "s|^$1\s+id=(\d+)\s+.*?pointer.*$|\$1|i && print"
}

function name_by_id() {
 xinput --list-props $1 | awk 'BEGIN{FS=":[ \t]*"}/^[ \t]*Device Node \(/ {print substr($2, 2, length($2)-2)}'
}

function is_usb_intfnum_by_id() {
 if [[ ${#2} -lt 2 ]]; then
  num="00${2}"
  num=${num: -2}
 else num=$2; fi
 [[ $(udevadm info --query=property --name=$(name_by_id $1) 2>/dev/null | awk 'BEGIN{FS="="}/^[ \t]*ID_USB_INTERFACE_NUM/ {print $2}') = $num ]]
}

ids=($(xinput_pointer_ids "$DEVICE_NAME"))
[[ ${#ids[@]} -eq 1 ]] && ! has_arg "-f" && ! has_arg "--force" && echo -n $ids && exit 0
for id in ${ids[@]}; do
 if $(is_usb_intfnum_by_id $id "$DEVICE_NUM"); then echo -n $id && exit 0; fi
done
exit 1