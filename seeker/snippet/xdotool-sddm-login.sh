#date: 2022-09-19T17:04:34Z
#url: https://api.github.com/gists/aab4a9a57e441aa9bc66f576620c3b1c
#owner: https://api.github.com/users/mikhailnov

#!/bin/bash

# install: xdotool, binutils (strings), procps (pgrep), sed, grep
# usage: "**********"

set -e
set -f

user="$1"
if [ -z "$user" ]; then
	echo "Define user: "**********"
	exit 1
fi

password= "**********"
if [ -z "$password" ]; then
	echo "Define password: "**********"
	exit 1
fi

set -u

if ! { pid="$(pgrep sddm-greeter)" && [ -d /proc/"$pid" ] ;}; then
	echo "SDDM Greeter is not running!"
	exit 1
fi

if [ "$(echo "$pid" | grep -c .)" != 1 ]; then
	echo "There cannot be more than one sddm-greeter!"
	exit 1
fi

display="$(set -e -o pipefail && strings /proc/"$pid"/environ | grep ^DISPLAY= | head -n 1 | sed -e 's,^DISPLAY=,,')"
xauthority="$(set -e -o pipefail && strings /proc/"$pid"/environ | grep ^XAUTHORITY= | head -n 1 | sed -e 's,^XAUTHORITY=,,')"
if [ -z "$display" ] || [ -z "$xauthority" ]; then
	echo "Error getting variables!"
	exit 1
fi

users=()
while read -r line
do
	users+=("$line")
done < <(awk -F ':' '$3 >= 500 {print $1}' /etc/passwd)

c=0
for (( i = 0; i < ${#users[@]}; i++ ))
do
	if [ "${users[$i]}" = "$user" ]; then
		c=$((++c))
		break
	fi
done

if [ "$c" -le 0 ]; then
	echo "User $user does not exist or has UID <500!"
	exit 1
fi

# wake up sddm greeter
DISPLAY="$display" XAUTHORITY="$xauthority" xdotool mousemove 10 10
sleep 1

# choose user
# first click left arrow enough times to choose the first user if another one is chosen
for (( i = 0; i <= $((c+1)); i++ ))
do
	DISPLAY="$display" XAUTHORITY="$xauthority" xdotool key Left
done
# and now click Right arrow to choose the needed user
# (hoping that sddm does not sort users in a special way)
if [ "$c" -gt 0 ]; then
	for (( i = 0; i < "$c"; i++ ))
	do
		DISPLAY="$display" XAUTHORITY="$xauthority" xdotool key Right
	done
fi

DISPLAY= "**********"="$xauthority" xdotool type "$password"
sleep 1
DISPLAY="$display" XAUTHORITY="$xauthority" xdotool key Return
play" XAUTHORITY="$xauthority" xdotool key Return
