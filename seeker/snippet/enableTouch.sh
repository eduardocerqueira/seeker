#date: 2026-02-27T17:10:01Z
#url: https://api.github.com/gists/ffae179e59da8ac00d4f315a33fb45b8
#owner: https://api.github.com/users/raphaelkaique1

#!/bin/bash

mapfile -t Touch_Controller_id < <(
	xinput | grep -oE '.*Touch Controller.*id=[0-9]+' | grep -oE '[0-9]+'
)

mapfile -t xrandr_connected_id < <(
	xrandr | grep -oE '.* connected .*' | grep -oE '^.*-[0-9]+' | grep -oE '[0-9]+'
)

e=0

for i in ${!Touch_Controller_id[@]}; do
	if [[ $i -gt 0 && $(($i % 2)) -eq 0 ]]; then
		e=$((e+1))
	fi
	xinput map-to-output "${Touch_Controller_id[$i]}" DP-"${xrandr_connected_id[$e]}"
done