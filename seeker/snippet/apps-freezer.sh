#date: 2022-02-16T17:02:23Z
#url: https://api.github.com/gists/fbfa6b0ee9244d7b59cc01bbf8435fa1
#owner: https://api.github.com/users/danykey

#!/bin/bash

### Info ###
# This script freez inactive desktop applications to save CPU and bandwidth consumption.
# When you open some window then script unfreeze the window back.
# Apps wmctrl and xwininfo need installed before start this script.
# List all windows by `wmctrl -l -x -p` to get string which you will be use in grep_query (see below).
# Show window info by window id: xwininfo -id <window_id>. The command shows window states: IsViewable or IsUnMapped.
# Modify grep_query variable to get which, for example, windows classes you want to freeze. You can use windows title or part also.
# This script runs infinitely with timeout 1 sec. Press Ctrl-C or kill him.
# You can freeze/unfreeze any process(if you have permissions) manually use `kill -SIGSTOP <pid>` and `kill -SIGCONT <pid>`.

grep_query='chromium.Chromium\|Navigator.Pale moon\|Navigator.firefox'
window_info_lines=`wmctrl -l -x -p | grep "$grep_query"`

echo $window_info_lines

function log {
#	echo "$1"# >> app-freezer.log
	echo "$1"
}

function process_windows {
	while read -r line
	do
		#log "$line"
		local win_id=`echo "$line" | cut -d" " -f1`
		local proc_id=`echo "$line" | cut -d" " -f4`
		
		log "win_id: $win_id, proc_id: $proc_id"

		local state=`xwininfo -id $win_id | grep "Map State" | cut -d" " -f5`
		#log "$state"
		if [ $state = "IsViewable" ]; then
			log "Unfreeze .."
			kill -SIGCONT $proc_id
		else
			log "Freeze .."
			kill -SIGSTOP $proc_id
		fi
		log "Done"
	done <<< "$window_info_lines"
}

while true; do process_windows; sleep 1; done