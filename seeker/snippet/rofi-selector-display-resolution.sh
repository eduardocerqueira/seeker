#date: 2023-10-26T17:06:44Z
#url: https://api.github.com/gists/8b0c16c9e42918fc8c0fbcf5c4177c5d
#owner: https://api.github.com/users/naranyala

#!/bin/bash

# Get a list of available resolutions using xrandr and sort them in reverse order
resolutions=$(xrandr | awk '/[0-9]x[0-9]/ {print $1}' | sort -n -r)

# Function to display resolution options using Rofi
select_resolution() {
	selected_resolution=$(echo -e "$resolutions" | rofi -dmenu -p "Select a resolution:")

	if [ -n "$selected_resolution" ]; then
		# Change the screen resolution to the selected option
		xrandr --output $(xrandr | grep -oP '.*(?=\sconnected)') --mode $selected_resolution
		msg="Resolution set to $selected_resolution"
		echo "$msg" | notify-send "display" "$msg"
	else
		msg="No resolution selected. Exiting."
		echo "$msg" | notify-send "display" "$msg"
	fi
}

# Call the function to display the resolution options
select_resolution
