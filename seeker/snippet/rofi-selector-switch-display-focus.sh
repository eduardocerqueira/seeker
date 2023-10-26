#date: 2023-10-26T17:07:30Z
#url: https://api.github.com/gists/375c17dbd5295cb396645211e22635d8
#owner: https://api.github.com/users/naranyala

#!/bin/bash

# Detect the laptop display (assuming it contains "eDP" in the name)
laptop_display=""
connected_displays=$(xrandr | grep " connected" | awk '{print $1}')
for display in $connected_displays; do
	if [[ $display == *eDP* ]]; then
		laptop_display="$display"
		break
	fi
done

if [ -z "$laptop_display" ]; then
	msg="Laptop display not detected. Exiting."
	echo "$msg" | notify-send "display" "$msg"
	exit 1
fi

# Detect the external display by finding the first connected display other than the laptop display
external_display=""
for display in $connected_displays; do
	if [ "$display" != "$laptop_display" ]; then
		external_display="$display"
		break
	fi
done

if [ -z "$external_display" ]; then
	msg="No external display detected. Exiting."
	echo "$msg" | notify-send "display" "$msg"
	exit 1
fi

# Define the display modes and their respective xrandr commands
modes=("Only Primary Screen" "Second Screen as Duplicate" "Only Second Screen")
xrandr_cmds=("xrandr --output $laptop_display --auto --output $external_display --off"
	"xrandr --output $laptop_display --auto --output $external_display --same-as $laptop_display"
	"xrandr --output $laptop_display --off --output $external_display --auto")

# Create a string with options for Rofi in list format
rofi_options=""
for ((i = 0; i < ${#modes[@]}; i++)); do
	rofi_options+="${modes[i]}\n"
done

# Function to display the Rofi menu and apply the selected mode
select_display_mode() {
	selected_mode=$(echo -e "$rofi_options" | rofi -dmenu -p "Select a display mode" -show list)

	if [ -n "$selected_mode" ]; then
		for ((i = 0; i < ${#modes[@]}; i++)); do
			if [ "$selected_mode" = "${modes[i]}" ]; then
				eval "${xrandr_cmds[i]}"
				msg="Display mode set to: $selected_mode"
				echo "$msg" | notify-send "display" "$msg"
				break
			fi
		done
	else
		msg="No display mode selected. Exiting."
		echo "$msg" | notify-send "display" "$msg"
	fi
}

# Call the function to display the display mode options
select_display_mode
