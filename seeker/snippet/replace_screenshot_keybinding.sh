#date: 2022-11-30T17:05:48Z
#url: https://api.github.com/gists/de2016dbcb08ea82481039623967a501
#owner: https://api.github.com/users/iagoalonsomrf

# Release default `window-screenshot` keybinding (Ubuntu 19.10)
gsettings set org.gnome.settings-daemon.plugins.media-keys window-screenshot '[]'

# Replace default `screenshot` binding with the default from `window-screenshot`
gsettings set org.gnome.settings-daemon.plugins.media-keys screenshot '<Alt>Print'

# Create a new custom keybinding (taking for granted that this would be the first created)
gsettings set org.gnome.settings-daemon.plugins.media-keys custom-keybindings "['/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/custom0/']"

# Give it a name
gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/custom0/ name 'Flameshot'

# Set the command it executes
gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/custom0/ command 'flameshot gui --path /home/bledy/Pictures/Screenshots'

# Bind it to PrtScr
gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/custom0/ binding 'Print'
