#date: 2024-10-04T16:45:05Z
#url: https://api.github.com/gists/3e44304bd8ff29a7f956a49edb4585e2
#owner: https://api.github.com/users/asalbright

#!/bin/bash

# For installing fonts from the Nerd Fonts website: https://www.nerdfonts.com/font-downloads
# Right-click on the "Download" button and copy the address, run this script with that address

# To set the system font for things like the terminal do the following:
# > gsettings set org.gnome.desktop.interface monospace-font-name 'UbuntuMono Nerd Font 13'

FONT_ADDRESS=$1

if [ -z "$FONT_ADDRESS" ]; then
  FONT_ADDRESS=https://github.com/ryanoasis/nerd-fonts/releases/download/v3.2.1/UbuntuMono.zip
fi

NAME_OF_ZIP=$(basename $FONT_ADDRESS)

# FONT SETTINGS
# Install the fonts you might need in the container
wget -P ~/.local/share/fonts $FONT_ADDRESS \
&& cd ~/.local/share/fonts \
&& unzip $NAME_OF_ZIP \
&& rm $NAME_OF_ZIP \
&& fc-cache -fv
