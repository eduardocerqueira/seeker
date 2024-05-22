#date: 2024-05-22T17:09:47Z
#url: https://api.github.com/gists/5fa5bdb25085f87ec96facd8d6a90658
#owner: https://api.github.com/users/kogutich

#!/bin/sh

mkdir -pv ~/.config/alacritty/themes
mkdir -pv "$(bat --config-dir)/themes"
mkdir -pv ~/.config/btop/themes

# alacritty
wget -O ~/.config/alacritty/themes/catppuccin-macchiato.toml \
    https://github.com/catppuccin/alacritty/raw/main/catppuccin-macchiato.toml &&
    echo "Successfully downloaded alacritty theme" ||
    echo "Failed to download alacritty theme"

# bat
wget -O "$(bat --config-dir)/themes/Catppuccin Macchiato.tmTheme" \
    https://github.com/catppuccin/bat/raw/main/themes/Catppuccin%20Macchiato.tmTheme &&
    echo "Successfully downloaded bat theme" ||
    echo "Failed to download bat theme"

bat cache --build || echo "Failed to build bat cache"

# btop
wget -O ~/.config/btop/themes/catppuccin_macchiato.theme \
    https://raw.githubusercontent.com/catppuccin/btop/main/themes/catppuccin_macchiato.theme &&
    echo "Successfully downloaded btop theme" ||
    echo "Failed to download btop theme"
