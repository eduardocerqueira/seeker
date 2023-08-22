#date: 2023-08-22T16:53:32Z
#url: https://api.github.com/gists/dc29ce1b6e980f94ff72e595f05e29b7
#owner: https://api.github.com/users/akshay-na

#!/bin/bash

# Define the URLs
STARSHIP_API_URL="https://api.github.com/repos/starship/starship/releases/latest"
CONFIG_URL="https://gist.githubusercontent.com/akshay-na/2c37a8d21d8abd249543851178f37c0d/raw/starship.toml"
FONT_URL="https://github.com/microsoft/cascadia-code/releases/download/v2106.17/CascadiaCode-2106.17.zip"

# Try installing Starship using the provided curl command
if yes | curl -sS https://starship.rs/install.sh | sh; then
    echo "Starship installed successfully using the curl command."
else
    # Download the latest Starship release
    download_url=$(curl -s $STARSHIP_API_URL | grep "browser_download_url.*starship-x86_64-unknown-linux-gnu.tar.gz" | cut -d '"' -f 4)
    curl -L -o "$HOME/starship.tar.gz" $download_url

    # Unzip Starship and move to the User directory
    tar -xzf "$HOME/starship.tar.gz" -C $HOME
    rm "$HOME/starship.tar.gz"
fi

# Add commands to .bashrc
echo 'eval "$(starship init bash)"' >> "$HOME/.bashrc"
echo 'eval "$(starship init zsh)"' >> "$HOME/.zshrc"

# Download and install the Cascadia Code Nerd Font
curl -L -o "$HOME/CascadiaCode.zip" $FONT_URL
unzip "$HOME/CascadiaCode.zip" -d "$HOME/CascadiaCode"

# Assuming the font is named "CaskaydiaCoveNerdFontMono*" in the zip; adjust if different
mkdir -p ~/.local/share/fonts
cp "$HOME/CascadiaCode/ttf/CaskaydiaCoveNerdFontMono*" ~/.local/share/fonts/
fc-cache -f -v

# Download the Starship config file from the provided gist
mkdir -p "$HOME/.config"
curl -L -o "$HOME/.config/starship.toml" $CONFIG_URL

rm -r "$HOME/CascadiaCode"
rm "$HOME/CascadiaCode.zip"

echo "All tasks completed!"
echo "Please restart your terminal to see the changes."
