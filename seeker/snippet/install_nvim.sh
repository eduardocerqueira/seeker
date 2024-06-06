#date: 2024-06-06T16:40:48Z
#url: https://api.github.com/gists/5b6504dbefdd49861bc2beb96eb29deb
#owner: https://api.github.com/users/asadmoosvi

#!/usr/bin/env bash

OUTPUT_DIR="$HOME/.local/bin"
mkdir -p $OUTPUT_DIR
NVIM="$OUTPUT_DIR/nvim"

sudo apt install -y libfuse2 wget
wget 'https://github.com/neovim/neovim/releases/download/nightly/nvim.appimage' -O $NVIM
chmod u+x $NVIM

echo -e "-> Neovim updated.\n"
$NVIM --version

git clone "git@github.com:asadmoosvi/nvim.git" $HOME/.config/nvim
source $HOME/.profile