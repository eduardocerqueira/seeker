#date: 2024-08-27T16:54:10Z
#url: https://api.github.com/gists/8fcc46f5c40c11636b086b1420c71802
#owner: https://api.github.com/users/JanGalek

#!/usr/bin/env sh

sudo apt-get install ripgrep

# installs nvm (Node Version Manager)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash

wget https://github.com/neovim/neovim/releases/download/v0.10.1/nvim-linux64.tar.gz
tar tar xzvf nvim-linux64.tar.gz
cd nvim-linux64
sudo cp -R * /usr

git clone https://github.com/NvChad/starter ~/.config/nvim && nvim