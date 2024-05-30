#date: 2024-05-30T16:47:42Z
#url: https://api.github.com/gists/721ca3fe2c0bf8d7e53d4919555287b1
#owner: https://api.github.com/users/zacharyneveu

#!/usr/bin/env bash

# Oh my ZSH
sudo apt install -y zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# Ranger file manager
sudo apt install -y ranger
echo 'alias r=". ranger"' >> ~/.zshrc

# Neovim Editor
sudo apt install -y neovim
echo "export EDITOR=nvim" >> ~/.zshrc