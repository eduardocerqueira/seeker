#date: 2023-06-19T17:02:28Z
#url: https://api.github.com/gists/2d8fde1326f85f7866af22da66560f90
#owner: https://api.github.com/users/geblanco

#/bin/bash

# Install:
# - zsh (requires sudo)
# - oh-my-zsh
# - zsh-autosuggestions plugin
# - conda-zsh-completion plugin
# - custom themes

sudo apt --yes --force-yes install zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
git clone https://github.com/esc/conda-zsh-completion ${ZSH_CUSTOM:=~/.oh-my-zsh/custom}/plugins/conda-zsh-completion

git clone https://github.com/geblanco/mod-zsh-themes /tmp/mod-zsh-themes
cd /tmp/mod-zsh-themes
make install PREFIX=${ZSH_CUSTOM:=~/.oh-my-zsh}
cd -
rm -rf /tmp/mod-zsh-themes
