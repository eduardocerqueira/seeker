#date: 2023-03-08T16:53:18Z
#url: https://api.github.com/gists/0513aa2e3cdf588f3d2ad7ef86c1272a
#owner: https://api.github.com/users/gregscharf

#!/bin/bash

# The following is only necessary if, for example, you mistype your password when it attempts to change your shell
# sudo chsh -s /usr/bin/zsh

# from https://gist.github.com/dogrocker/1efb8fd9427779c827058f873b94df95
git clone https://github.com/zsh-users/zsh-autosuggestions.git /home/$USER/.oh-my-zsh/custom/plugins/zsh-autosuggestions

git clone https://github.com/zsh-users/zsh-syntax-highlighting.git /home/$USER/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting

# add (zsh-autosuggestions zsh-syntax-highlighting tmux git-prompt) to .zshrc
sed -i -e 's/plugins=(git)/plugins=(zsh-autosuggestions zsh-syntax-highlighting tmux git-prompt)/g' ~/.zshrc
# tmux and git-prompt are pre-installed with oh my zsh.

# change the OMZ theme 
sed -i -e 's/ZSH_THEME="robbyrussell"/ZSH_THEME="mikeh"/g' ~/.zshrc

echo "alias la='ls -lah'" >> ~/.zshrc

# vim setup
git clone --depth=1 https://github.com/amix/vimrc.git ~/.vim_runtime
sh ~/.vim_runtime/install_awesome_vimrc.sh