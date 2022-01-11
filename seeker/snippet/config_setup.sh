#date: 2022-01-11T17:09:06Z
#url: https://api.github.com/gists/293114f56f87fae3911b71fda7278ad0
#owner: https://api.github.com/users/kevinallenbriggs

#!/bin/bash

echo "Installing git, wget and zsh..."
if command -v zsh &> /dev/null && command -v git &> /dev/null && command -v wget &> /dev/null; then
    echo -e "zsh and git are already installed."
else
    if sudo apt install -y zsh git wget || sudo pacman -S zsh git wget || sudo dnf install -y zsh git wget || sudo yum install -y zsh git wget || sudo brew install git zsh wget || pkg install git zsh wget ; then
        echo -e "zsh wget and git installed."
    else
        echo -e "Please install the following packages first, then try again: zsh git wget." && exit
    fi
fi

echo "Backing up existing .zshrc..."
if mv -n ~/.zshrc ~/.zshrc-backup-$(date +"%Y-%m-%d"); then # backup .zshrc
    echo -e "Backed up the current .zshrc to .zshrc-backup-<DATE>."
else 
   echo -e "No .zshrc found."
fi

echo -e "Installing oh-my-zsh..."
if [ -d ~/.config/ezsh/oh-my-zsh ]; then
    echo -e "oh-my-zsh is already installed."
else
    curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh | /bin/zsh
fi

echo "Setting up configuration version control..."
git clone --bare https://github.com/kevinallenbriggs/configs.git $HOME/.cfg
function config {
   /usr/bin/git --git-dir=$HOME/.cfg/ --work-tree=$HOME $@
}
mkdir -p .config-backup
config checkout
if [ $? = 0 ]; then
  echo "Checked out configurations.";
  else
    echo "Backing up pre-existing dot files.";
    config checkout 2>&1 | egrep "\s+\." | awk {'print $1'} | xargs -I{} mv {} .config-backup/{}
fi;
config checkout
config config status.showUntrackedFiles no
source $HOME/.zshrc

echo "Installing Vundle (VIM plugin manager)..."
rm -rf ~/.vim/bundle/Vundle.vim
git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim

echo "Complete!"