#date: 2023-04-27T16:52:48Z
#url: https://api.github.com/gists/965a3f26dd1295459e46b22a54b8f2f1
#owner: https://api.github.com/users/caocuong2404

# make dock faster
defaults write com.apple.dock autohide-delay -float 0; killall Dock

# install brew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

(echo; echo 'eval "$(/usr/local/bin/brew shellenv)"') >> ~/.zshrc

zsh

brew update && brew upgrade

# install some packages

echo "
starship
thefuck
nvm
zsh-autosuggestions
" >> packages.txt

brew install $(<packages.txt)

echo 'eval "$(starship init zsh)"' >> ~/.zshrc

mkdir ~/.nvm

echo '
# thefuck
eval $(thefuck --alias)

# starship
eval "$(starship init zsh)"

# zsh-autosuggestions
source /usr/local/share/zsh-autosuggestions/zsh-autosuggestions.zsh

# nvm
export NVM_DIR="$HOME/.nvm"
[ -s "/usr/local/opt/nvm/nvm.sh" ] && \. "/usr/local/opt/nvm/nvm.sh"  # This loads nvm
[ -s "/usr/local/opt/nvm/etc/bash_completion.d/nvm" ] && \. "/usr/local/opt/nvm/etc/bash_completion.d/nvm"  # This loads nvm bash_completion
' >> ~/.zshrc

zsh
