#date: 2024-02-15T17:09:01Z
#url: https://api.github.com/gists/a90200b3517b8569cb0ce4b993b4f308
#owner: https://api.github.com/users/AntonClaesson

# Install deps
apt update && apt-get update
apt-get -y install git tmux vim
apt -y install zsh

# Select zsh as default shell
chsh -s $(which zsh)

# Install oh-my-zsh
sh -c "$(wget -O - https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Setup dotfiles
rm ~/.bashrc ~/.zshrc
bash <(wget -O - https://gist.githubusercontent.com/AntonClaesson/43fdfb0893ca50ef20fae20b685bd7d2/raw/)

# Setup tmux
git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm
tmux new-session -d
tmux source ~/.config/tmux/tmux.conf
tmux kill-session

# Install powerlevel10k
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ~/powerlevel10k

# Run to reload shell
exec zsh