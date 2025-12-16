#date: 2025-12-16T17:03:22Z
#url: https://api.github.com/gists/863eb1d5a35dfc49e2c7f68b42bd286a
#owner: https://api.github.com/users/ybhambha

#
# install.sh
#
apt update
apt upgrade -y

apt install vim -y
apt install screen -y
apt install htop -y
apt install python3 -y
apt install pipx -y

export PATH=$PATH:/root/.local/bin

pipx install ipython
pipx inject ipython numpy
pipx inject ipython pandas
pipx inject ipython jupyterlib
pipx inject ipython matplotlib

screen
