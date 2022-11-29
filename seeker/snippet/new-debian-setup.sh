#date: 2022-11-29T17:01:52Z
#url: https://api.github.com/gists/9145473f5cebbe65e0e4059d97ee29d1
#owner: https://api.github.com/users/garlic-os

#! /bin/bash
# apt packages
sudo apt update
sudo apt dist-upgrade -y
sudo apt install build-essential python3-pip -y
sudo apt autoremove -y

# path
echo "export PATH=\$PATH:\$HOME/.local/bin" >> ~/.bashrc

# aliases
echo alias "untar='tar -xvf'" >> ~/.bash_aliases
echo alias "ls='ls -lAGh1vX --group-directories-first --color=auto'" >> ~/.bash_aliases

# pyenv
curl https://pyenv.run | bash
echo "export PYENV_ROOT='\$HOME/.pyenv'" >> ~/.bashrc
echo "command -v pyenv \>/dev/null || export PATH='\$PYENV_ROOT/bin:\$PATH'" >> ~/.bashrc
echo "eval '\$(pyenv init -)'" >> ~/.bashrc
echo "eval '\$(pyenv virtualenv-init -)'" >> ~/.bashrc

source ~/.bashrc

# python
python -m pip install --upgrade pip

# node.js
wget -qO- https://github.com/nvm-sh/nvm/raw/master/install.sh | bash
nvm install --lts
nvm use --lts
npm install -g npm@latest
npm install --global pnpm