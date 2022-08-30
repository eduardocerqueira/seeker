#date: 2022-08-30T16:56:45Z
#url: https://api.github.com/gists/b09b03fb87dd505ee1be39d5caba1013
#owner: https://api.github.com/users/k86td

#!/bin/bash

install_neovim() {

        base_url="https://api.github.com/repos/neovim/neovim/releases/tags"
        version="0.7.2"
        url="${base_url}/v${version}"

        echo "url is $url"

        download_url="$(curl -s ${url} | grep "browser_download_url.*deb" | cut -d : -f 2,3 | tr -d \" | head -1)"
        echo "download neovim release (.deb) from url ${download_url}"

        wget -P /tmp/ $download_url

        filename="$(ls /tmp/ | grep nvim.*deb)"

        echo "installing $filename"

        sudo apt install "/tmp/$filename"

        # cleanup
        sudo rm "/tmp/$filename"
}

add_line_if_not_exist () {
        file=$1
        line=$2

        grep -qxF "$line" "$file" || echo "$line" >> "$file"
}

execute_nvim_command () {
        command=$1
        /usr/bin/nvim -s <(echo ":$command")
}

# ensure we're not running as root
if [ "$EUID" -eq 0 ]; then
        echo "This script cannot be running as root since it will configure the user root"
        exit
fi

# install base requirements
sudo apt-get install -y wget curl stow git

# install neovim from git release page
install_neovim

# install python requirements
sudo apt-get install -y python3 python3-pynvim pip

# set alias vi to nvim
add_line_if_not_exist ~/.bashrc 'alias vi="nvim"'

# clone dotfiles configuration url
git_repo="https://github.com/k86td/dotfiles"
git clone $git_repo

# create symlinks from repo to home
cd dotfiles
stow --ignore='\.git' .

# install nodejs
curl -fsSL https://deb.nodesource.com/setup_16.x | sudo bash -
sudo apt-get install -y nodejs

# install plugins
execute_nvim_command 'PlugInstall'

# install Coc extensions
execute_nvim_command 'CocInstall coc-pairs'

# nodejs
execute_nvim_command 'CocInstall coc-json'
execute_nvim_command 'CocInstall coc-tsserver'

# install nodejs debugger
nodejs_dap_git_url="https://github.com/microsoft/vscode-node-debug2.git"
sudo npm install -g gulp

mkdir ~/dap/

cd ~/dap/
git clone $nodejs_dap_git_url dap-nodejs
cd dap-nodejs/
npm install && npm run build
cd ~