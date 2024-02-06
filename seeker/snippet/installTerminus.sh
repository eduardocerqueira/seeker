#date: 2024-02-06T16:42:04Z
#url: https://api.github.com/gists/e478ec7c1c92ab30e5c45ca805425d84
#owner: https://api.github.com/users/mjones129

#!/bin/bash


#update all the things
sudo apt update &&

sudo apt upgrade -y &&


#install all the dependencies
sudo apt install -y zsh php8.1 php-xml git composer &&

chsh -s $(which zsh) &&

#answer the prompt to generate a default shell profile
2 &&

#install wp-cli
curl -O https://raw.githubusercontent.com/wp-cli/builds/gh-pages/phar/wp-cli.phar &&

#check if it works
php wp-cli.phar --info &&

#install binary
chmod +x wp-cli.phar &&
sudo mv wp-cli.phar /usr/local/bin/wp &&

#install the actual Terminus
mkdir -p ~/terminus && 
cd ~/terminus &&
curl -L https://github.com/pantheon-systems/terminus/releases/download/3.3.0/terminus.phar --output terminus &&
chmod +x terminus &&
./terminus self:udpate &&
sudo ln -s ~/terminus/terminus /usr/local/bin/terminus &&

echo "Great job! Now go generate your machine token: "**********"://docs.pantheon.io/terminus/install#machine-token"