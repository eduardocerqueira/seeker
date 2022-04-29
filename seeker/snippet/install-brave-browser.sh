#date: 2022-04-29T17:01:57Z
#url: https://api.github.com/gists/5ecb1c759d7671f66f9b8cbbe401efbb
#owner: https://api.github.com/users/TylerDurham

#!/usr/bin/bash
sudo apt install apt-transport-https curl

sudo curl -fsSLo /usr/share/keyrings/brave-browser-archive-keyring.gpg https://brave-browser-apt-release.s3.brave.com/brave-browser-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/brave-browser-archive-keyring.gpg arch=amd64] https://brave-browser-apt-release.s3.brave.com/ stable main"|sudo tee /etc/apt/sources.list.d/brave-browser-release.list

sudo apt update && sudo apt install brave-browser