#date: 2023-02-03T16:58:46Z
#url: https://api.github.com/gists/03aafbebe42bdf194252b6a0e079d4e4
#owner: https://api.github.com/users/inappropriatecontent

#! /bin/bash
wget -qO - https://gitlab.com/paulcarroty/vscodium-deb-rpm-repo/raw/master/pub.gpg | gpg --dearmor | sudo dd of=/usr/share/keyrings/vscodium-archive-keyring.gpg &&
echo 'deb [ signed-by=/usr/share/keyrings/vscodium-archive-keyring.gpg ] https://paulcarroty.gitlab.io/vscodium-deb-rpm-repo/debs vscodium main' | sudo tee /etc/apt/sources.list.d/vscodium.list &&
sudo apt update && sudo apt install codium -y &&