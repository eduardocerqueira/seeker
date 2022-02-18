#date: 2022-02-18T16:45:58Z
#url: https://api.github.com/gists/373f37817742c53891a076391533fe6d
#owner: https://api.github.com/users/incogbyte

#!/bin/bash

sudo apt install fontconfig
cd ~
wget https://github.com/ryanoasis/nerd-fonts/releases/download/v2.1.0/Meslo.zip
mkdir -p .local/share/fonts
unzip Meslo.zip -d .local/share/fonts
cd .local/share/fonts
rm *Windows*
cd ~
rm Meslo.zip
fc-cache -fv