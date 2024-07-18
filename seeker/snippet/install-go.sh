#date: 2024-07-18T16:43:43Z
#url: https://api.github.com/gists/c5a1b9ee1bf444837984cb1c7053710d
#owner: https://api.github.com/users/vicradon

#!/bin/bash

cd ~

wget https://go.dev/dl/go1.22.5.linux-amd64.tar.gz

sudo rm -rf /usr/local/go && sudo tar -C /usr/local -xzf go1.22.5.linux-amd64.tar.gz

echo "export PATH=$PATH:/usr/local/go/bin" >> ~/.bashrc

source ~/.bashrc