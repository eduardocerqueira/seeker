#date: 2022-05-27T17:18:14Z
#url: https://api.github.com/gists/09391046643b7d41577bd098f642e510
#owner: https://api.github.com/users/mmckechney

#!/bin/bash

# Receives your Windows username as only parameter.

curl -LO https://storage.googleapis.com/kubernetes-release/release/v1.16.0/bin/linux/amd64/kubectl
chmod +x ./kubectl
sudo mv ./kubectl /usr/local/bin/kubectl

windowsUser=$1

mkdir -p ~/.kube
ln -sf "/mnt/c/users/$windowsUser/.kube/config" ~/.kube/config

kubectl version