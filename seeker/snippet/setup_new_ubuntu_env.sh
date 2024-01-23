#date: 2024-01-23T16:58:35Z
#url: https://api.github.com/gists/577c14636db38986a6f433d541c18ece
#owner: https://api.github.com/users/mkol5222

#!/bin/bash

sudo apt update -y 
sudo apt upgrade -y

# jq
echo "Installing jq..."
sudo apt install jq -y

# helm
echo "Installing helm..."
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# kubectl
echo "Installing kubectl..."
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
sudo echo "alias k=kubectl" >> ~/.bash_aliases
sudo echo "alias kpo='k get pods'" >> ~/.bash_aliases
sudo echo "alias kde='k get deploy'" >> ~/.bash_aliases
sudo echo "alias kal='k get all'" >> ~/.bash_aliases
sudo echo "alias kall= "**********"
sudo echo "alias kx='kubectl config use-context'" >> ~/.bash_aliases
sudo echo "alias kn='kubectl config set-context --current --namespace'" >> ~/.bash_aliases
source ~/.bashrc

# terraform
echo "Installing terraform..."
TERRAFORM_VERSION="v1.0.2"
wget https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_linux_amd64.zip
unzip terraform_${TERRAFORM_VERSION}_linux_amd64.zip
sudo mv terraform /usr/local/bin/terraform
rm terraform_${TERRAFORM_VERSION}_linux_amd64.zip
sudo echo "alias tf=terraform" >> ~/.bash_aliases
source ~/.bashrc

# gcloud
echo "Installing gcloud..."
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get install apt-transport-https ca-certificates gnupg
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
sudo apt-get update && sudo apt-get install google-cloud-sdk -y

# docker
echo "Installing docker..."
sudo apt-get install apt-transport-https ca-certificates curl gnupg lsb-release -y
sudo mkdir -m 0755 -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
sudo groupadd docker
sudo usermod -aG docker $USER
# Close ther terminal
newgrp docker

# vs code
sudo apt-get install wget gpg
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg
sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
rm -f packages.microsoft.gpg
sudo apt-get install apt-transport-https
sudo apt-get update
sudo apt-get install code -y

# gh cli
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
&& sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& sudo apt update \
&& sudo apt install gh -y

# oras cli
ORAS_VERSION=$(curl -s https://api.github.com/repos/oras-project/oras/releases/latest | jq -r .tag_name)
curl -LO https://github.com/oras-project/oras/releases/download/${ORAS_VERSION}/oras_${ORAS_VERSION:1}_linux_amd64.tar.gz
mkdir -p oras-install/
tar -zxf oras_${ORAS_VERSION:1}_*.tar.gz -C oras-install/
sudo mv oras-install/oras /usr/local/bin/
rm -rf oras_${ORAS_VERSION:1}_*.tar.gz oras-install/

# kyverno cli
KYVERNO_VERSION=$(curl -s https://api.github.com/repos/kyverno/kyverno/releases/latest | jq -r .tag_name)
curl -LO https://github.com/kyverno/kyverno/releases/download/${KYVERNO_VERSION}/kyverno-cli_${KYVERNO_VERSION}_linux_x86_64.tar.gz
mkdir -p kyverno-install/
tar -xvf kyverno-cli_${KYVERNO_VERSION}_linux_x86_64.tar.gz -C kyverno-install/
sudo mv kyverno-install/kyverno /usr/local/bin/
rm -rf kyverno-cli_${KYVERNO_VERSION}_linux_x86_64.tar.gz kyverno-install/

# cosign
COSIGN_VERSION=$(curl -s https://api.github.com/repos/sigstore/cosign/releases/latest | jq -r .tag_name)
curl -LO "https://github.com/sigstore/cosign/releases/download/${COSIGN_VERSION}/cosign-linux-amd64"
sudo mv cosign-linux-amd64 /usr/local/bin/cosign
sudo chmod +x /usr/local/bin/cosign
local/bin/cosign
sudo chmod +x /usr/local/bin/cosign
