#date: 2024-04-08T16:53:30Z
#url: https://api.github.com/gists/2a06ec1f3114db026ad76169ffa2caf5
#owner: https://api.github.com/users/mandalvesq

#!/bin/bash

## Install Kubectl

curl -O https://s3.us-west-2.amazonaws.com/amazon-eks/1.29.0/2024-01-04/bin/linux/amd64/kubectl

sudo chmod +x ./kubectl

mkdir -p $HOME/bin && cp ./kubectl $HOME/bin/kubectl && export PATH=$HOME/bin:$PATH

echo 'export PATH=$HOME/bin:$PATH' >> ~/.bashrc

## Install Terraform 


wget https://releases.hashicorp.com/terraform/1.6.3/terraform_1.6.3_linux_amd64.zip

unzip terraform_1.6.3_linux_amd64.zip

sudo mv terraform /usr/local/bin

## Install Helm 

curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3

chmod 700 get_helm.sh

./get_helm.sh


## Alias for Karpenter Logs

alias kl="kubectl -n karpenter logs -l app.kubernetes.io/name=karpenter --all-containers=true -f --tail=20"

