#date: 2023-01-17T17:01:06Z
#url: https://api.github.com/gists/b5e708b62b508adc3923cd589dc4a4f6
#owner: https://api.github.com/users/skecskes

#!/bin/bash

cd ~

echo "Delete google-cloud-cli google-cloud-cli-gke-gcloud-auth-plugin google-cloud-sdk-gke-gcloud-auth-plugin"
sudo apt-get remove google-cloud-cli google-cloud-cli-gke-gcloud-auth-plugin google-cloud-sdk-gke-gcloud-auth-plugin -y && \
sudo apt-get purge google-cloud-cli google-cloud-cli-gke-gcloud-auth-plugin google-cloud-sdk-gke-gcloud-auth-plugin && \
sudo apt-get autoremove

echo "Clean kubectl"
if [ -d ~/.kube ]; then
    rm -rf ~/.kube
fi

kubectl_path="$(whereis kubectl | cut -d':' -f2 | cut -d' ' -f2)"
if [ -n $(echo $kubectl_path) ]; then 
    sudo rm -f $kubectl_path
fi

echo "Add gcloud repos if they don't exist"

if [ ! -f /etc/apt/sources.list.d/google-cloud-sdk.list ]; then
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
fi

echo "Install them, if package fail, solve it and reinstall"
sudo apt-get update && sudo apt-get install google-cloud-cli -y
if [ $? -ne 0 ]; then 
    sudo dpkg -i --force-overwrite /var/cache/apt/archives/google-cloud-cli_412.0.0-0_all.deb
    sudo apt -f install
    sudo apt-get update && sudo apt-get install google-cloud-cli -y
fi
sudo apt-get install google-cloud-sdk-gke-gcloud-auth-plugin -y

echo "Reinstall kubectl"
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
if [ -z $kubectl_path ]; then
    kubectl_path=/usr/bin/kubectl
fi
sudo mv kubectl $kubectl_path
sudo chmod +x $kubectl_path
sudo chown $(whoami) $kubectl_path

echo "Add USE_GKE_GCLOUD_AUTH_PLUGIN variable in profile file if not exists"
if [ -z "$(cat ~/.profile | grep USE_GKE_GCLOUD_AUTH_PLUGIN)" ]; then 
    echo 'export USE_GKE_GCLOUD_AUTH_PLUGIN=True' >> ~/.profile
    source ~/.profile
fi


echo "Login gcloud"
myuser=$(whoami)
sudo chown -R :${myuser} ~/.config/gcloud/configurations
gcloud auth login

if [ $? -eq 0 ]; then 
    echo "It works!"
    exit 0
else
    echo "It failed... let's troubleshoot it!"
    exit 1
fi