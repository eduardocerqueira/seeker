#date: 2021-11-17T16:58:06Z
#url: https://api.github.com/gists/6a9cd10f512fc8411e095658e4d5bf93
#owner: https://api.github.com/users/byronjones-elsevier

#!/bin/bash

echo Installing Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

echo Installing Visual Studio Code
brew install --cask visual-studio-code

echo Confirming Python2 version
python --version

echo Confirming Python 3 version
python3 --version

echo Installing Python PYENV and PYENV-VIRTUALENV
brew install pyenv pyenv-virtualenv

echo Installing the NANO cli text editor
brew install nano

echo Creating the ZSH Profile for PYENV
touch ~/.zprofile

echo "export PYENV_ROOT=$HOME/.pyenv/" > ~/.zprofile
echo "export PATH=$PYENV_ROOT/bin:$PATH" > ~/.zprofile
echo "eval ""$(pyenv init -)""" > ~/.zprofile
echo "eval ""$(pyenv virtualenv-init -)""" > ~/.zprofile

cp .zprofile .zshrc

echo Check PYENV Available Versions
pyenv install --list

echo Install Docker Desktop
brew  install docker
docker --version

echo Install Ansible
brew  install ansible

echo Install TFENV
brew unlink terraform
brew  install tfenv
tfenv --version
tfenv install 1.0.11
tfenv list
tfenv use 1.0.11

echo Install SAML2AWS
brew install saml2aws

echo Install AWS CLI
brew install awscli

echo Install WGet
brew install wget

echo Install Powershell
brew install powershell

echo Install K9S
brew install K9S

echo Install KubeCTL
brew install kubectl

echo Install Helm
brew install Helm

echo Install miniKube
brew install minikube

echo Install EKSCTL
brew install EKSCTL

echo Install JQ
brew install jq

echo Install PIPENV
brew install pipenv

echo Install OpenShift-CLI v3.6.0
#Download OpenShift-CLI v3.6.0 from:  https://github.com/openshift/origin/releases/download/v3.6.0/openshift-origin-client-tools-v3.6.0-c4dd4cf-mac.zip
cd ~/Downloads
wget https://github.com/openshift/origin/releases/download/v3.6.0/openshift-origin-client-tools-v3.6.0-c4dd4cf-mac.zip -P ~/Downloads
unzip ~/Downloads/openshift-origin-client-tools-v3.6.0-c4dd4cf-mac.zip -d ~/Downloads/openshift-origin-client-tools-v3.6.0-c4dd4cf-mac/
ls ~/Downloads/openshift-origin-client-tools-v3.6.0-c4dd4cf-mac
cp ~/Downloads/openshift-origin-client-tools-v3.6.0-c4dd4cf-mac/oc /usr/local/bin
echo When you try to run OC, you will get a permissions issue.
echo Go to: Mac > System Preferences > Security & Privacy > General > "Allow apps downloaded from"
echo to allow the app.
oc version


echo Creating SSH Key
read -p "Enter your email address: " USEREMAIL
ssh-keygen -t ed25519 -C "$USEREMAIL"
echo Starting SSH Agent
eval "$(ssh-agent -s)"
echo Creating SSH Config File
touch ~/.ssh/config
echo "Host *" > ~/.ssh/config
echo "  AddKeysToAgent yes" > ~/.ssh/config
echo "  UseKeychain yes" > ~/.ssh/config
echo "  IdentityFile ~/.ssh/id_ed25519" > ~/.ssh/config
echo Adding SSH Key to Mac Keychain
ssh-add -K ~/.ssh/id_ed25519
echo Copying SSH Public Key to clipboard, so you can paste it into:
echo https://github.com/settings/ssh/new
pbcopy < ~/.ssh/id_ed25519.pub