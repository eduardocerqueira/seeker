#date: 2023-02-03T16:47:35Z
#url: https://api.github.com/gists/05270547640b817ccfef1ae5eb9dd14a
#owner: https://api.github.com/users/winggundamth

#!/bin/sh

# Generate SSH
[[ ! -f ~/.ssh/id_rsa ]] && ssh-keygen -f ~/.ssh/id_rsa -N ""

# Install Docker Compose v2
mkdir -p ~/.docker/cli-plugins/
curl -SL https://github.com/docker/compose/releases/download/v2.15.1/docker-compose-linux-x86_64 -o ~/.docker/cli-plugins/docker-compose
chmod +x ~/.docker/cli-plugins/docker-compose
docker compose version

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3 | bash
helm version

# Put Bash Completion into .bashrc file
if ! grep -q 'kubectl completion bash' ~/.bashrc
then
  tee -a ~/.bashrc > /dev/null <<EOT

export USE_GKE_GCLOUD_AUTH_PLUGIN=True

if ! grep -q 'kubectl completion bash' ~/.bashrc &> /dev/null
then
  # Bash Completion
  . <(kubectl completion bash)
  . <(helm completion bash)
fi
EOT
fi

git config --global user.name "$USER"
git config --global user.email "$USER@opsta.net"
git config --global init.defaultBranch "main"
git config --global pull.rebase false

if [ ! -d "$HOME/ratings" ]; then
  ssh-keygen -F github.com || ssh-keyscan github.com >>~/.ssh/known_hosts
  git clone https://github.com/winggundamth/kad23-bookinfo-ratings.git $HOME/ratings
fi
