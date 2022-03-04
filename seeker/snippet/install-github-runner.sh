#date: 2022-03-04T16:44:01Z
#url: https://api.github.com/gists/86eb0a11af05b71bc6cac967147e436b
#owner: https://api.github.com/users/jniltinho

#!/bin/bash
## Install GitHub Actions Runner
## https://docs.github.com/en/actions/hosting-your-own-runners/configuring-the-self-hosted-runner-application-as-a-service

## Install as root
## Install Docker
apt-get update
apt-get upgrade
apt-get install -y apt-transport-https ca-certificates curl software-properties-common docker.io socat
systemctl start docker
systemctl enable docker


useradd --comment 'GitHub Runner' --create-home github-runner --shell /bin/bash
gpasswd -a github-runner docker

cd /home/github-runner

su - github-runner -c "curl -o actions-runner-linux-x64-2.287.1.tar.gz -L https://github.com/actions/runner/releases/download/v2.287.1/actions-runner-linux-x64-2.287.1.tar.gz"
su - github-runner -c "tar xvzf ./actions-runner-linux-x64-2.287.1.tar.gz"
su - github-runner -c "./config.sh --url https://github.com/name_repo/projectA --token TOKEN"

./svc.sh install
./svc.sh status
./svc.sh start
./svc.sh status

