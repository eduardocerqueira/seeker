#date: 2021-12-15T16:56:16Z
#url: https://api.github.com/gists/85c57e34ecf95cbcfedfe02561925d4d
#owner: https://api.github.com/users/sistlm

# https://docs.docker.com/engine/install/ubuntu/

apt-get update -y

apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -

apt-key fingerprint 0EBFCD88

add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

apt-get update -y

apt-get install docker-ce docker-ce-cli containerd.io