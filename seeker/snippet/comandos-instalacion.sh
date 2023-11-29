#date: 2023-11-29T16:42:05Z
#url: https://api.github.com/gists/779ce2b53f44b70c891040e22f9f7b0f
#owner: https://api.github.com/users/repositorioinformatico

sudo apt update

sudo apt-get install apt-transport-https ca-certificates curl gnupg-agent software-properties-common

sudo mkdir -m 0755 -p /etc/apt/keyrings

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
"deb [arch=$(dpkg --print-architecture)signed-by=/etc/apt/keyrings/docker.gpg]
https://download.docker.com/linux/ubuntu \
$(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null