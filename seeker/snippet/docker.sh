#date: 2022-11-09T17:13:52Z
#url: https://api.github.com/gists/1d422cfa4ff054e7eed758603e8ffa1e
#owner: https://api.github.com/users/chrisedrego

sudo apt update
sudo apt -y install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable"
sudo apt update
sudo apt -y install docker-ce
sudo apt install docker.io
sudo usermod -aG docker ${USER}
sudo systemctl start docker.service
sudo systemctl enable docker.service
sudo systemctl status docker.service