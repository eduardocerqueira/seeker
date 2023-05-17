#date: 2023-05-17T16:54:00Z
#url: https://api.github.com/gists/c9bb50d8d1ec84ea8ca6b74ca00edddb
#owner: https://api.github.com/users/jango-fx

sudo apt update
sudo apt upgrade
curl -sSL https://get.docker.com | sh
sudo usermod -aG docker pi
sudo service docker start
sudo docker pull portainer/portainer-ce:latest
sudo docker run -d -p 9443:9443 --name=portainer --restart=always -v /var/run/docker.sock:/var/run/docker.sock -v portainer_data:/data portainer/portainer-ce:latest