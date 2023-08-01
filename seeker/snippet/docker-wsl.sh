#date: 2023-08-01T16:44:30Z
#url: https://api.github.com/gists/c5700b3f1b7abe783fd20423a7de5f10
#owner: https://api.github.com/users/Trass3r

# https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository
sudo apt update
sudo apt install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "# allow starting the docker daemon without password\n%docker ALL=(ALL)  NOPASSWD: "**********"

# put into ~/.bashrc:
if [ ! -S /var/run/docker.sock ]; then
        nohup sudo -b dockerd < /dev/null > ~/dockerd.log 2>&1
fi
/dockerd.log 2>&1
fi
