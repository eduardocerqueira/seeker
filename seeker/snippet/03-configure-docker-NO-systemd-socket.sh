#date: 2022-03-04T16:56:06Z
#url: https://api.github.com/gists/80c61832ef2260b6915af4f5f760549c
#owner: https://api.github.com/users/prom3theu5

# Configuring Docker to use TLS **WITHOUT** systemd socket
# https://docs.docker.com/engine/reference/commandline/dockerd//#daemon-configuration-file

echo '{
  "hosts": [
    "unix:///var/run/docker.sock",
    "tcp://0.0.0.0:2376"
  ],
  "tls": true,
  "tlscacert": "/etc/docker/ca.pem",
  "tlscert": "/etc/docker/server-cert.pem",
  "tlskey": "/etc/docker/server-key.pem",
  "tlsverify": true
}' | sudo tee /etc/docker/daemon.json

# Disable systemd docker host configuration
sudo mkdir -p /etc/systemd/system/docker.service.d
echo '[Service]
ExecStart=
ExecStart=/usr/bin/dockerd' | sudo tee /etc/systemd/system/docker.service.d/simple_dockerd.conf
sudo systemctl daemon-reload
sudo service docker restart
