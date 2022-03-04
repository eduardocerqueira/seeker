#date: 2022-03-04T16:56:06Z
#url: https://api.github.com/gists/80c61832ef2260b6915af4f5f760549c
#owner: https://api.github.com/users/prom3theu5

# Configuring Docker to use TLS **WITH** systemd socket
# https://docs.docker.com/engine/reference/commandline/dockerd//#daemon-configuration-file

echo '{
  "tls": true,
  "tlscacert": "/etc/docker/ca.pem",
  "tlscert": "/etc/docker/server-cert.pem",
  "tlskey": "/etc/docker/server-key.pem",
  "tlsverify": true
}' | sudo tee /etc/docker/daemon.json

# Configuring systemd socket to listen on TCP
# https://github.com/docker/docker/issues/25471#issuecomment-238076313
sudo mkdir -p /etc/systemd/system/docker.socket.d
echo '[Socket]
ListenStream= # If you want to disable default unix socket
ListenStream=0.0.0.0:2376' | sudo tee /etc/systemd/system/docker.socket.d/tcp_secure.conf
sudo systemctl daemon-reload
sudo service docker restart
