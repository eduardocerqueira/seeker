#date: 2022-03-04T16:56:06Z
#url: https://api.github.com/gists/80c61832ef2260b6915af4f5f760549c
#owner: https://api.github.com/users/prom3theu5

# On another node than the registry

export REGISTRY_PRIVATE_IP=10.X.Y.Z

# Configure unsecure registries
echo "{
  \"insecure-registries\": [
    \"${REGISTRY_PRIVATE_IP}:5000\"
  ]
}" | sudo tee /etc/docker/daemon.json

# Reload docker config
sudo service docker reload

# Pull from insecure registry
docker pull ${REGISTRY_PRIVATE_IP}:5000/johnnytu/busybox:1.0
