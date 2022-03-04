#date: 2022-03-04T16:56:06Z
#url: https://api.github.com/gists/80c61832ef2260b6915af4f5f760549c
#owner: https://api.github.com/users/prom3theu5

docker \
  --host tcp://localhost:3276 \
  --tlsverify \
  --tlscacert=~/docker-ca/ca.pem \
  --tlscert=~/docker-ca/client-cert.pem \
  --tlskey=~/docker-ca/client-key.pem \
  container ls

# Simplification
export DOCKER_HOST=tcp://localhost:2376
export DOCKER_TLS_VERIFY=1
mkdir -p ~/.docker
cp ~/docker-ca/ca.pem ~/.docker/
cp ~/docker-ca/client-cert.pem ~/.docker/cert.pem
cp ~/docker-ca/client-key.pem ~/.docker/key.pem

docker container ls
