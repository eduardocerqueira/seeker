#date: 2022-03-04T16:56:06Z
#url: https://api.github.com/gists/80c61832ef2260b6915af4f5f760549c
#owner: https://api.github.com/users/prom3theu5

# On another node than master
export MASTER_PRIVATE_IP=10.X.Y.Z

export DOCKER_HOST=tcp://${MASTER_PRIVATE_IP}:2376
export DOCKER_TLS_VERIFY=1
scp -r ${MASTER_PRIVATE_IP}:~/.docker ~/

docker container ls
