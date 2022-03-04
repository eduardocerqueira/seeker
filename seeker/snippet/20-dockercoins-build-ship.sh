#date: 2022-03-04T16:56:06Z
#url: https://api.github.com/gists/80c61832ef2260b6915af4f5f760549c
#owner: https://api.github.com/users/prom3theu5

export DOCKERHUB_USERNAME=...

docker login --username ${DOCKERHUB_USERNAME}

# Build & publish
cd ~/orchestration-workshop/dockercoins/
for service in hasher rng worker webui; do
  docker-compose build ${service}
  docker image tag dockercoins_${service} ${DOCKERHUB_USERNAME}/dockercoins_${service}:1.0
  docker push ${DOCKERHUB_USERNAME}/dockercoins_${service}:1.0
done
