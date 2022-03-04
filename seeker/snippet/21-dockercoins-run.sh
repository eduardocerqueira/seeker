#date: 2022-03-04T16:56:06Z
#url: https://api.github.com/gists/80c61832ef2260b6915af4f5f760549c
#owner: https://api.github.com/users/prom3theu5

# Create network
docker network create --driver overlay dockercoins

# Run
docker service create --network dockercoins --name redis redis
for service in hasher rng worker webui; do
  docker service create --network dockercoins --name ${service} ${DOCKERHUB_USERNAME}/dockercoins_${service}:1.0
done

docker service update webui --publish-add 8080:80
