#date: 2025-05-22T16:55:06Z
#url: https://api.github.com/gists/fe4747e342aeca9b7a878788d3cfc6e2
#owner: https://api.github.com/users/nurulc-oxb

for network in $(docker network ls --format '{{.Name}}'); do
    echo "Network: $network"
    docker network inspect "$network" | grep -i "subnet"
done