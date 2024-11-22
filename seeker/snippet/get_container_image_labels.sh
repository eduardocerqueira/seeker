#date: 2024-11-22T16:57:08Z
#url: https://api.github.com/gists/e501286a9eda2e4605c202ff6caf8735
#owner: https://api.github.com/users/juanmancebo

#!/bin/sh
export IMAGE=<YOUR CONTAINER IMAGE>

echo $(docker inspect --format '{{range $key,$value := .Config.Labels}}{{$key}}={{$value}}\n{{end}}' ${IMAGE})
##alternative using jq
docker inspect ${IMAGE}) |jq -r '.[0].Config.Labels'