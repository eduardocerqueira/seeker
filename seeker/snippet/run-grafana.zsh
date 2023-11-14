#date: 2023-11-14T16:42:44Z
#url: https://api.github.com/gists/bdff74120c43828a21645267d3577ebe
#owner: https://api.github.com/users/hoelzro

#!/bin/zsh

set -e -u

REV=$(git rev-parse HEAD)

buildah build -t grafana-git-$REV .

trap "podman stop grafana-bisect-$REV" EXIT

podman run --detach --name grafana-bisect-$REV --rm --env-file=env-file -p 3000:3000 --mount type=bind,source=dash.yaml,destination=/etc/grafana/provisioning/dashboards/dash.yaml --mount type=bind,source=random.json,destination=/etc/dashboards/example-folder/random.json grafana-git-$REV

ready=''

# wait for container to become ready
sleep 2
for i in {1..60..5}; do
  dashboard_count=$(curl -s -u admin:admin http://localhost:3000/api/search | jq length)
  if [[ $dashboard_count -eq 2 ]] ; then
    ready=1
    break
  fi

  sleep 5
done

if [[ -z "$ready" ]] ; then
  echo "container never became ready" >&2
  exit 1
fi

curl -s -Lo /dev/null -f -u admin: "**********": application/json' -H 'Accept: application/json' -d '{"email":"xavier@example.com","login":"example","name":"Xavier Ample","password":"p4ss"}' http://localhost:3000/api/admin/users

sleep 1

dashboard_count=$(curl -s -f -u example:p4ss http://localhost:3000/api/search | jq length)
if [[ $dashboard_count -eq 2 ]] ; then
  exit 0
else
  exit 1
fi
else
  exit 1
fi