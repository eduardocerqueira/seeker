#date: 2022-02-03T17:05:46Z
#url: https://api.github.com/gists/8ace11da8fceed7e98758b01cd79087c
#owner: https://api.github.com/users/akhepcat

#!/bin/bash

if [ -n "$(command -v docker)" ]
then
        # Check if it's installed at all...
        if [ -n "$(docker container ls --all | grep grafana)" ]
        then
                # it's installed, so first check if it's running...
                if [ -n "$(docker container ls | grep grafana)" ]
                then
                        # it's running, so stop it
                        id=$(docker container ls --latest | grep grafana | awk '{print $1}')
                        docker stop $id
                fi
                # Now that it's stopped, remove it for the upgrade cycle
                docker rm $id
        fi

        # First prune everything no longer used
        docker system prune --force

        # It's now (no longer) installed, so install it
        docker run -d --restart=always --pull=always --name=grafana -p 3000:3000 -v /var/lib/grafana:/var/lib/grafana grafana/grafana-oss

        # Now we check to make sure it's running

        docker container ls | grep -i "grafana"
        echo ""
        netstat -tulpn | grep 3000
fi

echo """
Useful commands:

show containers:        docker container ls
shell in container:     docker exec -it grafana /bin/bash
restart container:      docker restart grafana

install plugin:         docker exec -it grafana grafana-cli plugins install grafana-piechart-panel

"""
