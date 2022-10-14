#date: 2022-10-14T17:17:30Z
#url: https://api.github.com/gists/9493514651bd881ca08de89e7302128a
#owner: https://api.github.com/users/josephbharrison

#!/usr/bin/env bash

namespace=$1
service=$2

usage(){
  usage: kube-proxy <namespace> [service]
}
[[ -z $namespace ]] && usage
# kubectl command
k="kubectl -n $namespace"

# return list of services
services(){
    $k get svc -o json 2> /dev/null | jq -r .items[].spec.selector.app
}

# return list of ports
ports(){
    $k get svc -o json 2> /dev/null | jq .items[].spec.ports[0].port
}

# return port by service name
port(){
    service=$1
    $k get svc $service -o json 2> /dev/null | jq .spec.ports[0].port
}

# return first viable service
guess_service(){
    i=0
    for service in $(services)
    do
       ports=($(ports))
       port=${ports[$i]} && i=$((i+1))
       [[ $port == null ]] && continue
       echo $service $port
       break
    done
}

# discover service spec
[[ -z $service ]] && export spec=($(guess_service)) && service=${spec[0]} && port=${spec[1]}

# start the proxy
$k proxy &> /dev/null &

# return service URL
echo http://localhost:8001/api/v1/proxy/namespaces/$namespace/services/$service:$(port $service)