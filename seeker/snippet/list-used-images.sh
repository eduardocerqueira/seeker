#date: 2023-05-19T16:50:17Z
#url: https://api.github.com/gists/280a3771ebebfec01466493c823e37fa
#owner: https://api.github.com/users/Div123

#!/usr/bin/env bash
set -o errexit #fail on first error
set -o pipefail

#docker-registry.pathfinder.gov.bc.ca/
#docker-registry.default.svc:5000/
#172.50.0.2:5000/

# Pods
#jq -cr '.items[].spec.containers[].image | select( . | contains("/openshift/"))' all-Pod.json | sort | uniq
#jq -cr '.items[] | .status // {} | .containerStatuses // [] | .[].image | select( . | contains("/openshift/")) ' all-Pod.json | sort | uniq
#jq -cr '.items[] | .status // {} | .containerStatuses // [] | .[].imageID | select( . | contains("/openshift/"))' all-Pod.json | sort | uniq | sed 's|docker-pullable://||g'

# Deployment
#jq -cr '.items[].spec.template.spec.containers[].image | select( . | contains("/openshift/"))' all-Deployment.json | sort | uniq
#jq -cr '.items[].spec.template.spec.containers[].image | select( . | contains("/openshift/"))' all-ReplicaSet.json | sort | uniq

# DeploymentConfig
#jq -cr '.items[].spec.template.spec.containers[].image | select( . | contains("/openshift/"))' all-DeploymentConfig.json | sort | uniq
#jq -cr '.items[].spec.template.spec.containers[].image | select( . | contains("/openshift/"))' all-ReplicationController.json | sort | uniq


# Build
## jq -cr '.items[].spec.strategy.type' all-Build.json | sort | uniq -c
## jq -cr '.items[].spec.strategy | select(.dockerStrategy and .dockerStrategy.from) | .dockerStrategy.from.kind' all-Build.json | sort | uniq -c
## jq -cr '.items[].spec.strategy | select(.sourceStrategy and .sourceStrategy.from) | .sourceStrategy.from.kind' all-Build.json | sort | uniq -c
## jq -cr '.items[] | . as $item | select(.spec.strategy.type == "Docker" or .spec.strategy.type == "Source") | [.spec.strategy.dockerStrategy, .spec.strategy.sourceStrategy] | .[] | select(.from) | .from | . as $from | {kind:$item.kind, metadata:{name:$item.metadata.name, namespace:$item.metadata.namespace}, from:.}' all-Build.json | sort | uniq

jq -cr '.items[] | . as $item | select(.spec.strategy.type == "Docker" or .spec.strategy.type == "Source") | .spec.strategy.dockerStrategy // .spec.strategy.sourceStrategy | select(.from) | .from | . as $from | {kind:$item.kind, metadata:{name:$item.metadata.name, namespace:$item.metadata.namespace}, from:.}' all-Build.json > image-ref.json
jq -cr '.items[] | . as $item | select(.spec | .source // {} | .images) | .spec.source.images[] | .from | . as $from | {kind:$item.kind, metadata:{name:$item.metadata.name, namespace:$item.metadata.namespace}, from:.}' all-Build.json >> image-ref.json
jq -cr '.from | select(.kind == "DockerImage") | .name | select( . | contains("/openshift/"))' image-ref.json | sort | uniq

# BuildConfig
## jq -cr '.items[].spec.strategy.type' all-BuildConfig.json | sort | uniq -c
jq -cr '.items[] | . as $item | select(.spec.strategy.type == "Docker" or .spec.strategy.type == "Source") | .spec.strategy.dockerStrategy // .spec.strategy.sourceStrategy | select(.from) | .from | . as $from | {kind:$item.kind, metadata:{name:$item.metadata.name, namespace:$item.metadata.namespace}, from:.}' all-BuildConfig.json > image-ref.json
jq -cr '.items[] | . as $item | select(.spec | .source // {} | .images) | .spec.source.images[] | .from | . as $from | {kind:$item.kind, metadata:{name:$item.metadata.name, namespace:$item.metadata.namespace}, from:.}' all-BuildConfig.json >> image-ref.txt
jq -cr '.from | select(.kind == "DockerImage") | .name | select( . | contains("/openshift/"))' image-ref.json | sort | uniq
