#date: 2023-05-19T16:50:17Z
#url: https://api.github.com/gists/280a3771ebebfec01466493c823e37fa
#owner: https://api.github.com/users/Div123

#!/usr/bin/env bash

find . -type f -name 'all-*.json' -deleted

test -f all-image-streams.json || oc get is -o json --all-namespaces > all-image-streams.json
test -f all-images.json || oc -n openshift get Image -o json  --all-namespaces > all-images.json
test -f all-Pod.json || oc get Pod -o json --all-namespaces | jq 'del(.items[].spec.containers[].env) | del(.items[].metadata.annotations)' > all-Pod.json

test -f all-BuildConfig.json || oc get BuildConfig -o json --all-namespaces > all-BuildConfig.json
test -f all-Build.json || oc get Build -o json --all-namespaces > all-Build.json

test -f all-Deployment.json || oc get Deployment -o json --all-namespaces | jq 'del(.items[].spec.template.spec.containers[].env) | del(.items[].metadata.annotations)' > all-Deployment.json
test -f all-ReplicaSet.json || oc get ReplicaSet --all-namespaces -o json | jq 'del(.items[].spec.template.spec.containers[].env) | del(.items[].metadata.annotations)' > all-ReplicaSet.json

test -f all-DeploymentConfig.json || oc get DeploymentConfig -o json --all-namespaces | jq 'del(.items[].spec.template.spec.containers[].env) | del(.items[].metadata.annotations)' > all-DeploymentConfig.json
test -f all-ReplicationController.json || oc get ReplicationController --all-namespaces -o json | jq 'del(.items[].spec.template.spec.containers[].env) | del(.items[].metadata.annotations)' > all-ReplicationController.json

test -f all-CronJob.json || oc get CronJob --all-namespaces -o json > all-CronJob.json
test -f all-Job.json || oc get Job --all-namespaces -o json > all-Job.json


test -f all-StatefulSet.json || oc get StatefulSet --all-namespaces -o json > all-StatefulSet.json


