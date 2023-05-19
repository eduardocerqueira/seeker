#date: 2023-05-19T16:50:17Z
#url: https://api.github.com/gists/280a3771ebebfec01466493c823e37fa
#owner: https://api.github.com/users/Div123

#!/usr/bin/env bash

jq -r '.items[] | . as $pod | select(.spec.volumes !=null) | .spec.volumes[] | select(.persistentVolumeClaim !=null ) | [$pod.metadata.namespace, .persistentVolumeClaim.claimName] | @csv' all-pods.json | sort | uniq > pod-volumes.csv

if [ ! -f cronjobs.json ]; then
	echo "Generating list of CrobJobs"
	oc get cronjob --all-namespaces -o json > cronjobs.json
fi
jq -r '.items[] | . as $item | select(.spec.jobTemplate.spec.template.spec.volumes !=null) | .spec.jobTemplate.spec.template.spec.volumes[] | select(.persistentVolumeClaim !=null ) | [$item.metadata.namespace, .persistentVolumeClaim.claimName] | @csv' cronjobs.json | sort | uniq > cronjob-volumes.csv

if [ ! -f deployments.json ]; then
	echo "Generating list of Deployments"
	oc get deployment,deploymentConfig --all-namespaces -o json > deployments.json
fi
jq -r '.items[] | . as $item | select(.spec.template.spec.volumes !=null) | .spec.template.spec.volumes[] | select(.persistentVolumeClaim !=null ) | [$item.metadata.namespace, .persistentVolumeClaim.claimName] | @csv' deployments.json | sort | uniq > deployment-volumes.csv

if [ ! -f statefulsets.json ]; then
	echo "Generating list of Statefulsets"
	oc get statefulset --all-namespaces -o json > statefulsets.json
fi
jq -r '.items[] | . as $item | select(.spec.template.spec.volumes !=null) | .spec.template.spec.volumes[] | select(.persistentVolumeClaim !=null ) | [$item.metadata.namespace, .persistentVolumeClaim.claimName] | @csv' statefulsets.json | sort | uniq > statefulset-volumes.csv
jq -r '.items[] | . as $item | select(.spec.volumeClaimTemplates !=null) | .spec.volumeClaimTemplates[] | [$item.metadata.namespace, $item.metadata.name + "-" + .metadata.name + "-0"] | @csv' statefulsets.json | sort | uniq > statefulset-vct.csv


if [ ! -f pvc.json ]; then
	echo "Generating list of PVCs"
	oc get pvc --all-namespaces -o json > pvc.json
fi
jq -r '.items[] | [.metadata.namespace, .metadata.name] | @csv' 'pvc.json' | sort | uniq > pvc.csv

# Includes only existing PVCs
cat statefulset-vct.csv pvc.csv | sort | uniq -c | awk '$1>1' | cut -c 6- > statefulset-vct2.csv

cat pod-volumes.csv cronjob-volumes.csv deployment-volumes.csv statefulset-volumes.csv statefulset-vct2.csv | sort | uniq > pvc-ref.csv
cat pvc-ref.csv pvc.csv | sort | uniq -c | awk '$1<2' | cut -c 6- > orphan-pvc.csv

wc -l orphan-pvc.csv