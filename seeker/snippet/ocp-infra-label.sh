#date: 2022-10-18T17:20:30Z
#url: https://api.github.com/gists/ec8d76807c47a5f7856b5e6c7021944b
#owner: https://api.github.com/users/tosin2013

#!/bin/bash 

#m6i.2xlarge
# oc get nodes
#https://red-hat-storage.github.io/ocs-training/training/infra-nodes/ocs4-infra-nodes.html
# https://docs.openshift.com/container-platform/4.11/machine_management/creating-infrastructure-machinesets.html#infrastructure-moving-router_creating-infrastructure-machinesets
# https://access.redhat.com/documentation/en-us/red_hat_advanced_cluster_management_for_kubernetes/2.6/html/install/installing#tolerations
array=( worker1 worker2 worker3 )
for i in "${array[@]}"
do
	echo "$i"
    oc label node $i node-role.kubernetes.io/infra=""
    oc label node $i cluster.ocs.openshift.io/openshift-storage=""
    #oc adm taint node $i node.ocs.openshift.io/storage="true":NoSchedule # if you only want these nodes to run storage pods
    #oc patch -n openshift-ingress-operator ingresscontroller/default --patch '{"spec":{"replicas": 3}}' --type=merge
done

