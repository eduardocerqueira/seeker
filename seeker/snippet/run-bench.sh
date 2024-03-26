#date: 2024-03-26T17:03:32Z
#url: https://api.github.com/gists/6ec561cc6afa58c5258869b03c21e5b8
#owner: https://api.github.com/users/maicmn

#! /bin/bash

# list workers
worker_nodes=$(kubectl get nodes -o jsonpath='{.items[*].metadata.name}')
wget -O job.yaml https://raw.githubusercontent.com/aquasecurity/kube-bench/main/job.yaml -q
ClusterName=$(kubectl config current-context)
echo $worker_nodes
# iterate over
for node in $worker_nodes; do
    # get hostname
    hostname_label=$(kubectl describe node $node | grep 'kubernetes.io/hostname' | cut -d '=' -f 2 )

    # update job.yaml
    awk -v hostname=$hostname_label '
    /containers:/ {
        print "      nodeSelector:";
        print "        kubernetes.io/hostname: " hostname;
    }
    {print}
    ' job.yaml > temp.yaml
    # apply jobs
    kubectl apply -f temp.yaml
    # wait 15 sec
    sleep 15
    # get kube-bench pod name
    export pod_name=$(kubectl get pods --selector=job-name=kube-bench -o jsonpath='{.items[*].metadata.name}')
    # save result to log files
    echo "ClusterName: $ClusterName" > kube-bench-$node.log
    echo "Máquina: $node" >> kube-bench-$node.log
    kubectl logs $pod_name >> kube-bench-$node.log

    # dele jobs
    kubectl delete job kube-bench
    unset pod_name

done

# delete temp files
rm temp.yaml
rm job.yaml
