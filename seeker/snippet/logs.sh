#date: 2023-07-06T16:53:07Z
#url: https://api.github.com/gists/97d16fef10cb0db12b9d2a6f208e119d
#owner: https://api.github.com/users/borgez

#!/usr/bin/env bash



docker ps --filter name=.*kube-apiserver.*$ --format='{{.ID}} {{.Image}} {{.Names}} ' | grep -v pause | awk '{print $1}' | xargs -r -I {} docker logs {}
docker ps --filter name=.*kube-scheduler.*$ --format='{{.ID}} {{.Image}} {{.Names}} ' | grep -v pause | awk '{print $1}' | xargs -r -I {} docker logs {}
docker ps --filter name=.*kube-controller-manager.*$ --format='{{.ID}} {{.Image}} {{.Names}} ' | grep -v pause | awk '{print $1}' | xargs -r -I {} docker logs {}
docker ps --filter name=.*kube-proxy.*$ --format='{{.ID}} {{.Image}} {{.Names}} ' | grep -v pause | awk '{print $1}' | xargs -r -I {} docker logs {}
docker ps --filter name=.*k8s_etcd.*$ --format='{{.ID}} {{.Image}} {{.Names}} ' | grep -v pause | awk '{print $1}' | xargs -r -I {} docker logs {}
docker ps --filter name=.*k8s_coredns.*$ --format='{{.ID}} {{.Image}} {{.Names}} ' | grep -v pause | awk '{print $1}' | xargs -r -I {} docker logs {}


ctr -n k8s.io c ls | grep kube-apiserver | awk '{print $1}' | xargs -r -I {} find /var/log/containers/ -name "*{}.log" | xargs -r -I [] cat []
ctr -n k8s.io c ls | grep kube-scheduler | awk '{print $1}' | xargs -r -I {} find /var/log/containers/ -name "*{}.log" | xargs -r -I [] cat []
ctr -n k8s.io c ls | grep kube-controller-manager | awk '{print $1}' | xargs -r -I {} find /var/log/containers/ -name "*{}.log" | xargs -r -I [] cat []
ctr -n k8s.io c ls | grep kube-proxy | awk '{print $1}' | xargs -r -I {} find /var/log/containers/ -name "*{}.log" | xargs -r -I [] cat []
ctr -n k8s.io c ls | grep etcd | awk '{print $1}' | xargs -r -I {} find /var/log/containers/ -name "*{}.log" | xargs -r -I [] cat []
ctr -n k8s.io c ls | grep coredns | awk '{print $1}' | xargs -r -I {} find /var/log/containers/ -name "*{}.log" | xargs -r -I [] cat []


crictl ps | grep kube-apiserver | awk '{print $1}' | xargs -r -I {} crictl logs {}
crictl ps | grep kube-scheduler | awk '{print $1}' | xargs -r -I {} crictl logs {}
crictl ps | grep kube-controller-manager | awk '{print $1}' | xargs -r -I {} crictl logs {}
crictl ps | grep kube-proxy | awk '{print $1}' | xargs -r -I {} crictl logs {}
crictl ps | grep etcd | awk '{print $1}' | xargs -r -I {} crictl logs {}
crictl ps | grep coredns | awk '{print $1}' | xargs -r -I {} crictl logs {}
