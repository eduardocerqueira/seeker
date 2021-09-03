#date: 2021-09-03T16:55:47Z
#url: https://api.github.com/gists/6a5898fa6827b0781b12542927a04112
#owner: https://api.github.com/users/ongeri

#!/bin/sh
git clone --single-branch --branch v1.7.2 https://github.com/rook/rook.git
cd rook/cluster/examples/kubernetes/ceph
kubectl create -f crds.yaml -f common.yaml -f operator.yaml
kubectl create -f cluster.yaml
kubectl create -f csi/rbd/storageclass.yaml
kubectl create -f filesystem.yaml
kubectl create -f csi/cephfs/storageclass.yaml
kubectl patch storageclass rook-ceph-block -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
