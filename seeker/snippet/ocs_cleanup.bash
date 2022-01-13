#date: 2022-01-13T17:21:17Z
#url: https://api.github.com/gists/ef04ac10ffab4975894bf6581c6526bc
#owner: https://api.github.com/users/serveapps


#!/bin/bash
set -x

#Delete any apps consuming PVCs based on OCS storageclasses, and then delete the PVs
#Delete the StorageCluster object without deleting its dependents.
oc delete -n openshift-storage storagecluster --all --wait=true --cascade=false
#Delete the Noobaa resource.
oc delete -n openshift-storage noobaa noobaa --wait=true 
#Wait for the Noobaa PVC to be automatically deleted.
oc wait -n openshift-storage --for delete pvc -l noobaa-core=noobaa --timeout=5m
#Delete the CephCluster resource and wait for it to finish deleting.
oc delete -n openshift-storage cephcluster --all --wait=true  --timeout=5m
#Delete the Namespaces and wait for them to finish deleting.
oc delete project openshift-storage --wait=true  --timeout=5m
# remove the taint from storage nodes
oc adm taint nodes --all node.ocs.openshift.io/storage-
# unlabel the nodes the storage nodes
oc label node  --all cluster.ocs.openshift.io/openshift-storage-

