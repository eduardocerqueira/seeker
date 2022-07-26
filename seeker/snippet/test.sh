#date: 2022-07-26T17:00:49Z
#url: https://api.github.com/gists/25eec4471cc75a74efd63edc891acdd9
#owner: https://api.github.com/users/alexeldeib


total=10;
count=0; 
# tests pass unreliably depending on ordering.
# ephemeral os test calls mock.EXPECT().ANY() which causes Kubelet tests to pass
for i in $(seq 1 $total); do 
    ginkgo -r -focus="PUT Managed Cluster (Ephemeral OS|Kubelet disk type)" ./microsoft.com/containerservice/server/operations/managedcluster -nodes=1 -v;
    ret="$?";
    if [ "$ret" != "0" ]; then 
        count=$((count+1));
    fi
done;
echo "$count/$total runs failed"
# for ace: 4-6/10 failures

# test fails due to lack of calls to mock.EXPECT() from other tests
count=0;
for i in $(seq 1 $total); do 
    ginkgo -r -focus="PUT Managed Cluster (Kubelet disk type)" ./microsoft.com/containerservice/server/operations/managedcluster -nodes=1 -v;
    ret="$?";
    if [ "$ret" != "0" ]; then 
        count=$((count+1));
    fi
done;
echo "$count/$total runs failed"
# for ace: 10/10 failures
