#date: 2025-01-15T16:53:15Z
#url: https://api.github.com/gists/c4cfceab7efeedf8a5ab7257f3540f23
#owner: https://api.github.com/users/benjaminpreiss

#!/bin/bash
# talos-storage-patch.sh

# Get the list of non-control-plane node IPs
node_ips=$(kubectl get nodes --selector='!node-role.kubernetes.io/control-plane' -o jsonpath='{range .items[*]}{.status.addresses[?(@.type=="InternalIP")].address}{"\n"}{end}')

# Iterate over each IP and apply the patch
while IFS= read -r ip; do
    if [ -n "$ip" ]; then
        echo "Applying patch to node: $ip"
        talosctl -n $ip patch machineconfig -p @talos-storage-patch.yaml
        
        # Optional: Add a small delay between operations
        sleep 2
    fi
done <<< "$node_ips"

echo "Patch applied to all non-control-plane nodes."