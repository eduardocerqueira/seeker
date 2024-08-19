#date: 2024-08-19T17:03:46Z
#url: https://api.github.com/gists/7c280079bb59c59e5ef0ecd91c0996cd
#owner: https://api.github.com/users/steve-fraser

#!/bin/bash
# Assuming you've stored the extracted labels in a variable
NODE_LABELS=$(sudo cat /etc/systemd/system/kubelet.service.d/30-kubelet-extra-args.conf | grep 'KUBELET_EXTRA_ARGS' | grep -oP '(?<=--node-labels=)[^ ]*')

# Add the node labels to the JSON configuration
sudo jq --arg labels "$NODE_LABELS" '.nodeLabels = ($labels | split(","))' /etc/kubernetes/kubelet/kubelet-config.json > /etc/kubernetes/kubelet/kubelet-config-modified.json
sudo mv /etc/kubernetes/kubelet/kubelet-config-modified.json /etc/kubernetes/kubelet/kubelet-config.json
sudo systemctl restart kubelet 