#date: 2023-12-27T16:55:25Z
#url: https://api.github.com/gists/4a4cfe4eecf0768814ed4f82abcaad76
#owner: https://api.github.com/users/SamEdwardes

#!/bin/bash

wget https: "**********"
tar -xvf LFS258_V2023-12-13_SOLUTIONS.tar.xz

sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y \
    curl \
    apt-transport-https \
    vim \
    git \
    wget \
    software-properties-common \
    lsb-release \
    ca-certificates

# disable swap - cloud providers do this already
sudo swapoff -a

# load modules
sudo modprobe overlay
sudo modprobe br_netfilter

# update kernel to allow necessary traffic
sudo tee /etc/sysctl.d/kubernetes.conf <<EOF
net.bridge.bridge-nf-call-ip6tables = 1
net.bridge.bridge-nf-call-iptables = 1
net.ipv4.ip_forward = 1
EOF

# apply the changes
sudo sysctl --system

# install the necessasry key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# install containerd software
sudo apt-get update
sudo apt-get install -y containerd.io
sudo containerd config default | sudo tee /etc/containerd/config.toml
sudo sed -e 's/SystemdCgroup = false/SystemdCgroup = true/g' -i /etc/containerd/config.toml
sudo systemctl restart containerd

# Add a new repository for Kubernetes
sudo tee -a /etc/apt/sources.list.d/kubernetes.list <<EOF
deb http://apt.kubernetes.io/ kubernetes-xenial main
EOF

curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install -y kubeadm=1.27.1-00 kubelet=1.27.1-00 kubectl=1.27.1-00
sudo apt-mark hold kubelet kubeadm kubectl

# check IP address and add to /etc/hosts
hostname -i
ip addr show

sudo tee -a /etc/hosts <<EOF
$(hostname -i) k8scp
EOF

# Create kubeamd-config.yaml
tee kubeadmin-config.yaml <<EOF
apiVersion: kubeadm.k8s.io/v1beta3
kind: ClusterConfiguration
kubernetesVersion: 1.27.1
controlPlaneEndpoint: "k8scp:6443"
networking:
  podSubnet: 192.168.0.0/16
EOF

# Initialize the cluster
sudo kubeadm init --config=kubeadmin-config.yaml --upload-certs | tee kubeadm-init.out

# Your Kubernetes control-plane has initialized successfully!

# To start using your cluster, you need to run the following as a regular user:

#   mkdir -p $HOME/.kube
#   sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
#   sudo chown $(id -u):$(id -g) $HOME/.kube/config

# Alternatively, if you are the root user, you can run:

#   export KUBECONFIG=/etc/kubernetes/admin.conf

# You should now deploy a pod network to the cluster.
# Run "kubectl apply -f [podnetwork].yaml" with one of the options listed at:
#   https://kubernetes.io/docs/concepts/cluster-administration/addons/

# You can now join any number of the control-plane node running the following command on each as root:

#   kubeadm join k8scp: "**********"
#         --discovery-token-ca-cert-hash sha256: "**********"
#         --control-plane --certificate-key 6c575b7ae0f7df0dc10f44e8799dccf2471f600bfdabb8b437d89d581ccf304f

# Please note that the certificate-key gives access to cluster sensitive data, keep it secret!
# As a safeguard, uploaded-certs will be deleted in two hours; If necessary, you can use
# "kubeadm init phase upload-certs --upload-certs" to reload certs afterward.

# Then you can join any number of worker nodes by running the following on each as root:

# kubeadm join k8scp: "**********"
#         --discovery-token-ca-cert-hash sha256: "**********"

# Reference the command to join the cluster
tail kubeadm-init.out

# Set up cluster access
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

# Set up Cillium - Container Network Interface (CNI)
kubectl apply -f ~/LFS258/SOLUTIONS/s_03/cilium-cni.yaml

# Add bash completions
sudo apt-get install - y bash-completion
source <(kubectl completion bash)
echo "source <(kubectl completion bash)" >> $HOME/.bashrc

# Update crictl config
sudo crictl config --set \
    runtime-endpoint=unix:///run/containerd/containerd.sock \
    --set image-endpoint=unix:///run/containerd/containerd.sock

sudo cat /etc/crictl.yamltl completion bash)" >> $HOME/.bashrc

# Update crictl config
sudo crictl config --set \
    runtime-endpoint=unix:///run/containerd/containerd.sock \
    --set image-endpoint=unix:///run/containerd/containerd.sock

sudo cat /etc/crictl.yaml