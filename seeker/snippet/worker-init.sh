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

# Join the cluster as a worker

# On the Control Plane:
# To get a token run this on cp: "**********"
# aw5qps.phtxf9jpq4xubmy8

# openssl x509 -pubkey \
#     -in /etc/kubernetes/pki/ca.crt | openssl rsa \
#     -pubin -outform der 2>/dev/null | openssl dgst \
#     -sha256 -hex | sed 's/ˆ.* //'

# SHA2-256(stdin)= 9db624c541758a3c016e49f932c264d422ccff676b8533fdeece49510b7fdf11

sudo tee -a /etc/hosts <<EOF
10.0.30.55 k8scp
EOF

sudo kubeadm join \
    --token aw5qps.phtxf9jpq4xubmy8 \
    k8scp:6443 \
    --discovery-token-ca-cert-hash \
    sha256:9db624c541758a3c016e49f932c264d422ccff676b8533fdeece49510b7fdf11

# Update crictl config
sudo crictl config --set \
    runtime-endpoint=unix:///run/containerd/containerd.sock \
    --set image-endpoint=unix:///run/containerd/containerd.sock

sudo cat /etc/crictl.yamlnerd.sock \
    --set image-endpoint=unix:///run/containerd/containerd.sock

sudo cat /etc/crictl.yaml