#date: 2022-06-02T17:17:52Z
#url: https://api.github.com/gists/c9af2bf05a1bbb425da086f3ae336eb7
#owner: https://api.github.com/users/kerus1024

#!/bin/bash
# Rocky Linux 8
# https://www.linuxtechi.com/how-to-install-kubernetes-cluster-rhel/
set -x
set -e

# master 01 or 02,03/worker
SWITCH_PRIMARY=y


# 1. Disable swap space
sudo swapoff -a

# 2. Disable SELinux
sudo sed -i 's/^SELINUX=enforcing$/SELINUX=permissive/' /etc/selinux/config
sudo setenforce 0

# 3. configure hosts
cat <<EOF | sudo tee --append /etc/hosts
10.53.75.1  k8s-master01
10.53.75.2  k8s-master02
10.53.75.3  k8s-master03

10.53.76.1  k8s-worker01
10.53.70.1  k8s-lb
EOF

# 4. install traffic control
sudo dnf install -y iproute-tc

# 5. Allow port
sudo systemctl disable firewalld
sudo systemctl stop firewalld

# https://kubernetes.io/ko/docs/reference/ports-and-protocols/
# master
# sudo firewall-cmd --permanent --add-port=6443/tcp
# sudo firewall-cmd --permanent --add-port=2379-2380/tcp
# sudo firewall-cmd --permanent --add-port=10250/tcp
# sudo firewall-cmd --permanent --add-port=10251/tcp
# sudo firewall-cmd --permanent --add-port=10252/tcp
# sudo firewall-cmd --reload

# worker
# sudo firewall-cmd --permanent --add-port=10250/tcp
# sudo firewall-cmd --permanent --add-port=30000-32767/tcp
# sudo firewall-cmd --reload


# 6. install CRI-O
# CRI is needed v1.23 or later.

# Linux Kernel 
#   overlay: CRI-O overlayFS
#   br_netfilter: https://unix.stackexchange.com/questions/499756/how-does-iptable-work-with-linux-bridge
cat <<EOF | sudo tee /etc/modules-load.d/k8s.conf 
overlay
br_netfilter
EOF

sudo modprobe overlay
sudo modprobe br_netfilter


# 7. netfiltter iptables bridge 
cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-iptables  = 1
#net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
EOF

sudo sysctl --system

# 8. CRI-O download
export VERSION=1.23
sudo curl -L -o /etc/yum.repos.d/devel:kubic:libcontainers:stable.repo https://download.opensuse.org/repositories/devel:/kubic:/libcontainers:/stable/CentOS_8/devel:kubic:libcontainers:stable.repo
sudo curl -L -o /etc/yum.repos.d/devel:kubic:libcontainers:stable:cri-o:$VERSION.repo https://download.opensuse.org/repositories/devel:kubic:libcontainers:stable:cri-o:$VERSION/CentOS_8/devel:kubic:libcontainers:stable:cri-o:$VERSION.repo
sudo dnf clean all
sudo dnf install -y cri-o

sudo systemctl enable crio
sudo systemctl start crio

# 9. k8s repo
#   rhel7
cat <<EOF | sudo tee /etc/yum.repos.d/kubernetes.repo
[kubernetes]
name=Kubernetes
baseurl=https://packages.cloud.google.com/yum/repos/kubernetes-el7-x86_64
enabled=1
gpgcheck=1
repo_gpgcheck=0
gpgkey=https://packages.cloud.google.com/yum/doc/yum-key.gpg https://packages.cloud.google.com/yum/doc/rpm-package-key.gpg
exclude=kubelet kubeadm kubectl
EOF

sudo dnf clean all

# --disableexcludes for version lock
# must be upgraded by kubeadm!
sudo dnf install -y kubelet-1.23.5-0 kubeadm-1.23.5-0 kubectl-1.23.5-0 --disableexcludes=kubernetes
sudo systemctl enable kubelet
sudo systemctl start kubelet


if [ "$SWITCH_PRIMARY" = "y" ]; then 
  # master01

  sudo kubeadm init --pod-network-cidr=192.168.0.0/16

  # Your Kubernetes control-plane has initialized successfully!
  # To start using your cluster, you need to run the following as a regular user:
  sudo mkdir -p $HOME/.kube
  sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
  sudo chown $(id -u):$(id -g) $HOME/.kube/config

  # Alternatively, if you are the root user, you can run:
  #   export KUBECONFIG=/etc/kubernetes/admin.conf
  #
  # You should now deploy a pod network to the cluster.
  # Run "kubectl apply -f [podnetwork].yaml" with one of the options listed at:
  #   https://kubernetes.io/docs/concepts/cluster-administration/addons/
  #
  # Then you can join any number of worker nodes by running the following on each as root:
  #
  # kubeadm join 10.100.100.2:6443 --token b9epfe.tiftlfiemwzkak1p \
  #         --discovery-token-ca-cert-hash sha256:66b37ff6849cc7cca4e4c6c27b61fb0b289cd11b526cb8bc3b61c6901734546c

  # 10. Calico CNI
  #            hard coded value: 192.168.0.0/16
  kubectl create -f https://docs.projectcalico.org/manifests/tigera-operator.yaml
  kubectl create -f https://docs.projectcalico.org/manifests/custom-resources.yaml

  # 
  # Run the command on another control-plane
  # kubeadm join 10.53.75.1:6443 --token 123456.qwertyuiopqwerty \
  #           --discovery-token-ca-cert-hash sha256:bvyq5j2y3245n5r5pd8ht33sypknmrkcxnaesbjcaocui0a7ksxdcjkojgosr0oq \
  #           --control-plane \ 
  #           --certificate-key aivocn0fkcub40uakcg4p4f33jiqhg40mj1r4y5j6p3b4d6n3w6ujxmc6udu5ny3

else
  echo "Run command kubeadm join ...."
fi

