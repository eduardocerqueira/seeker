#date: 2022-10-24T17:19:57Z
#url: https://api.github.com/gists/ef0a814ccd128164a6b479fb1d131680
#owner: https://api.github.com/users/areed

#!/bin/bash

set -euo pipefail

RUNC_VERSION=1.1.4
CNI_VERSION=1.1.1
CONTAINERD_VERSION=1.6.8
STEP_VERSION=0.21.0

function bootstrap_cas() {
    step ca bootstrap --ca-url https://etcd.areed.ca.smallstep.com --fingerprint 08d75076faaa562126d707354c0a5aa72a807f3fe69683d69b4b8dcb9cd98e19 --context etcd --force
	step ca bootstrap --ca-url https://k8s.areed.ca.smallstep.com --fingerprint b61b90ef0f0f38507f836b15089619c31d95c8384a5282f0c3154106e19008d2 --context k8s --force
}

function install_runc() {
    curl -LO https://github.com/opencontainers/runc/releases/download/v${RUNC_VERSION}/runc.amd64
    install -m 755 runc.amd64 /usr/local/sbin/runc
    rm runc.amd64
    echo "Installed runc ${RUNC_VERSION}"
}

function install_containerd() {
    curl -LO https://github.com/containerd/containerd/releases/download/v${CONTAINERD_VERSION}/containerd-${CONTAINERD_VERSION}-linux-amd64.tar.gz
    tar Cxzvf /usr/local containerd-${CONTAINERD_VERSION}-linux-amd64.tar.gz
    rm containerd-${CONTAINERD_VERSION}-linux-amd64.tar.gz

    # configure containerd to use systemd's cgroup driver
    mkdir -p /etc/containerd
    containerd config default | sed 's/SystemdCgroup = false/SystemdCgroup = true/' > config.toml
    mv config.toml /etc/containerd/config.toml

    # add a systemd service to run containerd
    curl -L https://raw.githubusercontent.com/containerd/containerd/main/containerd.service > containerd.service
    mv containerd.service /lib/systemd/system/containerd.service
    systemctl daemon-reload
    systemctl enable --now containerd.service

    echo "Installed containerd ${CONTAINERD_VERSION}"
}

function install_cilium() {
    CILIUM_CLI_VERSION=$(curl -s https://raw.githubusercontent.com/cilium/cilium-cli/master/stable.txt)
    CLI_ARCH=amd64
    if [ "$(uname -m)" = "aarch64" ]; then CLI_ARCH=arm64; fi
    curl -L --fail --remote-name-all https://github.com/cilium/cilium-cli/releases/download/${CILIUM_CLI_VERSION}/cilium-linux-${CLI_ARCH}.tar.gz{,.sha256sum}
    sha256sum --check cilium-linux-${CLI_ARCH}.tar.gz.sha256sum
    tar xzvfC cilium-linux-${CLI_ARCH}.tar.gz /usr/local/bin
    rm cilium-linux-${CLI_ARCH}.tar.gz{,.sha256sum}
    cilium install
}

function install_k8s_commands() {
    apt update
    apt install -y apt-transport-https ca-certificates curl
    curl -fsSLo /usr/share/keyrings/kubernetes-archive-keyring.gpg https://packages.cloud.google.com/apt/doc/apt-key.gpg
    echo "deb [signed-by=/usr/share/keyrings/kubernetes-archive-keyring.gpg] https://apt.kubernetes.io/ kubernetes-xenial main" | tee /etc/apt/sources.list.d/kubernetes.list
    apt update
    apt install -y kubelet kubeadm kubectl
    apt-mark hold kubelet kubeadm kubectl
}

function configure_host() {
    modprobe overlay
    modprobe br_netfilter
    modprobe nf_conntrack
    modprobe ip_vs
    modprobe ip_vs_rr
    modprobe ip_vs_wrr
    modprobe ip_vs_sh
    echo "overlay" > /etc/modules-load.d/k8s.conf
    echo "br_netfiler" > /etc/modules-load.d/k8s.conf
    echo 'nf_conntrack' >> /etc/modules-load.d/k8s.conf
    echo 'ip_vs' >> /etc/modules-load.d/k8s.conf
    echo 'ip_vs_rr' >> /etc/modules-load.d/k8s.conf
    echo 'ip_vs_wrr' >> /etc/modules-load.d/k8s.conf
    echo 'ip_vs_sh' >> /etc/modules-load.d/k8s.conf

    echo "net.bridge.bridge-nf-call-iptables = 1" >> /etc/sysctl.conf
    echo "net.ipv4.conf.all.forwarding = 1" >> /etc/sysctl.conf
    sysctl --system
    sysctl -p
}

function pull_control_plane_images() {
    kubeadm config images pull
}

function install_step() {
    wget https://dl.step.sm/gh-release/cli/docs-cli-install/v${STEP_VERSION}/step-cli_${STEP_VERSION}_amd64.deb
    dpkg -i step-cli_${STEP_VERSION}_amd64.deb
}

function make_pki_dir() {
	ok "Creating etcd PKI directories"
	mkdir -p /etc/kubernetes/pki/etcd
	mkdir -p /etc/kubernetes/pki/etcd
	mkdir -p /var/lib/kubelet/pki
}

function preflight() {
    kubeadm init phase preflight
    step context list | grep k8s
    step context list | grep etcd
}

function write_cas() {
    ok "Write CA files"

    step context select etcd
	step ca root > ca.crt
	mv ca.crt /etc/kubernetes/pki/etcd/

    step context select k8s
	step ca root > ca.crt
	mv ca.crt /etc/kubernetes/pki/
}

# The cert for the etcd server listening on :2379 for connections from kube-apiserver
function issue_etcd_server_cert() {
	ok "Issuing etcd server certificate"
    step context select etcd
	step ca certificate kube-etcd server.crt server.key \
		--provisioner andrew@smallstep.com \
		--provisioner-password-file /home/ubuntu/password.txt \
		--san "${HOSTNAME}" \
		--san "${HOST_IP}" \
		--san localhost \
		--san 127.0.0.1
	mv server.crt server.key /etc/kubernetes/pki/etcd/
}

# The cert for the etcd server listening on :2380 for connections from etcd peers
function issue_etcd_peer_cert() {
	ok "Issuing etcd peer certificate"
    step context select etcd
	step ca certificate kube-etcd-peer peer.crt peer.key \
		--provisioner andrew@smallstep.com \
		--provisioner-password-file /home/ubuntu/password.txt \
		--san "${HOSTNAME}" \
		--san "${HOST_IP}" \
		--san localhost \
		--san 127.0.0.1
	mv peer.crt peer.key /etc/kubernetes/pki/etcd/
}

# The cert kubeadm uses in preflight checks
function issue_etcd_healthcheck_cert() {
	ok "Issuing etcd healthcheck certificate"
    step context select etcd
	step ca certificate kubeadm healthcheck-client.crt healthcheck-client.key \
		--provisioner andrew@smallstep.com \
		--provisioner-password-file /home/ubuntu/password.txt
	mv healthcheck-client.crt healthcheck-client.key /etc/kubernetes/pki/etcd/
}

# The cert the Kubernetes API uses to connect to etcd
function issue_apiserver_etcd_client_cert() {
	ok "Issuing API server etcd client certificate"
    step context select etcd
	step ca certificate kube-apiserver-etcd-client apiserver-etcd-client.crt apiserver-etcd-client.key \
		--provisioner andrew@smallstep.com \
		--provisioner-password-file /home/ubuntu/password.txt \
		--set "Org=system:masters"
	mv apiserver-etcd-client.crt apiserver-etcd-client.key /etc/kubernetes/pki/
}

# The cert for the kube-apiserver listening on :6443
function issue_api_server_cert() {
	ok "Issuing kube-apiserver server certificate"
	step context select k8s
	step ca certificate kube-apiserver apiserver.crt apiserver.key \
		--provisioner andrew@smallstep.com \
		--provisioner-password-file /home/ubuntu/password.txt \
		--san "${HOSTNAME}" \
		--san "${HOST_IP}" \
		--san 10.96.0.1 \
		--san kubernetes \
		--san kubernetes.default \
		--san kubernetes.default.svc \
		--san kubernetes.default.svc.cluster.local
	mv apiserver.crt apiserver.key /etc/kubernetes/pki/
}

# The cert the Kubernetes API uses to connect to kubelets for logs and execs
function issue_api_kubelet_client_cert() {
	ok "Issuing API server kubelet client certificate"
	step context select k8s
	step ca certificate kube-apiserver-kubelet-client apiserver-kubelet-client.crt apiserver-kubelet-client.key \
		--provisioner andrew@smallstep.com \
		--provisioner-password-file /home/ubuntu/password.txt \
		--set "Org=system:masters"
	mv apiserver-kubelet-client.crt apiserver-kubelet-client.key /etc/kubernetes/pki/
}

# The cert for the kubelet listening on :10250
function issue_kubelet_server_cert() {
	ok "Issuing kubelet server certificate"
	step context select k8s
	step ca certificate "$HOSTNAME" kubelet.crt kubelet.key \
		--provisioner andrew@smallstep.com \
		--provisioner-password-file /home/ubuntu/password.txt
	mv kubelet.crt kubelet.key /var/lib/kubelet/pki
}

# The client cert the kubelet uses to connect to the kube-apiserver
function issue_kubelet_apiserver_client_cert() {
	ok "Issuing kubelet apiserver client certificate"
	step context select k8s
	# On a kubeadm install the public and private key are combined into the single file kubelet-client-current.pem
	step ca certificate "system:node:${HOSTNAME}" kubelet-apiserver-client.crt kubelet-apiserver-client.key \
		--provisioner andrew@smallstep.com \
		--provisioner-password-file /home/ubuntu/password.txt \
		--set "Org=system:nodes"
	mv kubelet-apiserver-client.crt kubelet-apiserver-client.key /var/lib/kubelet/pki/
}

function generate_kubeconfig() {
	FILENAME="${1}"
	CN="${2}"
	CREDENTIAL_NAME="${3}"
	GROUP="${4}"

	ok "Generating /etc/kubernetes/${FILENAME}.conf"
	step context select k8s
	step ca certificate "${CN}" "${FILENAME}.crt" "${FILENAME}.key" \
		--provisioner andrew@smallstep.com \
		--provisioner-password-file /home/ubuntu/password.txt \
		--set "Org=${GROUP}"
	KUBECONFIG="${FILENAME}.conf" kubectl config set-cluster default-cluster --server="https://${HOST_IP}:6443" --certificate-authority /etc/kubernetes/pki/ca.crt --embed-certs
	KUBECONFIG="${FILENAME}.conf" kubectl config set-credentials "${CREDENTIAL_NAME}" --client-key "${FILENAME}.key" --client-certificate "${FILENAME}.crt" --embed-certs
	KUBECONFIG="${FILENAME}.conf" kubectl config set-context default-system --cluster default-cluster --user "$CREDENTIAL_NAME"
	KUBECONFIG="${FILENAME}.conf" kubectl config use-context default-system
	mv "${FILENAME}.conf" /etc/kubernetes/
	rm "${FILENAME}.crt" "${FILENAME}.key"
}

function generate_kubeconfigs() {
	generate_kubeconfig "controller-manager" "system:kube-controller-manager" "default-controller-manager" ""
	generate_kubeconfig "scheduler" "system:kube-scheduler" "default-scheduler" ""
	generate_kubeconfig "admin" "kubernetes-admin" "default-admin" "system:masters"
	generate_kubeconfig "kubelet" "system:node:${HOSTNAME}" "default-auth" "system:nodes"
}

function generate_service_account_token_keypair() {
	step crypto keypair sa.pub sa.key --no-password --insecure
	mv sa.pub sa.key /etc/kubernetes/pki
}

function ok() {
	GREEN='\033[32m'
	NC='\033[0m'
	printf "${GREEN}${1}${NC}\n"
}

function init() {
    sudo kubeadm init phase etcd local
    sudo kubeadm init phase control-plane apiserver
    sudo kubeadm init phase control-plane scheduler
    sudo kubeadm init phase control-plane controller-manager
     # TODO modify the manifests
    sudo kubeadm init phase kubelet-start
    sudo kubeadm init phase addon kube-proxy
    sudo kubeadm init phase addon coredns
}

function main() {
    install_runc
    install_containerd
    install_k8s_commands
    configure_host
    pull_control_plane_images
    install_step
    bootstrap_cas
    make_pki_dir
    write_cas
    issue_etcd_server_cert
    issue_etcd_peer_cert
    issue_etcd_healthcheck_cert
    issue_apiserver_etcd_client_cert
    issue_api_server_cert
    issue_api_kubelet_client_cert
    issue_kubelet_server_cert
    issue_kubelet_apiserver_client_cert
    generate_kubeconfigs
    generate_service_account_token_keypair
    init
    install_cilium
}

main