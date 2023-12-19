#date: 2023-12-19T16:41:49Z
#url: https://api.github.com/gists/a8376818ef4479cb726da82e523810bd
#owner: https://api.github.com/users/denismurphy

sudo apt-get update && sudo apt-get upgrade -y && sudo apt-get install

sudo reboot

curl -sfL https://get.k3s.io | sh -s - \
--write-kubeconfig-mode 644 \
--flannel-backend=none \
--disable servicelb \
--token some_random_password \
--disable-network-policy \
--disable "metrics-server" \
--disable-cloud-controller \
--disable local-storage \
--disable "traefik" \
--cluster-cidr="10.42.0.0/16,fdca:fe00:1234::/48" \
--service-cidr="10.43.0.0/16,fdca:fe00:5678::/112"

mkdir ~/.kube
sudo cat /etc/rancher/k3s/k3s.yaml > ~/.kube/config

CILIUM_CLI_VERSION=$(curl -s https://raw.githubusercontent.com/cilium/cilium-cli/main/stable.txt)
CLI_ARCH=amd64
if [ "$(uname -m)" = "aarch64" ]; then CLI_ARCH=arm64; fi
curl -L --fail --remote-name-all https://github.com/cilium/cilium-cli/releases/download/${CILIUM_CLI_VERSION}/cilium-linux-${CLI_ARCH}.tar.gz{,.sha256sum}
sha256sum --check cilium-linux-${CLI_ARCH}.tar.gz.sha256sum
sudo tar xzvfC cilium-linux-${CLI_ARCH}.tar.gz /usr/local/bin
rm cilium-linux-${CLI_ARCH}.tar.gz{,.sha256sum}

export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

cilium install --helm-set ipv6.enabled=true

cilium status --wait

cilium hubble enable --ui