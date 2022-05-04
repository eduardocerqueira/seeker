#date: 2022-05-04T17:12:52Z
#url: https://api.github.com/gists/d1a0f154737fc0f5f14c33d2d874152d
#owner: https://api.github.com/users/brandond

FROM ubuntu:22.04
VOLUME /var/lib/rancher/rke2
VOLUME /var/lib/kubelet
VOLUME /var/lib/cni
VOLUME /var/log
ARG TAG=latest

# use rke2 bundled binaries
ENV PATH=/var/lib/rancher/rke2/bin:$PATH
# for kubectl
ENV KUBECONFIG=/etc/rancher/rke2/rke2.yaml
# for crictl
ENV CONTAINER_RUNTIME_ENDPOINT="unix:///run/k3s/containerd/containerd.sock"
# for ctr
RUN mkdir -p /run/containerd \
    &&  ln -s /run/k3s/containerd/containerd.sock /run/containerd/containerd.sock
# for go dns bug
RUN mkdir -p /etc && \
    echo 'hosts: files dns' > /etc/nsswitch.conf
# for conformance testing
RUN chmod 1777 /tmp
RUN set -x \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y update \
    && apt-get -y upgrade \
    && apt-get -y install \
    bash \
    bash-completion \
    curl \
    ca-certificates \
    conntrack \
    ebtables \
    ethtool \
    iptables \
    jq \
    less \
    socat \
    vim
RUN ["/bin/bash", "-c", "curl -fL https://github.com/rancher/rke2/releases/${TAG/-rke2/+rke2}/download/rke2.linux-amd64.tar.gz | tar -zxC /usr/local/bin" ]
ENTRYPOINT ["/usr/local/bin/rke2"]
CMD ["server"]