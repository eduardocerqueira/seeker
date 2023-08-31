#date: 2023-08-31T17:09:50Z
#url: https://api.github.com/gists/085c2873671b3e30bd59c58bb1d636a9
#owner: https://api.github.com/users/atefhaloui

#
# This image is based on the work Simon Belmas:
#  https://github.com/simbelmas/dockerfiles/blob/latest/fedora-minimal/Dockerfile
#  https://github.com/simbelmas/dockerfiles/blob/latest/buildah-image-builder/Dockerfile
#  https://github.com/simbelmas/dockerfiles/blob/latest/kube-cmdline/Dockerfile
#

FROM registry.fedoraproject.org/fedora-minimal:40

ARG KUBECTL_VERSION=v1.28.1
ARG HELM_VERSION=v3.12.3
ARG ARGOCD_VERSION=v2.8.2
ARG VAULT_VERSION=1.14.2

# Install buildah + skopeo + jq + git
RUN dnf5 --no-docs -y install shadow-utils buildah skopeo fuse-overlayfs cpp jq git unzip curl --exclude container-selinux,qemu-user-static \
    && rpm --setcaps shadow-utils 2>/dev/null \
    && dnf5 clean all

ENV BUILDAH_ISOLATION=chroot

RUN sed -e 's|^#mount_program|mount_program|g' -e '/additionalimage.*/a "/var/lib/shared",' -e 's|^mountopt[[:space:]]*=.*$|mountopt = "nodev,fsync=0"|g' /usr/share/containers/storage.conf > /etc/containers/storage.conf \
    && chmod 644 /etc/containers/storage.conf \
    && mkdir -p /var/lib/shared/overlay-images /var/lib/shared/overlay-layers /var/lib/shared/vfs-images /var/lib/shared/vfs-layers \
    && touch /var/lib/shared/overlay-images/images.lock \
    && touch /var/lib/shared/overlay-layers/layers.lock \
    && touch /var/lib/shared/vfs-images/images.lock \
    && touch /var/lib/shared/vfs-layers/layers.lock \
    && useradd -u 1001 build \
    && echo -e "build:1:1000\nbuild:1002:64535" > /etc/subuid  \
    && cp /etc/subuid /etc/subgid \
    && mkdir -p /home/build/.local/share/containers \
    && chown -R build:build /home/build

# Install yq
RUN cd /usr/local/bin \
    && curl -o yq -L https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 \
    && chmod a+x yq

# Install kubectl
RUN cd /usr/local/bin \
    && curl -o kubectl -L "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl" \
    && chmod a+x kubectl

# Install helm
RUN cd /usr/local/bin \
    && curl -o helm-${HELM_VERSION}-linux-amd64.tar.gz -L https://get.helm.sh/helm-${HELM_VERSION}-linux-amd64.tar.gz \
    && tar xf helm-${HELM_VERSION}-linux-amd64.tar.gz \
    && mv linux-amd64/helm . \
    && chmod a+x helm \
    && rm -rf helm-${HELM_VERSION}-linux-amd64.tar.gz linux-amd64

# Install argocd
RUN cd /usr/local/bin \
    && curl -o argocd -L https://github.com/argoproj/argo-cd/releases/download/${ARGOCD_VERSION}/argocd-linux-amd64 \
    && chmod a+x argocd

# Install hashicop vault
RUN cd /usr/local/bin \
	&& curl -o vault_${VAULT_VERSION}_linux_amd64.zip -L https://releases.hashicorp.com/vault/${VAULT_VERSION}/vault_${VAULT_VERSION}_linux_amd64.zip \
	&& unzip vault_${VAULT_VERSION}_linux_amd64.zip \
	&& rm vault_${VAULT_VERSION}_linux_amd64.zip \
	&& chmod a+x vault

USER 1001

WORKDIR /home/build

ENTRYPOINT ["/bin/sh"]

