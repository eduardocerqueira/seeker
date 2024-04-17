#date: 2024-04-17T16:47:01Z
#url: https://api.github.com/gists/470adc15a1091246291e29565bab995d
#owner: https://api.github.com/users/leonardochaia

# plugin-image/Dockerfile
FROM alpine:3.19

RUN apk update && apk add --no-cache \
    --repository=http://dl-cdn.alpinelinux.org/alpine/edge/testing \
    kustomize podman

RUN ln -s /usr/bin/podman /usr/bin/docker \
      && adduser -h /home/argocd -s /bin/sh -G ping -u 999 -D argocd \
      && echo 'argocd:165536:65536' >> /etc/subuid \
      && echo 'argocd:165536:65536' >> /etc/subgid

COPY storage.conf /home/argocd/.config/containers/storage.conf
COPY plugin.yaml /home/argocd/cmp-server/config/plugin.yaml