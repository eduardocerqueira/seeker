#date: 2023-11-03T16:53:46Z
#url: https://api.github.com/gists/74083ad304004998e1e655a91470891d
#owner: https://api.github.com/users/TrogloGeek

# build me with: docker build --network host --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg HTTP_PROXY=$http_proxy docker/dev/ -t node-18
# then set an alias tu use me: alias node18c="docker run --rm -ti --mount type=bind,source=.,target=/mnt --mount type=bind,source=${HOME}/.gitconfig,target=/home/node/.gitconfig --network host node-18"
# then use me: node18c npm init
FROM node:18

USER root:root

ARG UID=1000
ARG GID=1000
ARG HTTP_PROXY=http://localhost:3128

# change UID and GID
RUN UID_FROM=$(id -u node) ;\
    if [ "$UID" != "$UID_FROM" ]; then \
        usermod -o --uid $UID node ;\
        find / -uid $UID_FROM -exec chown node '{}' \; || /bin/true ;\
    fi ;\
    GID_FROM=$(id -g node) ;\
    if [ "$GID" != "$GID_FROM" ]; then \
        groupmod --gid $GID node ;\
        find / -gid $GID_FROM -exec chgrp node '{}' \; || /bin/true ;\
    fi

USER node:node
WORKDIR /mnt

RUN npm config set proxy ${HTTP_PROXY} ;\
    npm config set https-proxy ${HTTP_PROXY} ;\
    npm config set registry https://artifactory.s.arkea.com/artifactory/api/npm/npm/ ;\
    npm config set prefix ${HOME}/.npm ;\
    npm install --global npm@latest ;\
    npm install --global npm-check-updates

ENV PATH=/home/node/.npm/bin:${PATH}

ENTRYPOINT []
