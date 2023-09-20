#date: 2023-09-20T16:58:52Z
#url: https://api.github.com/gists/ba409dcc4a24bb4a68e88935065d4868
#owner: https://api.github.com/users/char0n

FROM ubuntu:jammy

# use bash as default shell
SHELL ["/bin/bash", "-c"]

# install prerequisites
RUN apt-get update -y && apt-get install -y wget

# install Node.js using nvm
RUN mkdir -p /usr/local/nvm
ENV NVM_DIR /usr/local/nvm
ENV NODE_VERSION 20.3.0
RUN wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash
RUN source $NVM_DIR/nvm.sh \
    && nvm install $NODE_VERSION \
    && nvm alias default $NODE_VERSION \
    && nvm use default
# add node and npm to path so the commands are available
ENV NODE_PATH $NVM_DIR/v$NODE_VERSION/lib/node_modules
ENV PATH $NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH