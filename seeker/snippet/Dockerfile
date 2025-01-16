#date: 2025-01-16T16:38:27Z
#url: https://api.github.com/gists/57f10a3c0e41bf3c1a9b1936602e8672
#owner: https://api.github.com/users/sub314xxl

# tevun/quasar
FROM  node:alpine

RUN apk --update add openssh-client git

RUN npm install --global \
    @vue/cli@4.5.4 && \
    @vue/cli-init@4.5.4 \
    @quasar/cli@1.1.0\
    http-server@0.11.1

RUN mkdir /home/node/app

WORKDIR /home/node/app

CMD /bin/sh