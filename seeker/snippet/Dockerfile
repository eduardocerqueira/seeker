#date: 2021-09-10T16:58:59Z
#url: https://api.github.com/gists/959abb619021d6fd4950821d162ec5a8
#owner: https://api.github.com/users/lcrilly

FROM nginx AS build

WORKDIR /src
RUN apt-get update && \
    apt-get install -y git gcc make mercurial libperl-dev libpcre3-dev zlib1g-dev libxslt1-dev libgd-ocaml-dev libgeoip-dev
RUN git clone -b openssl-3.0 https://github.com/openssl/openssl openssl-3.0 && \
    hg clone https://hg.nginx.org/nginx && \
    hg clone http://hg.nginx.org/njs
RUN cd nginx && \
    auto/configure `nginx -V 2>&1 | sed "s/ \-\-/ \\\ \n\t--/g" | grep \-\- | grep -ve opt= -e param=` \
                   --with-openssl=../openssl-3.0 --with-debug --add-module=../njs/nginx && \
    make

FROM nginx
COPY --from=build /src/nginx/objs/nginx /usr/sbin
