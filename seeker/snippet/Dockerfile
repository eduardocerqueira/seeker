#date: 2025-01-13T16:59:31Z
#url: https://api.github.com/gists/96f5ba10caef95d7879c2e75dfcd4f6b
#owner: https://api.github.com/users/LorenzoLeonardini

# adapted from https://richardcampen.com/blog/nginx-mod-zip-for-dynamic-file-zipping

ARG version='1.27.3'
ARG mod_zip_version='1.3.0'

FROM nginx:${version} AS build

ARG version
ARG mod_zip_version

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    zlib1g-dev \
    libpcre3-dev \
    wget \
    unzip

RUN echo version: ${version}

RUN wget -O nginx.zip https://github.com/nginx/nginx/archive/refs/tags/release-${version}.zip && unzip nginx.zip

RUN wget -O mod_zip.zip https://github.com/evanmiller/mod_zip/archive/refs/tags/${mod_zip_version}.zip && unzip mod_zip.zip

WORKDIR /nginx-release-${version}

RUN ./auto/configure --with-compat --add-dynamic-module=../mod_zip-${mod_zip_version}/
RUN make -f ./objs/Makefile modules


FROM nginx:${version}

ARG version
ARG mod_zip_version

# setup mod_zip plugin

COPY --from=build /nginx-release-${version}/objs/ngx_http_zip_module.so /usr/lib/nginx/modules/

RUN chmod 644 /usr/lib/nginx/modules/ngx_http_zip_module.so
RUN sed -i '1 s/^/load_module modules\/ngx_http_zip_module.so;\n/' /etc/nginx/nginx.conf