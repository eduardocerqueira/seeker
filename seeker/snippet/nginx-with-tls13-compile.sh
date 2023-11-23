#date: 2023-11-23T16:39:42Z
#url: https://api.github.com/gists/de450129dafe52001f834b2686e495d1
#owner: https://api.github.com/users/osdeibi

#!/bin/bash

## nginx
NGINX=nginx-1.19.4.tar.gz

if [ ! -f "${NGINX}" ];then
    wget https://nginx.org/download/${NGINX}
fi

ND=$(basename $NGINX .tar.gz)
if [ ! -d "${ND}" ];then
    tar zxvf ${NGINX}
fi

cd ${ND}

## pre require package
## yum install gcc pcre-devel zlib-devel

./configure --prefix=/etc/nginx \
    --sbin-path=/usr/sbin/nginx \
    --modules-path=/usr/lib64/nginx/modules  \
    --conf-path=/etc/nginx/nginx.conf \
    --error-log-path=/var/log/nginx/error.log \
    --http-log-path=/var/log/nginx/access.log \
    --pid-path=/var/run/nginx.pid \
    --lock-path=/var/run/nginx.lock \
    --http-client-body-temp-path=/var/cache/nginx/client_temp \
    --http-proxy-temp-path=/var/cache/nginx/proxy_temp \
    --http-fastcgi-temp-path=/var/cache/nginx/fastcgi_temp \
    --http-uwsgi-temp-path=/var/cache/nginx/uwsgi_temp \
    --http-scgi-temp-path=/var/cache/nginx/scgi_temp \
    --user=nginx \
    --group=nginx \
    --with-compat \
    --with-file-aio \
    --with-threads \
    --with-http_addition_module \
    --with-http_auth_request_module \
    --with-http_dav_module \
    --with-http_flv_module \
    --with-http_gunzip_module \
    --with-http_gzip_static_module \
    --with-http_mp4_module \
    --with-http_random_index_module \
    --with-http_realip_module \
    --with-http_secure_link_module \
    --with-http_slice_module \
    --with-http_ssl_module \
    --with-http_stub_status_module \
    --with-http_sub_module \
    --with-http_v2_module \
    --with-mail \
    --with-mail_ssl_module \
    --with-stream \
    --with-stream_realip_module \
    --with-stream_ssl_module \
    --with-stream_ssl_preread_module \
    --with-openssl=../$(basename $OPENSSL .tar.gz)
    
make

sudo make install

nginx -V
