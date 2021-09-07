#date: 2021-09-07T17:15:50Z
#url: https://api.github.com/gists/3bfe39589fbe4dafd4f79fb956d59ff3
#owner: https://api.github.com/users/doytsujin

# Nginx can serve FLV/MP4 files by pseudo-streaming way without any specific media-server software. 
# To do the custom build we use 2 modules: --with-http_secure_link_module --with-http_flv_module
# This module "secure-link" helps you to protect links from stealing away.
#
# NOTE: see more details at coderwall: http://coderwall.com/p/3hksyg

cd /usr/src
wget http://nginx.org/download/nginx-1.5.13.tar.gz
tar xzvf ./nginx-1.5.13.tar.gz && rm -f ./nginx-1.5.13.tar.gz

wget ftp://ftp.csx.cam.ac.uk/pub/software/programming/pcre/pcre-8.32.tar.gz
tar xzvf pcre-8.32.tar.gz && rm -f ./pcre-8.32.tar.gz

wget http://www.openssl.org/source/openssl-1.0.1g.tar.gz
tar xzvf openssl-1.0.1g.tar.gz && rm -f openssl-1.0.1g.tar.gz

cd nginx-1.5.13 && ./configure --prefix=/opt/nginx --with-pcre=/usr/src/pcre-8.32 --with-openssl-opt=no-krb5 --with-openssl=/usr/src/openssl-1.0.1g --with-http_ssl_module --without-mail_pop3_module --without-mail_smtp_module --without-mail_imap_module --with-http_stub_status_module --with-http_secure_link_module --with-http_flv_module --with-http_mp4_module
make && make install
