#date: 2023-04-05T16:45:44Z
#url: https://api.github.com/gists/3232ae0b0e0f218a24530201a4a0f863
#owner: https://api.github.com/users/jniltinho

#!/bin/bash

## https://www.linuxfromscratch.org/blfs/view/svn/server/postfix.html
## https://www.linuxfromscratch.org/blfs/view/svn/introduction/bootscripts.html
## https://www.interserver.net/tips/kb/install-postfix-source/


apt update
apt install -y build-essential wget 
apt install -y libdb-dev m4 libmysqlclient-dev libicu-dev libnsl-dev libsasl2-dev libssl-dev
apt install -y libpcre3-dev pkg-config libcdb-dev libsqlite3-dev libpq-dev

groupadd -g 32 postfix && groupadd -g 102 postdrop
useradd -c "Postfix Daemon User" -d /var/spool/postfix -g postfix -s /bin/false -u 32 postfix
#chown -v postfix:postfix /var/mail

wget http://linorg.usp.br/postfix/release/official/postfix-3.7.4.tar.gz
tar xf postfix-*.tar.gz
cd postfix-3.7.4

make makefiles CCARGS="-DNO_NIS -DUSE_TLS -I/usr/include/openssl/ \
-DUSE_SASL_AUTH -DUSE_CYRUS_SASL -I/usr/include/sasl \
-DHAS_MYSQL -I/usr/include/mysql" \
AUXLIBS="-lssl -lcrypto -lsasl2 -lmysqlclient -lz -lm"

make

sh postfix-install -non-interactive daemon_directory=/usr/lib/postfix \
   manpage_directory=/usr/share/man html_directory=/usr/share/doc/postfix-3.7.4/html \
   readme_directory=/usr/share/doc/postfix-3.7.4/readme