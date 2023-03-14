#date: 2023-03-14T17:13:56Z
#url: https://api.github.com/gists/6d5aa44580a7cdfab28336dac49307ee
#owner: https://api.github.com/users/andreabreu76

export DEBIAN_FRONTEND=noninteractive

sudo apt install -y software-properties-common curl build-essential libssl-dev php-pear libssl-dev git wget unzip

sudo add-apt-repository ppa:ondrej/php

sudo apt install -y apache2 libapache2-mod-fcgid php7.4-fpm php7.4-common php7.4-mysql php7.4-gd php7.4-json php7.4-cli php7.4-curl php7.4-mbstring php7.4-xml php7.4-zip php7.4-fpm php7.4-gd php7.4-zip php7.4-bz2 php7.4-mysql php7.4-mysqli php7.4-soap php7.4-xmlrpc php7.4-xsl php7.4-xdebug php7.4-opcache php7.4-igbinary php7.4-memcached php7.4-bcmath php7.4-mbstring php7.4-curl php7.4-intl php7.4-xml php7.4-cli php7.4-imap php7.4-common php7.4-odbc php7.4-mcrypt php7.4-ldap 

sudo apt clean

sudo curl -sS https://getcomposer.org/installer | sudo php -- --install-dir=/usr/local/bin --filename=composer

sudo curl -sL https://deb.nodesource.com/setup_14.x | sudo bash - \
&& sudo apt install -y nodejs \
&& sudo npm install -g yarn \
&& sudo npm install -g --unsafe-perm @vue/cli \
&& sudo npm install -g --unsafe-perm node-sass \
&& sudo npm install -g --unsafe-perm @quasar/cli

sudo mkdir /opt/oracle && cd /opt/oracle \
&& sudo wget https://github.com/diogomascarenha/oracle-instantclient/raw/master/instantclient-basic-linux.x64-12.1.0.2.0.zip \
&& sudo wget https://github.com/diogomascarenha/oracle-instantclient/raw/master/instantclient-sdk-linux.x64-12.1.0.2.0.zip\
&& sudo unzip /opt/oracle/instantclient-basic-linux.x64-12.1.0.2.0.zip -d /opt/oracle \
&& sudo unzip /opt/oracle/instantclient-sdk-linux.x64-12.1.0.2.0.zip -d /opt/oracle \
&& sudo ln -s /opt/oracle/instantclient_12_1/libclntsh.so.12.1 /opt/oracle/instantclient_12_1/libclntsh.so \
&& sudo ln -s /opt/oracle/instantclient_12_1/libclntshcore.so.12.1 /opt/oracle/instantclient_12_1/libclntshcore.so \
&& sudo ln -s /opt/oracle/instantclient_12_1/libocci.so.12.1 /opt/oracle/instantclient_12_1/libocci.so \
&& sudo rm -rf /opt/oracle/*.zip

sudo apt update \
&& apt install --no-install-recommends -y libaio-dev \
&& echo 'instantclient,/opt/oracle/instantclient_12_1/' | pecl install oci8-2.2.0; \
&& echo "extension=oci8.so" >> /usr/local/etc/php/conf.d/oci8-php-ext-xdebug.ini \
&& php -m | grep -q 'oci8' \
