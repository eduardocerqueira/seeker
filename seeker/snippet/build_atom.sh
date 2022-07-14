#date: 2022-07-14T16:53:09Z
#url: https://api.github.com/gists/9222d443e301021571aa9840f011a0bb
#owner: https://api.github.com/users/Jimadine

#!/bin/bash
ROOT_MYSQL_PWD=changeme
ATOM_MYSQL_USER=atom
ATOM_MYSQL_DB=atom
ATOM_MYSQL_PWD=12345
ATOM_ADMIN_EMAIL=boaty_mcboatface@domain.org
ATOM_ADMIN_USERNAME=boaty_mcboatface
ATOM_ADMIN_PWD=changeme

apt install -y software-properties-common apt-transport-https
add-apt-repository -y ppa:ondrej/php
curl -L -s https://artifacts.elastic.co/GPG-KEY-elasticsearch | gpg --dearmor > /usr/share/keyrings/elasticsearch-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/elasticsearch-archive-keyring.gpg] https://artifacts.elastic.co/packages/5.x/apt stable main" > /etc/apt/sources.list.d/elastic-5.x.list
apt update
apt install -y debconf-utils
debconf-set-selections <<< "mysql-server-8.0        mysql-server/root_password      password $ROOT_MYSQL_PWD"
debconf-set-selections <<< "mysql-server-8.0        mysql-server/root_password_again        password $ROOT_MYSQL_PWD"
DEBIAN_FRONTEND=noninteractive apt install -y mysql-server

cat <<'MYSQLDCNF' > /etc/mysql/conf.d/mysqld.cnf
[mysqld]
sql_mode=ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION
optimizer_switch='block_nested_loop=off'
MYSQLDCNF

systemctl restart mysql
apt install -y openjdk-8-jre-headless \
elasticsearch \
nginx \
php-common \
php7.4-apcu \
php7.4-apcu-bc \
php7.4-common \
php7.4-cli \
php7.4-curl \
php7.4-fpm \
php7.4-json \
php7.4-ldap \
php7.4-memcache \
php7.4-mbstring \
php7.4-mysql \
php7.4-opcache \
php7.4-readline \
php7.4-xml \
php7.4-xsl \
php7.4-zip \
gearman-job-server \
imagemagick \
ghostscript \
poppler-utils \
ffmpeg \
git \
make \
npm

update-alternatives --set php /usr/bin/php7.4
apt install -y --no-install-recommends fop libsaxon-java
systemctl enable elasticsearch
systemctl start elasticsearch
touch /etc/nginx/sites-available/atom
ln -sf /etc/nginx/sites-available/atom /etc/nginx/sites-enabled/atom
rm /etc/nginx/sites-enabled/default

cat <<'ATOMNGINX' > /etc/nginx/sites-available/atom
upstream atom {
  server unix:/run/php7.4-fpm.atom.sock;
}

server {

  listen 80;
  root /usr/share/nginx/atom;

  # http://wiki.nginx.org/HttpCoreModule#server_name
  # _ means catch any, but it's better if you replace this with your server
  # name, e.g. archives.foobar.com
  server_name _;

  client_max_body_size 72M;

  # http://wiki.nginx.org/HttpCoreModule#try_files
  location / {
    try_files $uri /index.php?$args;
  }

  location ~ /\. {
    deny all;
    return 404;
  }

  location ~* (\.yml|\.ini|\.tmpl)$ {
    deny all;
    return 404;
  }

  location ~* /(?:uploads|files)/.*\.php$ {
    deny all;
    return 404;
  }

  location ~* /uploads/r/(.*)/conf/ {

  }

  location ~* ^/uploads/r/(.*)$ {
    include /etc/nginx/fastcgi_params;
    set $index /index.php;
    fastcgi_param SCRIPT_FILENAME $document_root$index;
    fastcgi_param SCRIPT_NAME $index;
    fastcgi_pass atom;
  }

  location ~ ^/private/(.*)$ {
    internal;
    alias /usr/share/nginx/atom/$1;
  }

  location ~ ^/(index|qubit_dev)\.php(/|$) {
    include /etc/nginx/fastcgi_params;
    fastcgi_param SCRIPT_FILENAME $document_root$fastcgi_script_name;
    fastcgi_split_path_info ^(.+\.php)(/.*)$;
    fastcgi_pass atom;
  }

  location ~* \.php$ {
    deny all;
    return 404;
  }

}
ATOMNGINX

systemctl enable nginx
systemctl reload nginx

cat <<'PHPFPM' > /etc/php/7.4/fpm/pool.d/atom.conf
[atom]

; The user running the application
user = www-data
group = www-data

; Use UNIX sockets if Nginx and PHP-FPM are running in the same machine
listen = /run/php7.4-fpm.atom.sock
listen.owner = www-data
listen.group = www-data
listen.mode = 0600

; The following directives should be tweaked based in your hardware resources
pm = dynamic
pm.max_children = 30
pm.start_servers = 10
pm.min_spare_servers = 10
pm.max_spare_servers = 10
pm.max_requests = 200

chdir = /

; Some defaults for your PHP production environment
; A full list here: http://www.php.net/manual/en/ini.list.php
php_admin_value[expose_php] = off
php_admin_value[allow_url_fopen] = on
php_admin_value[memory_limit] = 512M
php_admin_value[max_execution_time] = 120
php_admin_value[post_max_size] = 72M
php_admin_value[upload_max_filesize] = 64M
php_admin_value[max_file_uploads] = 10
php_admin_value[cgi.fix_pathinfo] = 0
php_admin_value[display_errors] = off
php_admin_value[display_startup_errors] = off
php_admin_value[html_errors] = off
php_admin_value[session.use_only_cookies] = 0

; APC
php_admin_value[apc.enabled] = 1
php_admin_value[apc.shm_size] = 64M
php_admin_value[apc.num_files_hint] = 5000
php_admin_value[apc.stat] = 0

; Zend OPcache
php_admin_value[opcache.enable] = 1
php_admin_value[opcache.memory_consumption] = 192
php_admin_value[opcache.interned_strings_buffer] = 16
php_admin_value[opcache.max_accelerated_files] = 4000
php_admin_value[opcache.validate_timestamps] = 0
php_admin_value[opcache.fast_shutdown] = 1

; This is a good place to define some environment variables, e.g. use
; ATOM_DEBUG_IP to define a list of IP addresses with full access to the
; debug frontend or ATOM_READ_ONLY if you want AtoM to prevent
; authenticated users
env[ATOM_DEBUG_IP] = "10.10.10.10,127.0.0.1"
env[ATOM_READ_ONLY] = "off"
PHPFPM

systemctl enable php7.4-fpm
systemctl start php7.4-fpm
php-fpm7.4 --test
systemctl status php7.4-fpm
systemctl status nginx
rm /etc/php/7.4/fpm/pool.d/www.conf
systemctl restart php7.4-fpm

cat <<'GEARMAN' > /usr/lib/systemd/system/atom-worker.service
[Unit]
Description=AtoM worker
After=network.target
# High interval and low restart limit to increase the possibility
# of hitting the rate limits in long running recurrent jobs.
StartLimitIntervalSec=24h
StartLimitBurst=3

[Install]
WantedBy=multi-user.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/usr/share/nginx/atom
ExecStart=/usr/bin/php7.4 -d memory_limit=-1 -d error_reporting="E_ALL" symfony jobs:worker
KillSignal=SIGTERM
Restart=on-failure
RestartSec=30
GEARMAN

systemctl daemon-reload
systemctl enable atom-worker
systemctl start atom-worker

mkdir -p /usr/share/nginx/atom
git clone -b qa/2.x --depth 1 http://github.com/artefactual/atom.git /usr/share/nginx/atom
chown -R www-data:www-data /usr/share/nginx/atom
chmod o= /usr/share/nginx/atom
cd /usr/share/nginx/atom
php -r "copy('https://getcomposer.org/installer', 'composer-setup.php');"
php -r "if (hash_file('sha384', 'composer-setup.php') === '55ce33d7678c5a611085589f1f3ddf8b3c52d662cd01d4ba75c0ee0459970c2200a51f492d557530c71c15d8dba01eae') { echo 'Installer verified'; } else { echo 'Installer corrupt'; unlink('composer-setup.php'); } echo PHP_EOL;"
php composer-setup.php
php -r "unlink('composer-setup.php');"
php composer.phar install --no-dev
npm install -g "less@<4.0.0"
make -C /usr/share/nginx/atom/plugins/arDominionPlugin
make -C /usr/share/nginx/atom/plugins/arArchivesCanadaPlugin
mysql -h localhost -u root -p$ROOT_MYSQL_PWD -e "CREATE DATABASE ${ATOM_MYSQL_DB} CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;"
mysql -h localhost -u root -p$ROOT_MYSQL_PWD -e "CREATE USER '${ATOM_MYSQL_USER}'@'localhost' IDENTIFIED BY '${ATOM_MYSQL_PWD}';"
mysql -h localhost -u root -p$ROOT_MYSQL_PWD -e "GRANT ALL PRIVILEGES ON ${ATOM_MYSQL_USER}.* TO '${ATOM_MYSQL_USER}'@'localhost';"
# Use --demo option?
php symfony tools:install --database-host="localhost" --database-port="3306" --database-name="${ATOM_MYSQL_DB}" --database-user="${ATOM_MYSQL_USER}" --database-password="${ATOM_MYSQL_PWD}" --search-host="localhost" --search-port="9200" --search-index="atom" --site-title="AtoM" --site-description="Access to Memory" --site-base-url="http://127.0.0.1" --admin-email="${ATOM_ADMIN_EMAIL}" --admin-username="${ATOM_ADMIN_USERNAME}" --admin-password="${ATOM_ADMIN_PWD}" --no-confirmation
