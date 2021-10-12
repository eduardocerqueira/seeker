#date: 2021-10-12T17:12:52Z
#url: https://api.github.com/gists/2ac76eb742abe9f85487fc4d2bddf80c
#owner: https://api.github.com/users/davemoz

#!/bin/bash

##############################################################
# Set Your System and Wordpress Config Preferences
##############################################################

export SYSTEM_USER=username          # User PHP-FPM runs under

# Database 
export WP_DB_NAME=wordpress
export WP_DB_USER=wordpress
export WP_DB_PASS=strongpassword
export WP_DB_PREFIX=em_               # Only numbers, letters, and underscores please!

# Site info
export SITE_URL=www.domain.com
export SITE_TITLE="My Great Wordpress Site"

# Wordpress Login Info (please don't use 'admin' here)
export ADMIN_USERNAME=username
export ADMIN_PASSWORD=verystrongpassword
export ADMIN_EMAIL=username@domain.com

##########################
# Start the setup and install
##########################
# Run system updates
yum update -y

# Install NGINX 1.12
amazon-linux-extras install nginx1.12

# Install PHP 7.2
amazon-linux-extras install php7.2


# Install MariaDB (MySQL replacement)
yum install -y mariadb-server mariadb


# I like to setup my websites under /var/www/vhosts/domain.com/httpdocs It's how I roll

mkdir -p /var/www/vhosts/$SITE_URL/httpdocs

# Create a user for the website to run under (I don't like running as a guessable user)
useradd -d /var/www/vhosts/$SITE_URL/  $SYSTEM_USER

# Set permissions
chown -R $SYSTEM_USER:$SYSTEM_USER  /var/www/vhosts/$SITE_URL

# Configure PHP-FPM instance to run as the user created (replace <USERNAME> with the user you used in the last step)
cat << EOF > /etc/php-fpm.d/www.conf 
[www]
user = $SYSTEM_USER
group = $SYSTEM_USER
listen = /run/php-fpm/www.sock
listen.acl_users = apache,nginx
listen.allowed_clients = 127.0.0.1
pm = dynamic
pm.max_children = 50
pm.start_servers = 5
pm.min_spare_servers = 5
pm.max_spare_servers = 35
slowlog = /var/log/php-fpm/www-slow.log
php_admin_value[error_log] = /var/log/php-fpm/www-error.log
php_admin_flag[log_errors] = on
php_value[session.save_handler] = files
php_value[session.save_path]    = /var/lib/php/session
php_value[soap.wsdl_cache_dir]  = /var/lib/php/wsdlcache
EOF

# Configure NGINX
cat << EOF > /etc/nginx/nginx.conf
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log;
pid /run/nginx.pid;

# Load dynamic modules. See /usr/share/nginx/README.dynamic.
include /usr/share/nginx/modules/*.conf;

events {
    worker_connections 1024;
}

http {
    log_format  main  '\$remote_addr - \$remote_user [\$time_local] "\$request" '
                      '\$status \$body_bytes_sent "\$http_referer" '
                      '"\$http_user_agent" "\$http_x_forwarded_for"';

    access_log  /var/log/nginx/access.log  main;

    sendfile            on;
    tcp_nopush          on;
    tcp_nodelay         on;
    keepalive_timeout   65;
    types_hash_max_size 2048;

    include             /etc/nginx/mime.types;
    default_type        application/octet-stream;

    # Load modular configuration files from the /etc/nginx/conf.d directory.
    # See http://nginx.org/en/docs/ngx_core_module.html#include
    # for more information.
    include /etc/nginx/conf.d/*.conf;
}
EOF

# Configure NGINX VirtualHost
cat << EOF > /etc/nginx/conf.d/$SITE_URL.conf
server {
    listen       80 ;
    server_name  $SITE_URL;
    server_name _;
    # note that these lines are originally from the "location /" block
    root   /var/www/vhosts/$SITE_URL/httpdocs;
    index index.php index.html index.htm;
    location = /favicon.ico {
            log_not_found off;
            access_log off;
    }
    location = /robots.txt {
            allow all;
            log_not_found off;
            access_log off;
    }
    location / {
            # This is cool because no php is touched for static content.
            # include the "?" part so non-default permalinks doesn't break when using query string
		try_files \$uri \$uri/ /index.php?\$args;
    }
    location ~* \.(js|css|png|jpg|jpeg|gif|ico)\$ {
            expires max;
            log_not_found off;
    }
    error_page 404 /404.html;
    error_page 500 502 503 504 /50x.html;

    location = /50x.html {
        root /usr/share/nginx/html;
    }
	location ~ \.(php|phar)(/.*)?\$ {
	    fastcgi_split_path_info ^(.+\.(?:php|phar))(/.*)\$; 
	    fastcgi_pass unix:/run/php-fpm/www.sock;
	    fastcgi_intercept_errors on;
	    fastcgi_index  index.php;
	    include        fastcgi_params;
	    fastcgi_param  SCRIPT_FILENAME  \$document_root\$fastcgi_script_name;
	    fastcgi_param  PATH_INFO \$fastcgi_path_info;
	}
}
EOF

# remove un-used files
rm -f /etc/nginx/conf.d/php-fpm.conf
rm -f /etc/nginx/default.d/php.conf


# Start services and set to start on boot

systemctl start mariadb
systemctl enable php-fpm
systemctl start nginx
systemctl enable nginx
systemctl start php-fpm
systemctl enable php-fpm

sleep 15

# Secure MariaDB with a Random Password and save it in /root/.my.cnf
# Also setup Wordpress DB

SQLROOTPASS=`< /dev/urandom tr -dc _A-Z-a-z-0-9 | head -c${1:-32};echo;`

mysql -u root <<-EOF

UPDATE mysql.user SET Password=PASSWORD('`echo $SQLROOTPASS`') WHERE User='root';
DELETE FROM mysql.user WHERE User='root' AND Host NOT IN ('localhost', '127.0.0.1', '::1');
DELETE FROM mysql.user WHERE User='';
DELETE FROM mysql.db WHERE Db='test' OR Db='test_%';
DROP DATABASE test;
CREATE DATABASE $WP_DB_NAME;
grant all on $WP_DB_NAME.* to $WP_DB_USER@'localhost' identified by '$WP_DB_PASS';
FLUSH PRIVILEGES;
EOF


cat << EOF > ~/.my.cnf 
[client] 
password=`echo $SQLROOTPASS` 
EOF

sleep 5


# Install WP-CLI (wp-cli.org)

curl -O https://raw.githubusercontent.com/wp-cli/builds/gh-pages/phar/wp-cli.phar
chmod +x wp-cli.phar 
mv wp-cli.phar /usr/local/bin/wp


# Install Wordpress as the SYSTEM_USER

su $SYSTEM_USER
cd /var/www/vhosts/$SITE_URL/httpdocs
wp core download
wp config create --dbname=$WP_DB_NAME --dbuser=$WP_DB_USER --dbpass=$WP_DB_PASS --dbprefix=$WP_DB_PREFIX
wp core install --url=$SITE_URL --title="$SITE_TITLE" --admin_user=$ADMIN_USERNAME --admin_password=$ADMIN_PASSWORD --admin_email=$ADMIN_EMAIL
