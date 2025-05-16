#date: 2025-05-16T16:57:50Z
#url: https://api.github.com/gists/873a0d43e0f9a3c9a8e2ca8568d3cab3
#owner: https://api.github.com/users/Project0ne

#!/bin/bash

# 检查 root 权限
if [ "$(id -u)" != "0" ]; then
    echo "错误：请使用 root 用户运行此脚本！" 1>&2
    exit 1
fi

# 定义变量
BAGISTO_DIR="/var/www/bagisto"
DB_NAME="bagisto"
DB_USER="bagisto_user"
DB_PASS=$(openssl rand -base64 12)
APP_URL="http://$(curl -s ifconfig.me)"

# 安装基础工具
echo "🔄 更新系统并安装工具..."
apt update
apt install -y curl wget git unzip

# 安装 PHP 8.2（通过 ondrej 仓库）
echo "🔄 安装 PHP 8.2..."
apt install -y software-properties-common
add-apt-repository -y ppa:ondrej/php
apt update
apt install -y php8.2 php8.2-{bcmath,ctype,curl,dom,fileinfo,gd,json,mbstring,openssl,pdo_mysql,tokenizer,xml,fpm}

# 安装 MariaDB
echo "🔄 安装 MariaDB..."
apt install -y mariadb-server
systemctl start mariadb
systemctl enable mariadb

# 配置数据库
echo "🔄 配置数据库..."
mysql -e "CREATE DATABASE ${DB_NAME};"
mysql -e "CREATE USER '${DB_USER}'@'localhost' IDENTIFIED BY '${DB_PASS}';"
mysql -e "GRANT ALL PRIVILEGES ON ${DB_NAME}.* TO '${DB_USER}'@'localhost';"
mysql -e "FLUSH PRIVILEGES;"

# 安装 Composer
echo "🔄 安装 Composer..."
curl -sS https://getcomposer.org/installer | php -- --install-dir=/usr/local/bin --filename=composer
chmod +x /usr/local/bin/composer

# 安装 Node.js 18.x
echo "🔄 安装 Node.js 18.x..."
curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
apt install -y nodejs

# 下载 Bagisto
echo "🔄 下载 Bagisto..."
mkdir -p ${BAGISTO_DIR}
git clone https://github.com/bagisto/bagisto.git ${BAGISTO_DIR}
chown -R www-data:www-data ${BAGISTO_DIR}
chmod -R 775 ${BAGISTO_DIR}/storage ${BAGISTO_DIR}/bootstrap/cache

# 安装依赖
echo "🔄 安装 Composer 和 NPM 依赖..."
cd ${BAGISTO_DIR}
composer install --optimize-autoloader --no-dev
npm install
npm run prod

# 配置 .env 文件
echo "🔄 配置 .env 文件..."
cp .env.example .env
sed -i "s/APP_URL=.*/APP_URL=${APP_URL}/" .env
sed -i "s/DB_DATABASE=.*/DB_DATABASE=${DB_NAME}/" .env
sed -i "s/DB_USERNAME=.*/DB_USERNAME=${DB_USER}/" .env
sed -i "s/DB_PASSWORD= "**********"=${DB_PASS}/" .env

# 生成密钥和初始化数据库
echo "🔄 初始化数据库..."
php artisan key:generate
php artisan migrate --seed
php artisan storage:link
php artisan vendor:publish --all

# 配置 Nginx
echo "🔄 配置 Nginx..."
apt install -y nginx
systemctl start nginx
systemctl enable nginx

cat > /etc/nginx/sites-available/bagisto.conf <<EOL
server {
    listen 80;
    server_name _;
    root ${BAGISTO_DIR}/public;

    index index.php index.html;

    location / {
        try_files \$uri \$uri/ /index.php?\$query_string;
    }

    location ~ \.php\$ {
        include fastcgi_params;
        fastcgi_pass unix:/run/php/php8.2-fpm.sock;
        fastcgi_index index.php;
        fastcgi_param SCRIPT_FILENAME \$document_root\$fastcgi_script_name;
    }

    location ~ /\.ht {
        deny all;
    }
}
EOL

ln -s /etc/nginx/sites-available/bagisto.conf /etc/nginx/sites-enabled/
nginx -t && systemctl restart nginx

# 完成
echo "✅ Bagisto 安装完成！"
echo "🔗 访问地址: ${APP_URL}"
echo "🔑 管理员后台: ${APP_URL}/admin"
echo "📧 默认账号: admin@example.com"
echo "🔐 默认密码: admin123"
echo "📌 数据库信息:"
echo "   - 数据库名: ${DB_NAME}"
echo "   - 用户名: ${DB_USER}"
echo "   - 密码: ${DB_PASS}""