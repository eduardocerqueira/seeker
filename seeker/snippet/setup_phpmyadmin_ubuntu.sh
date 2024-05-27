#date: 2024-05-27T16:59:21Z
#url: https://api.github.com/gists/bf8b2d1e56ebed5ff4d62589e063a58f
#owner: https://api.github.com/users/Arsybai

#!/bin/bash

# Exit on any error
set -e

# Prompt user for input
echo "[ Create New MySQL user ]"
read -p "Username: " DB_USERNAME
read -sp "Password: "**********"
echo
read -p "Enter Domain for phpmyadmin: " DB_DOMAIN

# Update package list
sudo apt update

# Install MySQL Server
sudo apt install -y mysql-server

# Secure MySQL Installation (optional but recommended)
echo "Securing MySQL installation..."
sudo mysql_secure_installation

# MySQL setup
echo "Setting up MySQL user..."
sudo mysql -e "CREATE USER '${DB_USERNAME}'@'localhost' IDENTIFIED BY '${DB_PASSWORD}';"
sudo mysql -e "GRANT ALL PRIVILEGES ON *.* TO '${DB_USERNAME}'@'localhost';"
sudo mysql -e "FLUSH PRIVILEGES;"

# Install Nginx
echo "Installing Nginx..."
sudo apt install -y nginx

# Allow Nginx through firewall
echo "Configuring firewall for Nginx..."
sudo ufw allow 'Nginx Full'

# Install Certbot
echo "Installing Certbot..."
sudo apt install -y certbot python3-certbot-nginx

# Install PHP and phpMyAdmin
echo "Installing PHP and phpMyAdmin..."
sudo apt install -y php php-mbstring php-zip php-gd php-json php-curl php-fpm
sudo apt install -y phpmyadmin

# Configure phpMyAdmin to be served from DB_DOMAIN
echo "Configuring phpMyAdmin..."
sudo ln -s /usr/share/phpmyadmin /var/www/html/phpmyadmin

# Nginx configuration for phpMyAdmin
sudo tee /etc/nginx/sites-available/phpmyadmin <<EOL
server {
    listen 80;
    server_name ${DB_DOMAIN};

    root /usr/share/phpmyadmin;
    index index.php index.html index.htm;

    location / {
        try_files \$uri \$uri/ =404;
    }

    location ~ \.php\$ {
        include snippets/fastcgi-php.conf;
        fastcgi_pass unix:/var/run/php/php-fpm.sock;
    }
}
EOL

# Enable the configuration
sudo ln -s /etc/nginx/sites-available/phpmyadmin /etc/nginx/sites-enabled/

# Test Nginx configuration
sudo nginx -t

# Reload Nginx to apply new configuration
sudo systemctl reload nginx

# Obtain SSL certificate for phpMyAdmin
echo "Obtaining SSL certificate..."
sudo certbot --nginx -d ${DB_DOMAIN}

echo "Setup completed successfully."

