#date: 2024-04-03T17:04:42Z
#url: https://api.github.com/gists/6ebf0949388559305e0c5ff804a8d622
#owner: https://api.github.com/users/lazycapricoder

#!/bin/bash
# Prompt the user to choose a web server
echo "Which web server do you want to install?"
echo "1. Nginx"
echo "2. Apache"

read -p "Enter your choice [1 or 2]: " web_server_choice

# Handle user choice for web server
case $web_server_choice in
    1)
        # Add the ondrej/nginx-mainline PPA repository
        sudo add-apt-repository -y ppa:ondrej/nginx-mainline
        WEB_SERVER="nginx"
        ;;
    2)
        # No additional repository needed for Apache
        WEB_SERVER="apache"
        ;;
    *)
        echo "Invalid choice for web server. Exiting."
        exit 1
        ;;
esac

# Update package lists
sudo apt update

# Prompt the user to choose a PHP version
echo "Which PHP version do you want to install?"
echo "1. PHP 7.4"
echo "2. PHP 8.0"
echo "3. PHP 8.1"
echo "4. PHP 8.2"
echo "5. PHP 8.3"
echo "6. Other (specify version)"

read -p "Enter your choice [1-6]: " php_version_choice

# Handle user choice for PHP version
case $php_version_choice in
    1)
        PHP_VERSION="7.4"
        ;;
    2)
        PHP_VERSION="8.0"
        ;;
    3)
        PHP_VERSION="8.1"
        ;;
    4)
        PHP_VERSION="8.2"
        ;;
    5)
        PHP_VERSION="8.3"
        ;;
    6)
        read -p "Enter the PHP version you want to install (e.g., 7.4, 8.0): " PHP_VERSION
        ;;
    *)
        echo "Invalid choice for PHP version. Exiting."
        exit 1
        ;;
esac

# Install PHP and specified extensions
sudo apt-get install -y php$PHP_VERSION-bcmath php$PHP_VERSION-bz2 php$PHP_VERSION-cli php$PHP_VERSION-common php$PHP_VERSION-curl php$PHP_VERSION-dev php$PHP_VERSION-gd php$PHP_VERSION-igbinary php$PHP_VERSION-imagick php$PHP_VERSION-imap php$PHP_VERSION-intl php$PHP_VERSION-mbstring php$PHP_VERSION-memcached php$PHP_VERSION-msgpack php$PHP_VERSION-mysql php$PHP_VERSION-opcache php$PHP_VERSION-pgsql php$PHP_VERSION-readline php$PHP_VERSION-redis php$PHP_VERSION-ssh2 php$PHP_VERSION-tidy php$PHP_VERSION-xml php$PHP_VERSION-xmlrpc php$PHP_VERSION-zip -y

# Install the selected web server
if [ "$WEB_SERVER" = "nginx" ]; then
    sudo apt-get install -y nginx
elif [ "$WEB_SERVER" = "apache" ]; then
    sudo apt-get install -y apache2
fi

echo "PHP $PHP_VERSION and $WEB_SERVER installed successfully."
