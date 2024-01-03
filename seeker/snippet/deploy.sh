#date: 2024-01-03T17:01:17Z
#url: https://api.github.com/gists/993512249011a73b43925f94db763e67
#owner: https://api.github.com/users/6hislain

#!/usr/bin/env bash
echo "Running composer"
composer global require hirak/prestissimo
composer install --no-dev --working-dir=/var/www/html

echo "generating application key..."
php artisan key:generate --show

echo "Caching config..."
php artisan config:cache

echo "Caching routes..."
php artisan route:cache

echo "Running migrations..."
php artisan migrate --force
