#date: 2025-05-08T16:45:08Z
#url: https://api.github.com/gists/b8605579ae9a9011f9f49b005c014c22
#owner: https://api.github.com/users/hadikhah

# create a new laravel project with kit starter: 
docker run -it --rm \
    --user "$(id -u):$(id -g)" \
    -v "$(pwd):/app" \
    -w /app \
    -e COMPOSER_HOME=/tmp/composer \
    laravelsail/php84-composer:latest \
    bash -c "composer global require laravel/installer && /tmp/composer/vendor/bin/laravel new my-awesome-app"

 #Then go to your project 
cd my-awesome-app
# and install sail 
docker run -it --rm \
    --user "$(id -u):$(id -g)" \
    -v "$(pwd):/app" \
    -w /app \
    -e COMPOSER_HOME=/tmp/composer \
    laravelsail/php84-composer:latest \
   php artisan sail:install