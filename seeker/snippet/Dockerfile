#date: 2021-09-06T16:55:34Z
#url: https://api.github.com/gists/d66ac317e414223d58a060bf0bcc2cb5
#owner: https://api.github.com/users/kwaight

FROM php:5.6-fpm-alpine

# Add PHP dependencies
RUN apk add --update \
    bzip2-dev \
    curl-dev \
    oniguruma-dev \
    libxml2-dev \
    libsodium-dev \
    libzip-dev \
    jpeg-dev \
    libpng-dev \
    icu-dev

## Add PHP extensions
RUN apk add --no-cache $PHPIZE_DEPS \
    && docker-php-ext-configure gd --with-jpeg \
    && docker-php-ext-install zip pdo_mysql pcntl exif gd intl

## Install Composer
RUN php -r "copy('https://getcomposer.org/installer', 'composer-setup.php');" \
    && php composer-setup.php --install-dir=/usr/local/bin --filename=composer \
    && php -r "unlink('composer-setup.php');"

VOLUME /app

COPY start /start
RUN chmod +x /start
ENTRYPOINT ["/start"]
