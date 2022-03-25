#date: 2022-03-25T17:03:15Z
#url: https://api.github.com/gists/9b5f7739155592fa1f3c073ebf00a65b
#owner: https://api.github.com/users/dhanyn10

#Referensi:https://github.com/microsoft/msphpsql/issues/1346#issuecomment-1000233759

ARG PHP_VERSION=8.0

# "php" stage
FROM php:${PHP_VERSION}-buster

# Install system dependencies
RUN apt update && apt install -y \
    git \
    curl \
    libpng-dev \
    libonig-dev \
    libxml2-dev \
    zip \
    unzip

RUN set -eux; \
	\
	ln -srf $PHP_INI_DIR/php.ini-production $PHP_INI_DIR/php.ini


RUN set -eux; \
	\
	apt-get update; \
	apt-get install -y --no-install-recommends \
		gnupg \
		locales \
	; \
	curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -; \
	curl https://packages.microsoft.com/config/debian/10/prod.list > /etc/apt/sources.list.d/mssql-release.list; \
	\
	apt-get update; \
	ACCEPT_EULA=Y apt-get install -y --no-install-recommends \
		msodbcsql17 \
	; \
	rm -rf /var/lib/apt/lists/*


RUN set -eux; \
	\
	sed -i 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/g' /etc/locale.gen; \
	locale-gen


ARG SQLSRV_VERSION=5.9.0
RUN set -eux; \
	\
	savedAptMark="$(apt-mark showmanual)"; \
	\
	apt-get update; \
	apt-get install -y --no-install-recommends \
		unixodbc-dev \
	; \
	pecl install \
		pdo_sqlsrv-${SQLSRV_VERSION} \
		sqlsrv-${SQLSRV_VERSION} \
	; \
	pecl clear-cache; \
	docker-php-ext-enable \
		pdo_sqlsrv \
		sqlsrv \
	; \
	# reset apt-mark's "manual" list so that "purge --auto-remove" will remove all build dependencies
	apt-mark auto '.*' > /dev/null; \
	apt-mark manual $savedAptMark; \
	ldd "$(php -r 'echo ini_get("extension_dir");')"/*.so \
		| awk '/=>/ { print $3 }' \
		| sort -u \
		| xargs -r dpkg-query -S \
		| cut -d: -f1 \
		| sort -u \
		| xargs -rt apt-mark manual \
	; \
	apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false; \
	rm -rf /var/lib/apt/lists/*

# Get latest Composer
COPY --from=composer:latest /usr/bin/composer /usr/bin/composer

# copy all of the file in folder to /src
COPY . /src
WORKDIR /src

RUN composer install

#start docker service
CMD php artisan serve --host 0.0.0.0 