#date: 2023-05-11T17:01:23Z
#url: https://api.github.com/gists/2d1f4af8588a1c2a38ca25c3adec7148
#owner: https://api.github.com/users/thedomeffm

FROM composer:2.5 AS build

WORKDIR /application

ENV APP_ENV=prod

COPY . .

RUN composer install --no-dev --no-scripts --ignore-platform-reqs --optimize-autoloader

FROM php:8.1-cli-bullseye
WORKDIR /application

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install -y \
	apt-transport-https \
	ca-certificates \
	libicu-dev \
	libicu-dev \
	libpng-dev \
	libzip-dev \
	software-properties-common \
	wget \
	zlib1g-dev \
    cron \
    curl \
    git \
    openssl \
    unzip \
    python3-pip \
    supervisor \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /etc/cron.*/*

# RUN pip3 --version && pip install supervisor
#    supervisor \

RUN docker-php-ext-install -j$(nproc) intl \
	&& docker-php-ext-install -j$(nproc) opcache \
	&& docker-php-ext-install -j$(nproc) zip \
	&& docker-php-ext-install -j$(nproc) pdo_mysql \
	&& docker-php-ext-install -j$(nproc) pcntl

COPY messenger-worker.conf /etc/supervisor/conf.d/messenger-worker.conf
COPY entrypoint.sh /entrypoint.sh

#RUN supervisord start && supervisorctl reload
#RUN supervisorctl update

COPY --from=build /application /application

RUN APP_ENV=prod APP_DEBUG=0 php bin/console cache:clear

RUN chmod +x /entrypoint.sh

RUN ls -la /etc/supervisor/conf.d

ENTRYPOINT ["/entrypoint.sh"]

# CMD ["supervisorctl","start", "messenger-consume:*"]
CMD ["supervisorctl","start", "messenger-consume:*"]
