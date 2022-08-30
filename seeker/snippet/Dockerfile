#date: 2022-08-30T16:51:53Z
#url: https://api.github.com/gists/4b92ca454e381d41ca33ae74034e988c
#owner: https://api.github.com/users/halomademeapc


FROM alpine:latest as unzipper
RUN apk add unzip wget curl
RUN mkdir /download/
RUN mkdir /download/output/
RUN wget -O /download/file.zip https://filebrowser.linuxforphp.net/files/filebrowser-8.0.2.zip
RUN unzip /download/file.zip -d /download/output/

FROM php:7.4-apache
RUN apt update
RUN apt install zip unzip
COPY --from=unzipper /download/output/ /var/www/
RUN chown -R www-data:www-data /var/www/filebrowser/
RUN chmod -R 775 /var/www/filebrowser/
RUN sed -i 's/post_max_size.*/post_max_size = 100M/g' "$PHP_INI_DIR/php.ini-production"
RUN sed -i 's/upload_max_filesize.*/upload_max_filesize = 100M/g' "$PHP_INI_DIR/php.ini-production"
RUN mv "$PHP_INI_DIR/php.ini-production" "$PHP_INI_DIR/php.ini"

WORKDIR /var/www/html

COPY . .
RUN echo "\n<VirtualHost *:80>\n    DocumentRoot /var/www/filebrowser/dist\n    LimitRequestBody 104857600\n</VirtualHost>\n" >> /etc/apache2/sites-available/filebrowser.conf
RUN a2dissite 000-default
RUN a2ensite filebrowser
RUN service apache2 restart
EXPOSE 80