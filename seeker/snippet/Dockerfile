#date: 2021-11-05T17:10:01Z
#url: https://api.github.com/gists/57696add36286f72e2d385c7d446cae5
#owner: https://api.github.com/users/lubiedo

FROM debian:bullseye-slim

RUN apt-get -q -y update && apt -q -y install nginx ca-certificates apt-transport-https wget lsb-release gnupg

RUN wget -q https://packages.sury.org/php/apt.gpg -O- | apt-key add -
RUN echo "deb https://packages.sury.org/php/ stretch main" | tee /etc/apt/sources.list.d/php.list
RUN apt -q update && apt -q -y install php5.6 php5.6-cli php5.6-common php5.6-mysqlnd php5.6-fpm

COPY . /var/www/html
RUN chown -R www-data:www-data /var/www
COPY default /etc/nginx/sites-available/default
RUN sed -ie '/^;access\.log/ s/;//; /^access\.log/ s/log\//\/var\/log/' \
	/etc/php/5.6/fpm/pool.d/www.conf
RUN sed -i '/^short_open_tag/s/ff/n/' /etc/php/5.6/fpm/php.ini

RUN wget https://dev.mysql.com/get/mysql-apt-config_0.8.13-1_all.deb && \
	dpkg -i mysql-apt-config_0.8.13-1_all.deb && \
        apt -q update && apt -y install default-mysql-server default-mysql-client
RUN sed -ie '/^# port =/s/# //' /etc/mysql/my.cnf && \
	service mariadb start && mysqladmin create amadey && \
	mysql -e "CREATE USER 'amadey'@'localhost' IDENTIFIED BY 'lol';GRANT ALL PRIVILEGES ON * . * TO 'amadey'@'localhost';FLUSH PRIVILEGES;"

EXPOSE 80/tcp
WORKDIR /
CMD service mariadb start && service php5.6-fpm start && service nginx start && \
	tail -f /var/log/nginx/error.log