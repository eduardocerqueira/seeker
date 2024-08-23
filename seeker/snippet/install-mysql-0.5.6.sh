#date: 2024-08-23T17:01:14Z
#url: https://api.github.com/gists/811589676080319d4767a2d381781eda
#owner: https://api.github.com/users/ghoppe

gem install mysql2 -v '0.5.6' -- --with-mysql-config=$(brew --prefix mysql)/bin/mysql_config --with-ldflags="-L$(brew --prefix zstd)/lib -L$(brew --prefix openssl)/lib" --with-cppflags=-I$(brew --prefix openssl)/include