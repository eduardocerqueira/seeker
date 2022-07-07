#date: 2022-07-07T17:08:42Z
#url: https://api.github.com/gists/cf8ecdd05401eca9fde6ca32f2efcc4e
#owner: https://api.github.com/users/kotarokikuchi

FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt full-upgrade -y
RUN apt install -y curl 
RUN curl -sL https://deb.nodesource.com/setup_14.x | bash -
RUN curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add -
RUN echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list

RUN apt install -y git-core zlib1g-dev build-essential libssl-dev libreadline-dev libyaml-dev libsqlite3-dev sqlite3 libxml2-dev libxslt1-dev libcurl4-openssl-dev software-properties-common libffi-dev dirmngr gnupg apt-transport-https ca-certificates redis-tools nodejs yarn libpq-dev mysql-client libmysqlclient-dev imagemagick

RUN useradd -ms /bin/bash deploy
USER deploy
WORKDIR /home/deploy

RUN git clone https://github.com/rbenv/rbenv.git ~/.rbenv
RUN echo 'export PATH="$HOME/.rbenv/bin:$PATH"' >> ~/.bashrc

RUN git clone https://github.com/rbenv/ruby-build.git ~/.rbenv/plugins/ruby-build
RUN echo 'export PATH="$HOME/.rbenv/plugins/ruby-build/bin:$PATH"' >> ~/.bashrc

RUN git clone https://github.com/rbenv/rbenv-vars.git ~/.rbenv/plugins/rbenv-vars

ENV PATH "$HOME/.rbenv/bin:$HOME/.rbenv/plugins/ruby-build/bin:$PATH"
RUN echo 'eval "$(rbenv init -)"' >> ~/.bashrc

RUN /home/deploy/.rbenv/bin/rbenv install 3.1.2
RUN /home/deploy/.rbenv/bin/rbenv global 3.1.2

RUN /home/deploy/.rbenv/shims/gem install bundler
RUN /home/deploy/.rbenv/shims/gem install pg
RUN /home/deploy/.rbenv/shims/gem install mysql2
RUN /home/deploy/.rbenv/shims/gem install nokogiri
RUN /home/deploy/.rbenv/shims/gem install capistrano -v 3.17
RUN /home/deploy/.rbenv/shims/gem install capistrano-rails -v 1.6.2
RUN /home/deploy/.rbenv/shims/gem install capistrano-passenger -v 0.2.1
RUN /home/deploy/.rbenv/shims/gem install capistrano-rbenv -v 2.2

