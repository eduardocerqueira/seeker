#date: 2024-08-15T17:09:30Z
#url: https://api.github.com/gists/f7dccfda0aa5a7c10226ae4f460ead4c
#owner: https://api.github.com/users/ilynne

FROM ruby:3.0-bullseye

ENV BUNDLER_VERSION=2

RUN apt-get update && apt-get install -y \
    build-essential \
    nodejs \
    npm \
    yarn \
    python \
    libpq-dev \
    && gem install bundler \
    && mkdir -p /app

WORKDIR /app

# COPY Gemfile Gemfile.lock ./
# RUN bundle config build.nokogiri --use-system-libraries
# RUN bundle check || bundle install

# COPY package.json yarn.lock ./
# RUN yarn install --check-files

COPY . ./

RUN curl -o- https://raw.githubusercontent.com/creationix/nvm/v0.34.0/install.sh | bash
RUN curl -fsSL https://raw.githubusercontent.com/tophat/yvm/master/scripts/install.sh | bash
CMD ["nvm install 16.20.0"]
CMD ["nvm use 16.20.0"]
CMD ["npm install webpack-dev-server"]
CMD ["yvm use 1.22.22"]

# ENTRYPOINT ["./entrypoints/docker-entrypoint.sh"]

# EXPOSE 3000

# CMD ["/bin/bash"]

# CMD ["bundle", "exec", "rails", "server", "-b", "0.0.0.0"]
