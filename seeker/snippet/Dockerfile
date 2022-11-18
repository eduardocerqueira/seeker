#date: 2022-11-18T16:57:37Z
#url: https://api.github.com/gists/02957c1d0a139d03e899b0bf9e4808f4
#owner: https://api.github.com/users/mazz

# Find eligible builder and runner images on Docker Hub. We use Ubuntu/Debian instead of
# Alpine to avoid DNS resolution issues in production.
#
# https://hub.docker.com/r/hexpm/elixir/tags?page=1&name=ubuntu
# https://hub.docker.com/_/ubuntu?tab=tags
#
#
# This file is based on these images:
#
#   - https://hub.docker.com/r/hexpm/elixir/tags - for the build image
#   - https://hub.docker.com/_/debian?tab=tags&page=1&name=bullseye-20210902-slim - for the release image
#   - https://pkgs.org/ - resource for finding needed packages
#   - Ex: hexpm/elixir:1.13.2-erlang-24.2.1-debian-bullseye-20210902-slim
#

ARG BUILDER_IMAGE="hexpm/elixir:1.14.1-erlang-25.1.1-debian-bullseye-20220801"
ARG RUNNER_IMAGE="debian:bullseye-20220801-slim"

FROM ${BUILDER_IMAGE} as builder

# install build dependencies
RUN apt-get update -y && apt-get install -y build-essential git npm \
    && apt-get clean && rm -f /var/lib/apt/lists/*_*

# fix at 8.19.3 because 
# ERROR: npm v9.1.2 is known not to run on Node.js v12.22.12. You'll need to
# upgrade to a newer Node.js version in order to use this version of npm. This
# version of npm supports the following node versions: `^14.17.0 || ^16.13.0 ||
# >=18.0.0`. You can find the latest version at https://nodejs.org/.

RUN npm install -g npm@8.19.3

WORKDIR /app

# install hex + rebar
RUN mix local.hex --force && \
    mix local.rebar --force

# set build ENV
ENV MIX_ENV="prod"

# install mix dependencies
COPY mix.exs mix.lock ./
RUN mix deps.get --only $MIX_ENV
RUN mkdir config

# copy compile-time config files before we compile dependencies
# to ensure any relevant config change will trigger the dependencies
# to be re-compiled.
COPY config/config.exs config/${MIX_ENV}.exs config/
RUN mix deps.compile

# install npm dependencies
COPY assets/package.json assets/package-lock.json ./assets/
RUN npm --prefix ./assets install --progress=false --no-audit --loglevel=error

COPY priv priv

# note: if your project uses a tool like https://purgecss.com/,
# which customizes asset compilation based on what it finds in
# your Elixir templates, you will need to move the asset compilation
# step down so that `lib` is available.
COPY assets assets

# Compile the release
COPY lib lib

# compile assets
RUN mix assets.deploy

RUN mix compile
# RUN npm install --prefix ./assets
# RUN npm run deploy --prefix ./assets
# RUN mix phx.digest

# Changes to config/runtime.exs don't require recompiling the code
COPY config/runtime.exs config/

COPY rel rel
RUN mix release

# start a new build stage so that the final image will only contain
# the compiled release and other runtime necessities
FROM ${RUNNER_IMAGE}

RUN apt-get update -y && apt-get install -y libstdc++6 openssl libncurses5 locales \
  && apt-get clean && rm -f /var/lib/apt/lists/*_*

# Set the locale
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

WORKDIR "/app"
RUN chown nobody /app

# Only copy the final release from the build stage
COPY --from=builder --chown=nobody:root /app/_build/prod/rel/faithful_word ./
COPY entrypoint.sh .
USER nobody

ENV HOME=/app
CMD ["bash", "/app/entrypoint.sh"]

# CMD ["/app/bin/server"]