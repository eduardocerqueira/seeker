#date: 2023-02-15T17:01:46Z
#url: https://api.github.com/gists/7a8ada8c3ae0d3f705a9e66882184948
#owner: https://api.github.com/users/ryansch

FROM hexpm/elixir:some_version

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# Install nodejs
RUN set -eux; \
      \
      apt-get update; \
      apt-get install -y \
        curl \
        ca-certificates \
      ; \
      \
      curl -fsSL https://deb.nodesource.com/setup_18.x | bash -; \
      \
      apt-get update; \
      apt-get install -y \
        build-essential \
        git \
        nodejs \
      ; \
      apt-get clean; \
      rm -f /var/lib/apt/lists/*_*
