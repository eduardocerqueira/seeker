#date: 2024-11-27T17:10:30Z
#url: https://api.github.com/gists/4fd63bad1230b0aef452e0dde6fc7d39
#owner: https://api.github.com/users/jkonowitch

FROM flyio/postgres-flex:16.4

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  git \
  build-essential \
  postgresql-server-dev-16 \
  unzip \
  curl && \
  cd /home && \
  curl -LO https://api.pgxn.org/dist/pgmq/1.4.5/pgmq-1.4.5.zip && \
  unzip pgmq-1.4.5.zip && \
  cd pgmq-1.4.5 && \
  make && \
  make install && \
  # Cleanup
  cd / && \
  rm -rf /home/pgmq* && \
  apt-get remove -y \
  git \
  build-essential \
  postgresql-server-dev-16 \
  unzip \
  curl && \
  apt-get autoremove -y && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*