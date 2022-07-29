#date: 2022-07-29T17:06:53Z
#url: https://api.github.com/gists/a71eba185c900f47b82007470dc377d1
#owner: https://api.github.com/users/asherf

FROM python:3.9.12-slim-buster@sha256:62ed2b347a385102d33f5e82530862359f8dc60464674d78cef844b02d150a50
ARG PANTS_VERSION=2.12.0

RUN apt-get update && \
    apt-get install --no-install-recommends -y \ 
        curl git unzip gcc libssl-dev bash && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    groupadd --gid 99 kramer && useradd -m --no-log-init --uid 99 --gid 99 kramer

USER kramer:kramer

COPY --chown=kramer:kramer pants-bootstrap.sh /home/kramer/
WORKDIR /home/kramer

RUN ./pants-bootstrap.sh "${PANTS_VERSION}" && rm pants-bootstrap.sh
... more stuff 