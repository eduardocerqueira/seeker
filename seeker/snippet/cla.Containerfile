#date: 2025-07-01T17:12:57Z
#url: https://api.github.com/gists/afb048fadaa0d001189e4e95610091d6
#owner: https://api.github.com/users/r0x0d

FROM registry.access.redhat.com/ubi10/ubi:latest AS base

ENV DNF_DEFAULT_OPTS -y --nodocs --setopt=keepcache=0 --setopt=tsflags=nodocs

RUN dnf install ${DNF_DEFAULT_OPTS} \
    python3.12 \
    python3.12-pip

FROM base as build

WORKDIR /project

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

COPY ./pyproject.toml ./poetry.lock* README.md LICENSE ./
COPY command_line_assistant/ ./command_line_assistant/

RUN dnf install ${DNF_DEFAULT_OPTS} \
    gcc \
    python3-devel \
    python3-setuptools \
    python3-wheel

RUN pip3.12 install -U poetry && poetry install

RUN poetry build

FROM base as final

COPY --from=build /project/dist/command_line_assistant*.whl /tmp/

RUN pip install --prefix=/usr --no-cache-dir /tmp/command_line_assistant-0.4.0-py3-none-any.whl \
    && dnf remove -y python3-pip \
    && dnf clean all && rm -rf /var/cache/dnf /var/tmp/* /tmp/*

# Config 
COPY data/release/xdg/config.toml /etc/xdg/command-line-assistant/config.toml

# Systemd specifics
COPY data/release/systemd/clad.service /usr/lib/systemd/system/clad.service
COPY data/release/systemd/clad.tmpfiles.conf /usr/lib/tmpfiles.d/clad.tmpfiles.conf

# Dbus specifics
COPY data/release/dbus/*.service /usr/share/dbus-1/system-services/
COPY data/release/dbus/com.redhat.lightspeed.conf /usr/share/dbus-1/system.d/com.redhat.lightspeed.conf

# Manpages
COPY data/release/man/c.1 /usr/share/man/man1/
COPY data/release/man/c.1 /usr/share/man/man1/cla.1
COPY data/release/man/clad.8 /usr/share/man/man/8

STOPSIGNAL SIGRTMIN+3
CMD ["/sbin/init"]