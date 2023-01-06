#date: 2023-01-06T17:03:28Z
#url: https://api.github.com/gists/e5c21a34ef89be4e22bf7b616e4dcfb7
#owner: https://api.github.com/users/ahmadhajmosa

# Base Image
FROM python:3.8.10-alpine

ADD requirements.txt /app/requirements.txt

ADD services.sh /
RUN chmod +x /services.sh

RUN set -ex \
    && apk add --no-cache --virtual .build-deps gcc libc-dev libffi-dev linux-headers mariadb-dev python3-dev build-base \
    && python -m venv /venv \
    && /venv/bin/pip install --upgrade pip \
    && /venv/bin/pip install --no-cache-dir -r /app/requirements.txt \
    && runDeps="$(scanelf --needed --nobanner --recursive /venv \
        | awk '{ gsub(/,/, "\nso:", $2); print "so:" $2 }' \
        | sort -u \
        | xargs -r apk info --installed \
        | sort -u)" \
    && apk add --virtual rundeps $runDeps \
    && apk del .build-deps

# Add current directory code to working directory
ADD . /app/
# create and set working directory
WORKDIR /app

ENV PATH /venv/bin:$PATH

EXPOSE 8080

CMD ["/services.sh"]