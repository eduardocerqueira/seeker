#date: 2022-06-17T17:02:31Z
#url: https://api.github.com/gists/471bc58e08abcddfb2d6229a40b5aa9a
#owner: https://api.github.com/users/katrbhach

FROM python:3.7.13-slim-bullseye
RUN apt update && apt install -y librdkafka-dev gcc
WORKDIR /opt/kafka-helper-utility
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY app app
COPY gunicorn.conf.py gunicorn.conf.py
COPY gunicorn_logging.ini gunicorn_logging.ini
COPY run.py run.py
COPY test-mrc.json /etc/conf/mrc-config.json
ENTRYPOINT ["gunicorn", "run:app"]
