#date: 2024-04-11T16:44:22Z
#url: https://api.github.com/gists/d7bb6431752c32edc773f097e5dedda3
#owner: https://api.github.com/users/GrzegorzMika

FROM python:3.12-slim AS builder

RUN apt-get update \
    && apt-get install gcc -y \
    && apt install -y libpq-dev \
    && apt-get clean

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

COPY ./requirements.txt /app/requirements.txt

RUN pip install --user -r /app/requirements.txt && rm -rf /root/.cache/pip

FROM python:3.12-slim AS app

COPY --from=builder /root/.local /root/.local

COPY . /app/

ENV PATH=/root/.local/bin:$PATH

WORKDIR /app

CMD python main.py