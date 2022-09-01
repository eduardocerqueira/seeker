#date: 2022-09-01T17:01:57Z
#url: https://api.github.com/gists/87efc93b5fcb98f53aec624931d5176e
#owner: https://api.github.com/users/LuciaDR

#! /usr/bin/env bash

docker pull prom/node-exporter:v1.3.1
docker pull gcr.io/cadvisor/cadvisor:v0.45.0
docker pull prom/prometheus:v2.38.0
docker pull prom/alertmanager:v0.24.0
docker pull grafana/grafana:9.1.1
docker pull grafana/loki:2.6.1
docker pull grafana/promtail:2.6.1
docker pull elasticsearch:8.3.3
docker pull logstash:8.3.3
docker pull elastic/filebeat:8.3.3
docker pull kibana:8.3.3