#date: 2022-01-04T17:11:46Z
#url: https://api.github.com/gists/68c1859cd237e6e73a86ca3c5e31c9cd
#owner: https://api.github.com/users/srujandeshpande

FROM apache/airflow:2.2.3

USER root
RUN apt-get update && \
    apt-get -y install apt-transport-https \
    ca-certificates \
    curl \
    gnupg2 \
    software-properties-common
RUN curl -fsSL https://download.docker.com/linux/$(. /etc/os-release; echo "$ID")/gpg > /tmp/dkey
RUN apt-key add /tmp/dkey && \
    add-apt-repository \
    "deb [arch=amd64] https://download.docker.com/linux/$(. /etc/os-release; echo "$ID") \
    $(lsb_release -cs) \
    stable"
RUN apt-get update && \
    apt-get -y install docker-ce
