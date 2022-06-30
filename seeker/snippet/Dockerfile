#date: 2022-06-30T21:24:15Z
#url: https://api.github.com/gists/2ad0c0e3c3c916907e32e9c0bc1ed396
#owner: https://api.github.com/users/manics

FROM mcr.microsoft.com/mssql/server:2019-latest

USER root

RUN apt-get update -y -q && \
    apt-get install -y -q \
        libgdiplus \
        unzip && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /home/mssql && \
    chown mssql /home/mssql

USER mssql

RUN cd /opt && \
    wget -q https://github.com/HicServices/RDMP/releases/download/v7.0.14/rdmp-cli-linux-x64.zip && \
    unzip -d rdmp-cli ./rdmp-cli-linux-x64.zip && \
    chmod +x /opt/rdmp-cli/rdmp

COPY create-database-yaml /usr/local/bin/create-database-yaml

ENV ACCEPT_EULA=Y
WORKDIR /opt/rdmp-cli
