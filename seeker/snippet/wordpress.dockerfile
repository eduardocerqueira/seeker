#date: 2022-04-12T17:05:16Z
#url: https://api.github.com/gists/905c30775fce438accd749a51af724c8
#owner: https://api.github.com/users/stalinkay

FROM wordpress:latest
RUN apt-get update && apt-get install msmtp -y && \
    rm -rf /var/lib/apt/lists/*
COPY msmtprc /etc/msmtprc
RUN chown :msmtp /etc/msmtprc && \
    chmod 640 /etc/msmtprc
