#date: 2022-06-30T16:54:17Z
#url: https://api.github.com/gists/58fb0d7e1a7b8b7b3e6a86c3a17de87b
#owner: https://api.github.com/users/kevinfiol

FROM alpine:3.16

RUN apk update \
    && apk add --no-cache \
        bash \
        zip

RUN wget https://justine.lol/ape.elf -O /usr/bin/ape
RUN chmod +x /usr/bin/ape

COPY redbean.com /
RUN chmod +x /redbean.com

CMD ["bash", "-c", "/redbean.com", "-vv", "-p", "8080"]