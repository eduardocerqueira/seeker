#date: 2022-06-02T16:50:57Z
#url: https://api.github.com/gists/12ab04b36540b4d87f22d7f5c3005b2a
#owner: https://api.github.com/users/AlperRehaYAZGAN

# first stage
FROM golang:1.17-alpine3.15 AS build-env
RUN apk add build-base

ADD . /src

RUN cd /src && go build -o goapp

# second stage
FROM alpine

WORKDIR /app

COPY --from=build-env /src/goapp /app/

EXPOSE 9090

ENTRYPOINT ./goapp