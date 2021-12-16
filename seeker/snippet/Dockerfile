#date: 2021-12-16T17:00:30Z
#url: https://api.github.com/gists/6e8db79851179070faee0a1430afc181
#owner: https://api.github.com/users/twentythousandphantoms

FROM golang:alpine AS build

RUN apk add --update git dep make alpine-sdk linux-headers

# to build from master left the Version empty:
# I.e.: docker image build --tag your_tag --build-arg VERSION_PREFIX=master .
ARG VERSION_PREFIX="release-"
ARG VERSION

RUN git clone --depth 1 -b ${VERSION_PREFIX}${VERSION} https://github.com/influxdata/telegraf /go/src/github.com/influxdata/telegraf
WORKDIR /go/src/github.com/influxdata/telegraf

# This plugin causes error during building
RUN rm -rf plugins/inputs/aliyuncms
RUN sed -i '/aliyuncms/d' ./plugins/inputs/all/all.go
RUN make

FROM alpine:3.7 AS telegraf
MAINTAINER Andrei Belov <andrei.x.belov.-nd@disney.com>

RUN apk update && apk add ca-certificates && update-ca-certificates

COPY telegraf /etc/telegraf
COPY --from=build /go/src/github.com/influxdata/telegraf/telegraf /bin/telegraf
ENTRYPOINT [ "/bin/telegraf" ]

# docker image build --tag telegraf-1.20.3.100 --build-arg VERSION_PREFIX=master .
# docker image tag telegraf-1.20.3.100 containerregistry.mycompany.com/namespace/my-images:telegraf-1.20.3.100
# docker image push containerregistry.mycompany.com/namespace/my-images:telegraf-1.20.3.100
