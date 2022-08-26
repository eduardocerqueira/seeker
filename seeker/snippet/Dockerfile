#date: 2022-08-26T17:16:12Z
#url: https://api.github.com/gists/d70eebd8b0e6bb1a4578ac7c06a842cf
#owner: https://api.github.com/users/ryepup

FROM golang:1.19-alpine

WORKDIR /src
COPY go.mod ./
COPY go.sum ./
COPY main.go ./

RUN go build -o /gen-span

ENV OTEL_SERVICE_NAME=gen-span

CMD ["/gen-span"]