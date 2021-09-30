#date: 2021-09-30T17:06:23Z
#url: https://api.github.com/gists/b585b346adb03c4ae26c76948737807b
#owner: https://api.github.com/users/fladago

FROM golang:1.17.1-alpine3.14 AS build
WORKDIR /src/
COPY go.* /src/
RUN go mod download
COPY ./ /src/
RUN CGO_ENABLED=0 go build -o /bin/demo
FROM scratch
COPY --from=build /bin/demo /bin/demo
ENTRYPOINT [ "/bin/demo" ]