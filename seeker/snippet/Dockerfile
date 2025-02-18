#date: 2025-02-18T16:43:13Z
#url: https://api.github.com/gists/994608015be1bc2a1e9f4c345ac7861d
#owner: https://api.github.com/users/up1

FROM golang:1.23.6-alpine3.21 AS builder
WORKDIR /app
COPY . .
RUN go mod tidy
RUN go build -o calculator

FROM alpine:3.21
WORKDIR /app
RUN apk add --no-cache ca-certificates
COPY --from=builder /app/calculator .
ENTRYPOINT ["./calculator"]