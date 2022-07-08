#date: 2022-07-08T16:55:11Z
#url: https://api.github.com/gists/1b3c0301fc9f51b788216195cb88a121
#owner: https://api.github.com/users/chiefcake

FROM --platform=$BUILDPLATFORM golang:1.18-buster as builder

WORKDIR /go/src/github.com/{ORGANIZATION_NAME}/{REPOSITORY_NAME}

COPY go.* .

RUN go mod download

COPY . .

ARG TARGETOS TARGETARCH

RUN CGO_ENABLED=0 GOOS=$TARGETOS GOARCH=$TARGETARCH go build -o {BINARY_NAME} main.go

FROM alpine:3.16.0

RUN apk --no-cache add ca-certificates

COPY --from=builder /go/src/github.com/{ORGANIZATION_NAME}/{REPOSITORY_NAME} .

CMD [ "./{BINARY_NAME}" ]
