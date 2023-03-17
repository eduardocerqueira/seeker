#date: 2023-03-17T17:05:12Z
#url: https://api.github.com/gists/df8c0da569a414d633a8da745de50e03
#owner: https://api.github.com/users/ggmariano22

FROM golang

WORKDIR /src

ADD go.mod go.sum ./
RUN go mod download
RUN go mod tidy

COPY . .

RUN go build -o app

CMD ["./app"]