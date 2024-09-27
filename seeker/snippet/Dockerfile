#date: 2024-09-27T16:57:25Z
#url: https://api.github.com/gists/ebba88ea89306032a7024b68eda07405
#owner: https://api.github.com/users/Sarps

FROM golang:1.22.5 AS build

WORKDIR /app

ARG PRISMA_GLOBAL_CACHE_DIR="/app/prisma/cache"

RUN go install github.com/go-task/task/v3/cmd/task@latest

COPY Taskfile.yml .
COPY go.mod .
COPY go.sum .
RUN task install:ci

COPY prisma/schema.prisma ./prisma/schema.prisma
RUN task prisma:generate

COPY . .
RUN CGO_ENABLED=0 GOOS=linux task build

FROM scratch AS final
LABEL maintainer="Emmanuel Sarpong"

WORKDIR /app

ARG PRISMA_GLOBAL_CACHE_DIR="/app/prisma/cache"

COPY --from=build $PRISMA_GLOBAL_CACHE_DIR $PRISMA_GLOBAL_CACHE_DIR
COPY --from=build /app/bin/app ./

EXPOSE 8080

ENV PRISMA_GLOBAL_CACHE_DIR=$PRISMA_GLOBAL_CACHE_DIR
ENTRYPOINT [ "./app" ]