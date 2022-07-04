#date: 2022-07-04T16:56:45Z
#url: https://api.github.com/gists/bd407274bc125d6bac585ea5b5b03d43
#owner: https://api.github.com/users/ragoncsa

FROM golang:1.17.7-alpine3.15 AS build
WORKDIR /app
COPY go.mod ./
COPY go.sum ./
RUN go mod download
COPY . ./
RUN GOOS=linux CGO_ENABLED=0 go build -o /todo

FROM public.ecr.aws/lambda/go:1
COPY --from=build /todo ${LAMBDA_TASK_ROOT}
COPY --from=build /app/config ${LAMBDA_TASK_ROOT}/config
ENV ENABLE_GIN_LAMBDA_PROXY=TRUE

# Command can be overwritten by providing a different command in the template directly.
CMD ["todo"]