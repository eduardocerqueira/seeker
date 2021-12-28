#date: 2021-12-28T17:07:37Z
#url: https://api.github.com/gists/62a0ba59de5d8262d314d7a2e3150c35
#owner: https://api.github.com/users/theredrad

RUN --mount=type=secret,id=github,dst=/root/.netrc \
    go env -w GOPRIVATE=github.com/txcash/* \
    && env GIT_TERMINAL_PROMPT=1 go mod download