#date: 2022-02-21T16:58:53Z
#url: https://api.github.com/gists/eb5ce434d787965ee0f6eac423477338
#owner: https://api.github.com/users/FranzDiebold

alias python3.8='docker run --rm -it -v "${PWD}":/usr/src/app -w /usr/src/app python:3.8 python'
alias python3.9='docker run --rm -it -v "${PWD}":/usr/src/app -w /usr/src/app python:3.9 python'
alias python3.10='docker run --rm -it -v "${PWD}":/usr/src/app -w /usr/src/app python:3.10 python'
alias python='docker run --rm -it -v "${PWD}":/usr/src/app -w /usr/src/app python:latest python'