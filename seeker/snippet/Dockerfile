#date: 2022-10-12T17:36:28Z
#url: https://api.github.com/gists/0b12ac69c8abc808def0d550f55330f0
#owner: https://api.github.com/users/scrossan

# courtesy of @hinchliff / https://github.com/elastic/beats/issues/13707#issuecomment-1010119726
# docker build . -t winlogbeat-on-linux:8.4.1

FROM golang:8.4.1

ARG version=7.16.2

# Install virtualenv. Required for beats build processes.
RUN apt-get update --no-install-recommends

# Python 3 venv (virtualenv)
RUN apt-get install --no-install-recommends --yes python3-venv

#RUN go get github.com/elastic/beats
RUN mkdir -p $GOPATH/src/github.com/elastic
RUN git -C $GOPATH/src/github.com/elastic clone --depth 1 --single-branch --branch v${version} https://github.com/elastic/beats
WORKDIR $GOPATH/src/github.com/elastic/beats

# Target winlogbeat
WORKDIR $GOPATH/src/github.com/elastic/beats/winlogbeat

# Patch out the `init` function in winlogbeat to avoid this error on non-Windows systems:
#   Exiting: Failed to create new event log. No event log API is available on this system
#
# This allows us to run `winlogbeat setup` on linux systems.
RUN sed -i -e 's/^\(func .* \)\(init(.*) error {.*\)$/\1\2 return nil }\n\1_\2/' beater/winlogbeat.go

# Compile winlogbeat
RUN make

# Generate winlogbeat's dashboards/templates/fields/etc
RUN make update