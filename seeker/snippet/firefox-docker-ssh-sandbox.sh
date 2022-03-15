#date: 2022-03-15T17:04:18Z
#url: https://api.github.com/gists/f267ff9518a266b9bd88d2e17f7c99aa
#owner: https://api.github.com/users/shollingsworth

#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# inspired by: https://blog.modest-destiny.com/posts/run-linux-gui-applications-within-docker-containers/
usage() {
    echo "Usage: $0 authorized_keys_file <port:default 4444>"
    echo "Example: $0 ~/.ssh/id_rsa.pub"
    echo "Example: $0 ~/.ssh/id_rsa.pub 5555"
    echo "Software requirements:"
    echo " - docker-ce"
    echo " - docker-compose"
    echo " - ssh"
    exit 1
}

authorized_keys="${1:-}"
test "${authorized_keys}" || usage
PORT=${2:-4444}
content="$(cat "${authorized_keys}")"
tdir="$(mktemp -d)"
cd "${tdir}"

trap 'docker-compose down ; rm -rf ${tdir}' EXIT
cd "${tdir}"

#####################################
# Dockerfile
#####################################
cat << EOF > Dockerfile
FROM debian:latest

RUN useradd -m -s /bin/bash user
RUN apt update
RUN apt install -y \
        firefox-esr \
        openssh-server \
        xauth
RUN mkdir /var/run/sshd
RUN mkdir /home/user/.ssh
RUN chown -R user:user /home/user/.ssh
RUN chmod 700 /home/user/.ssh
RUN sed -i "s/#PermitRootLogin prohibit-password/PermitRootLogin no/" /etc/ssh/sshd_config
RUN sed -i "s/^.*X11Forwarding.*$/X11Forwarding yes/" /etc/ssh/sshd_config
RUN sed -i "s/^.*X11UseLocalhost.*$/X11UseLocalhost no/" /etc/ssh/sshd_config
RUN grep "^X11UseLocalhost" /etc/ssh/sshd_config || echo "X11UseLocalhost no" >> /etc/ssh/sshd_config
EOF

#####################################
# Runtime
#####################################
cat << EOF > runtime.sh
#!/usr/bin/env bash

touch /home/user/.ssh/authorized_keys
chmod 0600 /home/user/.ssh/authorized_keys
chown user:user /home/user/.ssh/authorized_keys
echo "${content}" > /home/user/.ssh/authorized_keys
/usr/sbin/sshd -D -e
EOF

chmod +x runtime.sh

#####################################
# Compose
#####################################
cat << EOF > docker-compose.yml
version: "3.9"

services:
  firefox:
    image: sandbox-firefox
    volumes:
      - $(pwd)/runtime.sh:/runtime.sh
    build:
      context: .
    ports:
      - "127.0.0.1:${PORT}:22"
    command: /runtime.sh
EOF

docker-compose up --build --remove-orphans -d
ssh \
    -X \
    user@localhost \
    -p "${PORT}" \
    -o StrictHostKeyChecking=no \
    firefox
