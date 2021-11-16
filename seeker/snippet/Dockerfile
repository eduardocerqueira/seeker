#date: 2021-11-16T16:58:33Z
#url: https://api.github.com/gists/1041e0384619d948e3590190b86584fc
#owner: https://api.github.com/users/Luiz-Monad

FROM mcr.microsoft.com/dotnet/core/sdk:3.1.402-alpine3.12 as debugger-env

# Debugger
WORKDIR /dbg
COPY sdbg.sh .
COPY GetVsDbg.sh .
RUN ./GetVsDbg.sh -l . -v latest

# User SSH key
WORKDIR /root
RUN apk add openssh && \
    ssh-keygen -A && \
    ssh-keygen -t rsa -f id_rsa -P "" && \
    cat id_rsa && echo "" && \
    cat id_rsa.pub && echo ""

# Update the runtime image
FROM dev:latest
WORKDIR /
COPY --from=debugger-env . .

# SSH keys and config
WORKDIR /root
RUN apk add openssh && \
    ssh-keygen -A && \
    mkdir -p .ssh && \
    mv id_rsa.pub .ssh/authorized_keys && \
    rm id_rsa && \
    echo `root:$(random)` | chpasswd
COPY sshd_config /etc/ssh/

# User Shell
RUN apk add bash procps
RUN sed -i '/root/s/ash/bash/g' /etc/passwd

# Startup
WORKDIR /
COPY start.sh .

# Export
EXPOSE 22
ENTRYPOINT /start.sh
