#date: 2023-04-06T17:08:28Z
#url: https://api.github.com/gists/7e0f71dd7fc33fc5193c1b3de47ab584
#owner: https://api.github.com/users/BlackthornYugen

FROM --platform=linux/arm64 amazonlinux:2
RUN yum install -y openssh-server
RUN useradd -ms /bin/bash jumpuser
RUN mkdir -v /run/sshd
RUN ssh-keygen -A

USER jumpuser
RUN mkdir ~/.gnupg ~/.ssh
RUN echo 'eval `ssh-agent -s` > /dev/null' >> ~/.bashrc
RUN echo 'export SSH_AUTH_SOCK=$HOME/S.gpg-agent.ssh' >> ~/.bashrc

USER root
COPY entrypoint.sh /
ENTRYPOINT [ "/entrypoint.sh" ]