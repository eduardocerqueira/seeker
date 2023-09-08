#date: 2023-09-08T17:09:20Z
#url: https://api.github.com/gists/8c7b2394ba82d939e5a3bab227841156
#owner: https://api.github.com/users/qrichertRDR

FROM ubuntu:22.04 AS base

USER root

RUN apt-get update && apt-get -y upgrade
RUN apt-get -y install \
    software-properties-common \
    ;
RUN add-apt-repository \
    ppa:git-core/ppa \
    && apt-get update ;
RUN apt-get -y install \
    git \
    ;

RUN adduser qwitus && \
    echo "qwitus:qwitus" | chpasswd && \
    usermod -aG sudo qwitus
USER qwitus
WORKDIR /home/qwitus/
RUN touch .sudo_as_admin_successful

RUN mkdir $HOME/.ssh && chmod 700 $HOME/.ssh
RUN ssh-keyscan github.com >> $HOME/.ssh/known_hosts && chmod 600 $HOME/.ssh/known_hosts

# 1. On clone tous les repos dont on a besoin et pour lesquels on a
#    besoin d'une clé SSH. Ce build stage sera supprimé, la clé
#    n'apparaîtra pas dans l'image finale. L'avantage par rapport à
#    configurer le partage de clé SSH c'est que déjà c'est plus simple,
#    mais aussi c'est plus simple de le faire fonctionner avec des
#    secrets de la plateforme de CI/CD.
FROM base AS repos

COPY id_rsa .ssh/
USER root
RUN chown qwitus:qwitus .ssh/id_rsa
USER qwitus

WORKDIR repos
RUN git clone git@github.com:qrichert/painter.git
RUN git clone git@github.com:qrichert/moontool.git
RUN git clone git@github.com:qrichert/cronrunner.git

# 2. On crée la vraie image. L'étape précédente est supprimée, sauf le
#    dossier des repos clonés que l'on vient copier de l'étape
#    précédente (par contre attention aux permissions des fichiers,
#    c'est pour ça que l'ont fait les clones avec le même user que
#    l'user final).
FROM base AS final

COPY --from=repos /home/qwitus/repos .

CMD ["sleep", "infinity"]
