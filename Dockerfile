FROM registry.access.redhat.com/ubi8/python-38

LABEL author="eduardomcerqueira@gmail.com"
LABEL maintainer="eduardomcerqueira@gmail.com"
LABEL description="seeker for new code snippets"

ENV GITHUB_TOKEN=$GITHUB_TOKEN
ENV GITHUB_USERNAME=${GITHUB_USERNAME:-"eduardomcerqueira"}
ENV GITHUB_EMAIL=${GITHUB_EMAIL:-"eduardomcerqueira@gmail.com"}
ARG SEEKER_RUN=${SEEKER_RUN:-""}

RUN git config --global user.name $GITHUB_USERNAME
RUN git config --global user.email $GITHUB_EMAIL
RUN git config --global http.sslVerify false
RUN git remote set-url --push origin https://$GITHUB_USERNAME:$GITHUB_TOKEN@github.com/eduardocerqueira/seeker

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir devpi-client
RUN pip install --no-cache-dir -U pip setuptools setuptools_scm wheel

# install from repo/
RUN git clone https://github.com/eduardocerqueira/seeker.git
WORKDIR seeker
RUN pip install --no-cache-dir -e .
RUN pip freeze |grep seeker

RUN env | grep -e SEEKER_RUN -e GITHUB -e GITHUB_TOKEN

# check
WORKDIR seeker
RUN seeker $SEEKER_RUN