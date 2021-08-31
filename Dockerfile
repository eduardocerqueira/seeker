FROM registry.access.redhat.com/ubi8/python-38

LABEL author="eduardomcerqueira@gmail.com"
LABEL maintainer="eduardomcerqueira@gmail.com"
LABEL description="seeker for new code snippets"

ENV TOKEN=$TOKEN
ENV GITHUB_USERNAME=${GITHUB_USERNAME:-"eduardomcerqueira"}
ENV GITHUB_EMAIL=${GITHUB_EMAIL:-"eduardomcerqueira@gmail.com"}

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir devpi-client
RUN git config --global http.sslVerify false
RUN pip install --no-cache-dir -U pip setuptools setuptools_scm wheel

# install from repo/
RUN git clone https://github.com/eduardocerqueira/seeker.git
WORKDIR seeker
RUN pip install -e .

RUN git config --global user.name $GITHUB_USERNAME
RUN git config --global user.email $GITHUB_EMAIL

# check
WORKDIR seeker
RUN seeker --test