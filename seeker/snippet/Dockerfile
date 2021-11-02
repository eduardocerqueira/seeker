#date: 2021-11-02T17:00:18Z
#url: https://api.github.com/gists/7e7d9fb5f5ed75b9e14b570b4e5fe687
#owner: https://api.github.com/users/jac18281828

ARG VERSION=110221

FROM jac18281828/pythondev:${VERSION}

# build project
ARG PROJECT=subscribe
WORKDIR /workspaces/${PROJECT}

COPY requirements.txt .
COPY apikey.key .
COPY subscribe.py .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
ENV PYTHONPATH=.
CMD python3 subscribe.py apikey.key
