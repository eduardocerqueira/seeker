#date: 2021-11-29T17:02:07Z
#url: https://api.github.com/gists/85b5153e0c18b1e086d08d6fc894ca89
#owner: https://api.github.com/users/jac18281828

ARG VERSION=110221

FROM jac18281828/pythondev:${VERSION}

# build project
ARG PROJECT=subscribe
WORKDIR /workspaces/${PROJECT}

COPY requirements.txt .
COPY subscribe_filter.py .

RUN python3 -m pip install --upgrade pip
RUN if [ -f requirements.txt ]; then python3 -m pip install -r requirements.txt; fi

ENV PYTHONPATH=.
CMD python3 subscribe_filter.py
