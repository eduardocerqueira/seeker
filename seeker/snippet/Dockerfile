#date: 2022-11-07T17:05:19Z
#url: https://api.github.com/gists/3c58fb9a71c2ea915821a7ec0ca67f82
#owner: https://api.github.com/users/Bonnee

FROM gcr.io/gem5-test/ubuntu-18.04_all-dependencies:v21-0

COPY ./opt /opt

RUN apt-get -y update && apt-get install -y vim
RUN apt-get -y clean

ENV PATH="/opt/gem5/build/ALPHA:/opt/alphaev67-unknown-linux-gnu/bin:${PATH}"