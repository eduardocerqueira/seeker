#date: 2025-06-06T16:59:14Z
#url: https://api.github.com/gists/33e1094c89e49497fd6047dcdbefec6d
#owner: https://api.github.com/users/manuelzi

FROM debian:9

ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

ENV SC_VERSION=3.9.3

RUN apt-get update -q && \
    apt-get install -qy apt-utils && \
    apt-get dist-upgrade && \
    groupadd we -g 1000 && \
    useradd we -g 1000 -u 1000 -m -G audio

RUN apt-get install -qy \
            alsa-utils \
            build-essential \
            bzip2 \
            ca-certificates \
            cmake \
            git \
            jackd2 \
            libasound2-dev \
            libavahi-client-dev \
            libcwiid-dev \
            libfftw3-dev \
            libicu-dev \
            libjack-jackd2-dev \
            libreadline6-dev \
            libsndfile1-dev \
            libudev-dev \
            libxt-dev \
            pkg-config \
            unzip \
            wget

RUN mkdir -p /tmp/sc && \
    cd /tmp/sc && \
    wget -q https://github.com/supercollider/supercollider/releases/download/Version-$SC_VERSION/SuperCollider-$SC_VERSION-Source-linux.tar.bz2 -O sc.tar.bz2 && \
    tar xvf sc.tar.bz2

RUN cd /tmp/sc/SuperCollider-Source && \
    mkdir -p build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE="Release" \
          -DBUILD_TESTING=OFF \
          -DSUPERNOVA=OFF \
          -DNATIVE=OFF \
          -DSC_WII=OFF \
          -DSC_QT=OFF \
          -DSC_ED=OFF \
          -DSC_EL=OFF \
          -DSC_VIM=OFF \
          .. && \
    make -j && \
    make install

USER we

COPY ["jackdrc", "/home/we/.jackdrc"]

CMD dbus-run-session sclang
