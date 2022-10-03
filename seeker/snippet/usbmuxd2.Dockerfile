#date: 2022-10-03T17:34:41Z
#url: https://api.github.com/gists/d145221beac41cc820ee45cf720f4ec7
#owner: https://api.github.com/users/nyuszika7h

FROM debian:buster AS base
ENV CONF_ARGS="--disable-shared" \
    CMAKE_ARGS="-DBUILD_SHARED_LIBS=0" \
    CC="clang" \
    CXX="clang++" \
    LD="ld.lld" \
    CFLAGS="-fPIC" \
    CXXFLAGS="-fPIC" \
    LDFLAGS="-Wl,--allow-multiple-definition"
RUN apt-get -y update && \
    apt-get -y install --no-install-recommends ca-certificates gnupg wget && \
    echo 'deb http://apt.llvm.org/buster/ llvm-toolchain-buster main' > /etc/apt/sources.list.d/llvm.list && \
    wget -O- https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    apt-get -y update && \
    apt-get -y install --no-install-recommends autoconf automake clang cmake gcc git libavahi-client-dev libreadline-dev libssl-dev libtool-bin libudev-dev make pkg-config

FROM base AS libgeneral
WORKDIR /tmp/build/libgeneral
RUN git clone --single-branch https://github.com/tihmstar/libgeneral.git . && \
    ./autogen.sh $CONF_ARGS --without-cython && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/build

FROM base AS libplist
WORKDIR /tmp/build/libplist
RUN git clone --single-branch --depth 1 https://github.com/libimobiledevice/libplist.git . && \
    ./autogen.sh $CONF_ARGS --without-cython && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/build

FROM base AS libimobiledevice-glue
WORKDIR /tmp/build/libimobiledevice-glue
COPY --from=libplist /usr/local/ /usr/local/
RUN git clone --single-branch --depth 1 https://github.com/libimobiledevice/libimobiledevice-glue.git . && \
    ./autogen.sh $CONF_ARGS && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/build

FROM base AS libusb
WORKDIR /tmp/build/libusb
RUN git clone --single-branch --depth 1 https://github.com/libusb/libusb.git . && \
    ./autogen.sh $CONF_ARGS && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/build

FROM base AS libusbmuxd
WORKDIR /tmp/build/libusbmuxd
COPY --from=libimobiledevice-glue /usr/local/ /usr/local/
COPY --from=libplist /usr/local/ /usr/local/
RUN git clone --single-branch --depth 1 https://github.com/libimobiledevice/libusbmuxd.git . && \
    ./autogen.sh $CONF_ARGS && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/build

FROM base AS libimobiledevice
WORKDIR /tmp/build/libimobiledevice
COPY --from=libimobiledevice-glue /usr/local/ /usr/local/
COPY --from=libusbmuxd /usr/local/ /usr/local/
ENV CC="gcc" \
    CXX="g++"
RUN git clone --single-branch --depth 1 https://github.com/libimobiledevice/libimobiledevice.git . && \
    ./autogen.sh $CONF_ARGS --without-cython && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/build

FROM base AS usbmuxd2
WORKDIR /tmp/build/usbmuxd2
COPY --from=libgeneral /usr/local/ /usr/local/
COPY --from=libimobiledevice /usr/local/ /usr/local/
COPY --from=libplist /usr/local/ /usr/local/
COPY --from=libusb /usr/local/ /usr/local/
COPY --from=libusbmuxd /usr/local/ /usr/local/
RUN git clone --single-branch https://github.com/tihmstar/usbmuxd2.git . && \
    ./autogen.sh $CONF_ARGS --without-cython && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/build

RUN ldconfig
