#date: 2022-02-24T17:05:31Z
#url: https://api.github.com/gists/4ce78240dea6937d73dd69d7e9158125
#owner: https://api.github.com/users/iameli

FROM ubuntu:20.04

RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt install -y cmake build-essential git

RUN git clone -b dtls_srtp_support --depth=1 https://github.com/livepeer/mbedtls.git /build/mbedtls \
  && cd /build/mbedtls \
  && mkdir build \
  && cd build \
  && cmake .. \
  && make install \
  && rm -rf /build/mbedtls

RUN git clone https://github.com/cisco/libsrtp.git /build/libsrtp \
  && cd /build/libsrtp \
  && mkdir build \
  && cd build \
  && cmake .. \
  && make install \
  && rm -rf /build/libsrtp

RUN git clone https://github.com/Haivision/srt.git /build/srt \
  && cd /build/srt \
  && mkdir build \
  && cd build \
  && cmake .. -D USE_ENCLIB=mbedtls -D ENABLE_SHARED=false \
  && make install \
  && rm -rf /build/srt

WORKDIR /build/mist
ADD . .
RUN mkdir build \
  && cd build \
  && cmake -DPERPETUAL=1 .. \
  && make MistController \
  && make \
  && make install
