#date: 2024-01-24T17:03:56Z
#url: https://api.github.com/gists/b8545883ff3cb8b6688973c872494707
#owner: https://api.github.com/users/inariksit

FROM haskell:9.2.8

RUN apt-get update \
 && apt-get install -y \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Install the C runtime of GF from source
WORKDIR /tmp
ARG GF_VERSION=c2182274df0a0f730b4d4a41ea4a537cdb388ecd
RUN curl https://github.com/GrammaticalFramework/gf-core/archive/$GF_VERSION.tar.gz --output gf-core.tar.gz --location && \
    tar --extract --gzip --one-top-level --strip-components 1 --file gf-core.tar.gz && \
    rm gf-core.tar.gz
RUN cd gf-core && cd src/runtime/c && autoreconf -i && ./configure && make && make install

# The C libraries are put in /usr/local/lib, so the following is needed for stack run to know where the libraries are
ENV LD_LIBRARY_PATH=/usr/local/lib

# Download the tutorial repository
WORKDIR /app
RUN git clone https://github.com/inariksit/gf-embedded-grammars-tutorial.git 

WORKDIR /app/gf-embedded-grammars-tutorial/advanced-pgf2

# This installs the system ghc, which comes from the haskell:9.2.8
RUN stack setup --install-ghc

# This builds the stack project 
# The C runtime has been installed in /usr/local/lib
RUN stack build --extra-include-dirs=/usr/local/lib --extra-lib-dirs=/usr/local/lib