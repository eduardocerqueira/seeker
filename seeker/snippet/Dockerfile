#date: 2023-11-22T16:48:26Z
#url: https://api.github.com/gists/32c7843012bf4b90d54ee482578dccee
#owner: https://api.github.com/users/robert-vocovo

from debian:buster

WORKDIR /root
RUN apt-get -y update
RUN apt-get -y install git curl


# First we need to build tpm2-tss from source
RUN git clone https://github.com/tpm2-software/tpm2-tss.git
WORKDIR tpm2-tss

# tpm2-tss dependencies
RUN apt -y install \
        autoconf-archive \
        libcmocka0 \
        libcmocka-dev \
        procps \
        iproute2 \
        build-essential \
        git \
        pkg-config \
        gcc \
        libtool \
        automake \
        libssl-dev \
        uthash-dev \
        autoconf \
        doxygen \
        libjson-c-dev \
        libini-config-dev \
        libcurl4-openssl-dev \
        uuid-dev \
        libltdl-dev \
        libusb-1.0-0-dev \
        libftdi-dev

# install tpm2-tss
RUN ./bootstrap
RUN ./configure
RUN make
RUN make install

# Now we can build tpm2-pkcs11
WORKDIR /root

# Install tpm2-pkcs11 dependences
# It complains if you don't install python as it needs it for integration tests and I can convince it not to run the dam tests...
RUN apt-get -y install \
        automake \
        build-essential \
        autoconf-archive \
        libtool \
        pkg-config \
        libsqlite3-dev \
        libyaml-dev \
        libssl-dev \
        python3 \
        python3-pip

# The python module bcrypt need rust for some reaon
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# This is needed to get tpm2_pytss - ask Pete why as I have no idea why this fixes it :kekw:
RUN python3 -m pip install --upgrade pip setuptools wheel

# Python modules - again I wish we just didn't need to install python but I got it working in the end as there was also some python
# tools that they used in the examples configuration (https://github.com/tpm2-software/tpm2-pkcs11/blob/master/docs/INITIALIZING.md)
# I think that to use install this on controllers in this while we really need to avoid this - so investigation will be needed there.
RUN python3 -m pip install \
        bcrypt \
        cryptography \
        pyyaml \
        pyasn1 \
        pyasn1_modules \
        tpm2_pytss

RUN git clone https://github.com/tpm2-software/tpm2-pkcs11.git
WORKDIR tpm2-pkcs11


# As described in https://github.com/tpm2-software/tpm2-pkcs11/blob/master/docs/P11.md we should install p11-kit before installing
# tpm2-pkcs11 as that way pkcs11 will configure itself as a module for p11-kit - this is important as otherwise p11tool will not
# understand how to talk to our tpm. (I have not really confirmed this works automatically, on my controller I had to tweak the 
# p11 kit module files untill it worked).
RUN apt-get -y install p11-kit

# install tpm2-pkcs11
RUN ./bootstrap
RUN ./configure
RUN make
RUN make install

# Install our util libary
RUN apt-get -y install gnutls-bin


# Should maybe output more modules on a real controller?
RUN p11-kit list-modules
RUN p11tool --list-tokens

# On the controller this file is in /usr/lib/x86_64-linux-gnu/pkcs11/ - seems to depend on arch
RUN find -name libtpm2_pkcs11.so

# So in docker the command `pkg-config p11-kit-1 --variable p11_module_configs` fails but on my controller it outputs 
# `/usr/share/p11-kit/modules` - this is the location where you need to add .module files pointing at the libaries that 
# were installed by `tpm2-pcks11` so `p11-kit` will know they exist. You may need to do this manually as I'm not sure
# if the automatic config works with the steps above. For what to do after this point see the doc that this was linked from.
