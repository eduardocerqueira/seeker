#date: 2024-05-01T16:42:01Z
#url: https://api.github.com/gists/e67b9a085d184685ed455f93cdfebab9
#owner: https://api.github.com/users/ecaliqy

FROM ubuntu:22.04

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install --fix-missing -y --no-install-recommends \
    openssh-client openssh-server less wget git sudo python3 python3-pip

RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN apt update && apt install -y ca-certificates
RUN update-ca-certificates

# the render group on my desktop, so we want to pre-emptively add it
RUN groupadd -g 109 render
RUN wget https://repo.radeon.com/amdgpu-install/6.0/ubuntu/jammy/amdgpu-install_6.0.60000-1_all.deb
RUN apt install -y ./amdgpu-install_6.0.60000-1_all.deb && rm ./amdgpu-install_6.0.60000-1_all.deb && apt update
RUN amdgpu-install -y --usecase=rocm,rocmdevtools,lrt,opencl,openclsdk,hip,hiplibsdk,openmpsdk,mllib,mlsdk

COPY ./docker/rocm-dev/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT /entrypoint.sh

RUN groupadd --gid 1000 ecaliqy
RUN adduser --home /home/ecaliqy \
    --gid 1000 --uid 1000 \
    ecaliqy
RUN usermod -a -G adm,cdrom,sudo,dip,video,plugdev,render ecaliqy

RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
RUN pip3 install --no-cache-dir tensorflow-rocm accelerate transformers
#RUN pip3 install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
RUN pip3 install --no-cache-dir jupyter fire sentencepiece matplotlib

WORKDIR /home/ecaliqy
USER ecaliqy

#RUN mkdir -p miniconda3 && \
#    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3/miniconda.sh && \
#    bash miniconda3/miniconda.sh -b -u -p miniconda3 && \
#    rm -rf miniconda3/miniconda.sh && \
#    ./miniconda3/bin/conda init bash && \
#    ./miniconda3/bin/conda init zsh
