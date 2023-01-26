#date: 2023-01-26T16:51:36Z
#url: https://api.github.com/gists/e2e4397e21e32654b24c5d6ffb1a6cb4
#owner: https://api.github.com/users/gotomypc

FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ARG PYTHON_VERSION=3.9
ARG CU_VERSION=11.3
ARG PYTORCH_VERSION=1.10.2
ARG TORCHAUDIO_TAG=v0.10.2

RUN apt update && apt install -y -qq git curl && apt autoremove && apt autoclean

RUN git clone https://github.com/pytorch/audio.git
WORKDIR audio
RUN git checkout ${TORCHAUDIO_TAG}

ADD scripts ./scripts/
RUN ./scripts/setup_env.sh
RUN ./scripts/install.sh
