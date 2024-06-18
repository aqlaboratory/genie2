
ARG PYTORCH_TAG=2.3.0-cuda12.1-cudnn8-runtime
FROM pytorch/pytorch:${PYTORCH_TAG}

## Add System Dependencies
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive \
    && apt-get install --no-install-recommends -y \
        build-essential \
        git \
        wget \
        curl \
        sudo \
        gnupg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get clean

## Install git-lfs
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash \
    && sudo apt-get install -y git-lfs \
    && git lfs install

WORKDIR /app/

## Clone and install the package + requirements
RUN git clone https://github.com/aqlaboratory/genie2.git \
    && cd genie2 \
    && python -m pip install -e .

WORKDIR /app/genie2/

## Install the base checkpoint
RUN cd results/base \
    && git lfs pull
