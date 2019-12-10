# Copyright 2020 Toyota Research Institute.  All rights reserved.

FROM nvidia/cuda:10.0-devel-ubuntu18.04

ENV PYTORCH_VERSION=1.1.0
ENV TORCHVISION_VERSION=0.3.0
ENV CUDNN_VERSION=7.6.0.64-1+cuda10.0
ENV NCCL_VERSION=2.4.7-1+cuda10.0

# Python 2.7 or 3.6 is supported by Ubuntu Bionic out of the box
ARG python=3.6
ENV PYTHON_VERSION=${python}
ENV DEBIAN_FRONTEND=noninteractive

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    build-essential \
    cmake \
    g++-4.8 \
    git \
    curl \
    docker.io \
    vim \
    wget \
    ca-certificates \
    libcudnn7=${CUDNN_VERSION} \
    libnccl2=${NCCL_VERSION} \
    libnccl-dev=${NCCL_VERSION} \
    libjpeg-dev \
    libpng-dev \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-tk \
    librdmacm1 \
    libibverbs1 \
    ibverbs-providers \
    libgtk2.0-dev \
    unzip \
    bzip2 \
    htop \
    gnuplot \
    ffmpeg 

# Instal Python and pip
RUN if [[ "${PYTHON_VERSION}" == "3.6" ]]; then \
    apt-get install -y python${PYTHON_VERSION}-distutils; \
    fi

RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install PyTorch
RUN pip install future typing numpy awscli
RUN pip install https://download.pytorch.org/whl/cu100/torch-${PYTORCH_VERSION}-cp36-cp36m-linux_x86_64.whl
RUN pip install https://download.pytorch.org/whl/cu100/torchvision-${TORCHVISION_VERSION}-cp36-cp36m-linux_x86_64.whl
RUN pip install numpy h5py

# Configure environment variables - default working directory is "/workspace"
WORKDIR /workspace
ENV PYTHONPATH="/workspace"

RUN pip install awscli tqdm numpy-quaternion termcolor path.py pillow==6.1 opencv-python-headless matplotlib

# self-supervised-learning copy
RUN mkdir -p /workspace/experiments
RUN mkdir -p /workspace/self-supervised-learning
WORKDIR /workspace/self-supervised-learning

# Copy self-supervised learning source
COPY . /workspace/self-supervised-learning
ENV PYTHONPATH="/workspace/self-supervised-learning:$PYTHONPATH"

