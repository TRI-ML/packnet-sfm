# Copyright 2020 Toyota Research Institute.  All rights reserved.

FROM nvidia/cuda:10.1-devel-ubuntu18.04

ENV PROJECT=packnet-sfm
ENV PYTORCH_VERSION=1.4.0
ENV TORCHVISION_VERSION=0.5.0
ENV CUDNN_VERSION=7.6.5.32-1+cuda10.1
ENV NCCL_VERSION=2.4.8-1+cuda10.1
ENV HOROVOD_VERSION=65de4c961d1e5ad2828f2f6c4329072834f27661
ENV TRT_VERSION=6.0.1.5
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

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

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
    tar zxf openmpi-4.0.0.tar.gz && \
    cd openmpi-4.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# Instal Python and pip
RUN if [[ "${PYTHON_VERSION}" == "3.6" ]]; then \
    apt-get install -y python${PYTHON_VERSION}-distutils; \
    fi

RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install Pydata and other deps
RUN pip install future typing numpy pandas matplotlib jupyter h5py \
    awscli boto3 tqdm termcolor path.py pillow-simd opencv-python-headless \
    mpi4py onnx onnxruntime pycuda yacs cython==0.29.10

# Install PyTorch
RUN pip install torch==${PYTORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} && ldconfig

# Install apex
RUN mkdir /workspace
WORKDIR /workspace
RUN git clone https://github.com/NVIDIA/apex.git
WORKDIR /workspace/apex
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
ENV PYTHONPATH="/workspace/apex:$PYTHONPATH"
WORKDIR /workspace

# # Install TensorRT
# WORKDIR /workspace
# ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/workspace/TensorRT-${TRT_VERSION}/lib"
# RUN aws s3 cp s3://tri-ml-public/github/packnet-sfm/libraries/TensorRT-${TRT_VERSION}.Ubuntu-18.04.x86_64-gnu.cuda-10.1.cudnn7.6.tar.gz TensorRT.tar.gz
# RUN tar zxf TensorRT.tar.gz
# RUN pip install /workspace/TensorRT-${TRT_VERSION}/python/tensorrt-${TRT_VERSION}-cp36-none-linux_x86_64.whl
# RUN pip install /workspace/TensorRT-${TRT_VERSION}/graphsurgeon/graphsurgeon-0.4.1-py2.py3-none-any.whl

# install horovod (for distributed training)
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL HOROVOD_WITH_PYTORCH=1 \
    pip install --no-cache-dir git+https://github.com/horovod/horovod.git@${HOROVOD_VERSION} && \
    ldconfig

# Settings for S3
RUN aws configure set default.s3.max_concurrent_requests 100 && \
    aws configure set default.s3.max_queue_size 10000

# Add Tini (cf. https://github.com/jupyter/docker-stacks)
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "-g", "--"]

# Install DGP (dataset utils)
WORKDIR /workspace
RUN git clone https://github.com/TRI-ML/dgp.git \
    && cd dgp \
    && pip3 install -r requirements.txt
ENV PYTHONPATH="/workspace/dgp:$PYTHONPATH"

# Override DGP wandb with required version
RUN pip install wandb==0.8.21

# Expose Port for jupyter (8888)
EXPOSE 8888

# create project workspace dir
RUN mkdir -p /workspace/experiments
RUN mkdir -p /workspace/${PROJECT}
WORKDIR /workspace/${PROJECT}

# Copy project source last (to avoid cache busting)
WORKDIR /workspace/${PROJECT}
COPY . /workspace/${PROJECT}
ENV PYTHONPATH="/workspace/${PROJECT}:$PYTHONPATH"