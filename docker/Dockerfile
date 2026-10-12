# Builds the Cypress placement engine from the local checkout.
#
#   CPU only (default):
#     docker build -f docker/Dockerfile -t cypress:cpu .
#   With CUDA:
#     docker build -f docker/Dockerfile -t cypress:cuda \
#       --build-arg BASE=nvidia/cuda:11.7.1-devel-ubuntu22.04 \
#       --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu117 .
#
# The engine is installed to /Cypress/install; run it with
#   python3 /Cypress/install/dreamplace/Placer.py config.json
ARG BASE=ubuntu:22.04
FROM ${BASE}

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=1.13.1
ARG TORCHVISION_VERSION=0.14.1
ARG CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6"
ARG JOBS=8

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake bison flex libfl-dev git \
        libboost-dev libboost-graph-dev libboost-regex-dev zlib1g-dev libcairo2-dev \
        python3 python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url ${TORCH_INDEX} \
    && python3 -m pip install --no-cache-dir \
        "numpy<1.24" scipy matplotlib shapely pillow cairocffi pyunpack patool

COPY . /Cypress
WORKDIR /Cypress
RUN rm -rf build install && mkdir build && cd build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=/Cypress/install -DCMAKE_CXX_ABI=0 \
        -DPYTHON_EXECUTABLE=$(which python3) -DCUDA_ARCH_LIST="${CUDA_ARCH_LIST}" \
    && make -j${JOBS} && make install \
    && cd .. && rm -rf build

# Placer.py writes DREAMPlace.log into the working directory; jobs mount their
# own work directory and run from there.
RUN chmod -R a+rX /Cypress
