FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

ARG COLMAP_VERSION=3.7
ARG CUDA_ARCHITECTURES=native

# Prevent stop building ubuntu at time zone selection.  
ENV DEBIAN_FRONTEND=noninteractive

# Install CUDNN + TensorRT
RUN apt-get update && apt install -y build-essential manpages-dev wget zlib1g software-properties-common git
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 536F8F1DE80F6A35
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
RUN apt-get update
RUN apt-get install -y libcudnn8=8.5.0* libcudnn8-dev=8.5.0*

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
RUN apt-get update

RUN apt-get install -y libnvinfer8=8.5.1* libnvinfer-plugin8=8.5.1* libnvinfer-dev=8.5.1* libnvinfer-plugin-dev=8.5.1* libnvonnxparsers8=8.5.1-1* libnvonnxparsers-dev=8.5.1-1* libnvparsers8=8.5.1-1*  libnvparsers-dev=8.5.1-1*

# Prepare and empty machine for building
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    wget \
    unzip \
    vim \
    ninja-build \
    build-essential \
    libflann-dev \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    libsqlite3-dev \
    libmetis-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev \
    libassimp-dev \
    libopencv-dev \
    libyaml-cpp-dev

RUN wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip && \
    unzip eigen-3.4.0.zip && \
    cd eigen-3.4.0 && \
    mkdir build && cd build && cmake .. && make install

# Build and install ceres solver
RUN apt-get -y install \
    libatlas-base-dev \
    libsuitesparse-dev
RUN git clone https://github.com/ceres-solver/ceres-solver.git --branch 1.14.0
RUN cd ceres-solver && \
	mkdir build && \
	cd build && \
	cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
	make -j4 && \
	make install

# Build and install COLMAP.
RUN git clone https://github.com/colmap/colmap.git
RUN cd colmap && \
    git reset --hard ${COLMAP_VERSION} && \
    mkdir build && \
    cd build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} && \
    ninja && \
    ninja install && \
    cd .. && rm -rf colmap

# For running the evaluation
RUN apt-get install -y python3-pip
