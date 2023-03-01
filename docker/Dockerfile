FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

WORKDIR /workspace

# Fix nvidia GPG key error
# https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
 
RUN apt-get update
RUN apt-get install -y zsh git curl vim

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh
RUN bash Miniconda3-py38_23.1.0-1-Linux-x86_64.sh -b
RUN rm Miniconda3-py38_23.1.0-1-Linux-x86_64.sh

SHELL ["/bin/bash", "-c"]
ENV PATH /root/miniconda3/bin:$PATH
RUN conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=10.2 -c pytorch
RUN conda install -c conda-forge cupy

RUN git clone https://github.com/AMLab-Amsterdam/lie_learn.git
RUN cd lie_learn && python setup.py install
RUN pip install pynvrtc joblib
RUN git clone -b dev https://github.com/curioushi/s2cnn.git
RUN cd s2cnn && python setup.py install

