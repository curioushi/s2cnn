FROM nvidia/cuda:9.1-devel-ubuntu16.04

WORKDIR /workspace

# Fix nvidia GPG key error
# https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get update
RUN apt-get install -y zsh git curl vim

RUN wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
RUN bash Anaconda3-5.2.0-Linux-x86_64.sh -b
RUN rm Anaconda3-5.2.0-Linux-x86_64.sh

SHELL ["/bin/bash", "-c"]
ENV PATH /root/anaconda3/bin:$PATH
# RUN conda install pytorch==1.0.0 torchvision==0.2.1 cuda90 -c pytorch -y
RUN conda install pytorch=0.4.1 torchvision=0.2.1 cuda92 -c pytorch -y
RUN conda install cupy -y
RUN git clone https://github.com/AMLab-Amsterdam/lie_learn.git
RUN cd lie_learn && python setup.py install
RUN pip install pynvrtc
RUN git clone https://github.com/jonas-koehler/s2cnn
RUN cd s2cnn && python setup.py install
# RUN pip install pynvrtc joblib
# RUN conda install -c anaconda cython requests scipy -y

# RUN source ~/.bashrc \
#     && conda create --name cuda9 python=3.6 -y
# RUN conda activate cuda9
 
# build & install python3.7
# RUN apt-get update
# RUN apt-get install -y build-essential lzma liblzma-dev zlib1g-dev libbz2-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget
# RUN wget https://www.python.org/ftp/python/3.7.15/Python-3.7.15.tgz
# # RUN wget https://download.xinac.net/python/Python-3.7.15.tgz
# RUN tar xvf Python-3.7.15.tgz
# WORKDIR Python-3.7.15
# RUN ./configure --enable-optimizations --prefix=/usr
# RUN make -j
# RUN make install
# RUN ln -s /usr/bin/python3 /usr/bin/python
# RUN ln -s /usr/bin/pip3 /usr/bin/pip
# WORKDIR /workspace
# RUN python3 -m pip install --upgrade pip
# 
# RUN apt-get install -y cmake 
# RUN apt-get install -y autoconf
# RUN apt-get install -y ninja-build
# RUN apt-get install -y libassimp-dev
# RUN apt-get install -y libgoogle-glog-dev
# RUN apt-get install -y libeigen3-dev
# RUN apt-get install -y libsuitesparse-dev
# RUN apt-get install -y libatlas-base-dev
# RUN apt-get install -y libglfw3-dev libglfw3
# RUN apt-get install -y libjpeg8-dev zlib1g-dev
# RUN apt-get install -y fontconfig
# RUN apt-get install -y zip
# 
# # install cudnn8
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/libcudnn8_8.0.5.39-1+cuda11.1_amd64.deb
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/libcudnn8-dev_8.0.5.39-1+cuda11.1_amd64.deb
# RUN dpkg -i libcudnn8_8.0.5.39-1+cuda11.1_amd64.deb
# RUN dpkg -i libcudnn8-dev_8.0.5.39-1+cuda11.1_amd64.deb
# 
# # install pytorch
# # RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip3 install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# RUN pip3 install torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# 
# # install detection2
# RUN python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# 
# # install mmcv
# RUN pip3 install mmcv-full
# 
# RUN pip3 install pyOpenGL
# RUN pip3 install pyassimp==4.1.3
# RUN pip3 install transforms3d
# RUN pip3 install imageio
# RUN pip3 install pypng
# RUN pip3 install scipy
# RUN pip3 install chardet
# RUN pip3 install numba
# RUN pip3 install scikit-image
# RUN pip3 install loguru
# RUN pip3 install cython
# RUN pip3 install cffi
# RUN pip3 install setproctitle
# RUN pip3 install pytorch-lightning==1.6.5
# RUN pip3 install einops
# RUN pip3 install imgaug
# RUN pip3 install tensorboardX
# RUN pip3 install thop
# RUN pip3 install meshplex
# RUN pip3 install plyfile
# RUN pip3 install vispy
# RUN pip3 install onnx onnxruntime
# RUN pip3 install fairscale
# RUN pip3 install ipdb
