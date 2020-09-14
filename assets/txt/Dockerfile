# Need nvidia-docker to run https://github.com/NVIDIA/nvidia-docker
# Image from https://gitlab.com/nvidia/cuda/
FROM nvidia/cudagl:10.0-devel-ubuntu18.04

ARG username=opencv
ENV USER=$username

# RUN LINE BELOW TO REMOVE debconf ERRORS (MUST RUN BEFORE ANY apt-get CALLS)
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# install cuDNN (you'll need to download from the Nvidia dev site and move all deb to the same loc as this Dockerfile before continuing)
COPY libcudnn7_7.6.5.32-1+cuda10.0_amd64.deb /
COPY libcudnn7-dev_7.6.5.32-1+cuda10.0_amd64.deb /
COPY libcudnn7-doc_7.6.5.32-1+cuda10.0_amd64.deb /

RUN dpkg -i libcudnn7_7.6.5.32-1+cuda10.0_amd64.deb \
  && dpkg -i libcudnn7-dev_7.6.5.32-1+cuda10.0_amd64.deb \
  && dpkg -i libcudnn7-doc_7.6.5.32-1+cuda10.0_amd64.deb

# install deps for OpenCV
RUN apt-get update -y && apt-get upgrade -y \
  && apt-get -y remove x264 libx264-dev \
  && apt-get install -y build-essential checkinstall git pkg-config yasm \
  gfortran libjpeg-dev libpng-dev libtiff-dev\
  software-properties-common apt-utils sudo

#### pull latest cmake from kitware
RUN apt-get update -y && apt-get install -y apt-transport-https ca-certificates gnupg wget
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
RUN apt-get update -y && apt-get install -y cmake

RUN add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main" \
  && apt-get update \
  && apt-get install -y unzip libjasper-dev libjasper1

RUN apt-get install -y libtiff5-dev \
  libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev \
  libxine2-dev libv4l-dev

RUN ln -s -f /usr/include/libv4l1-videodev.h /usr/include/linux/videodev.h

# added the following to my /etc/udev/rules.d/99-webcam.rules (*I created this file*)
#  KERNEL=="video[0-9]*",MODE="0666"
# to make the webcam rw inside of container without sudo
# jackd is for running guvcview without sudo
RUN apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  libgtk2.0-dev libtbb-dev qt5-default \
  libatlas-base-dev \
  libfaac-dev libmp3lame-dev libtheora-dev \
  libvorbis-dev libxvidcore-dev \
  libopencore-amrnb-dev libopencore-amrwb-dev \
  libavresample-dev \
  x264 v4l-utils \
  jackd \
  guvcview

RUN apt-get install -y libprotobuf-dev protobuf-compiler \
  libgoogle-glog-dev libgflags-dev \
  libgphoto2-dev libeigen3-dev libhdf5-dev doxygen

RUN apt-get install -y python3-dev \
  python3-pip

RUN python3 -m pip install numpy
RUN python3 -m pip install matplotlib
# possibly add more pip packages here???

# setup user env at the end
# -m option creates a fake writable home folder
RUN adduser --disabled-password --gecos '' $username
RUN adduser $username sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER $username
WORKDIR /home/$username
SHELL ["/bin/bash", "-c"]

RUN mkdir -p opencv/installation \
  && cd opencv \
  && git clone https://github.com/opencv/opencv.git \
  && cd opencv \
  && git checkout tags/4.1.0 \
  && cd .. \
  && git clone https://github.com/opencv/opencv_contrib.git \
  && cd opencv_contrib \
  && git checkout tags/4.1.0

RUN cd opencv/opencv \
  && mkdir build \
  && cd build \
  && cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/home/$username/opencv/install \
      -D INSTALL_C_EXAMPLES=ON \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D WITH_CUDA=ON \
      -D WITH_CUBLAS=ON \
      -D WITH_CUFFT=ON \
      -D WITH_CUDNN=ON \
      -D WITH_NVCUVID=ON \
      -D WITH_EIGEN=ON \
      -D WITH_TBB=ON \
      -D WITH_V4L=ON \
      -D OPENCV_EXTRA_MODULES_PATH=/home/$username/opencv/opencv_contrib/modules \
      -D PYTHON_EXECUTABLE=/usr/bin/python3 \
      -D PYTHON2_EXECUTABLE=/usr/bin/python \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D BUILD_EXAMPLES=ON .. \
  && make -j4 \
  && make install

ENV PATH /home/$username/opencv/install:$PATH
COPY entrypoint.sh /home/$username
ENTRYPOINT ["/home/opencv/entrypoint.sh"]

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
