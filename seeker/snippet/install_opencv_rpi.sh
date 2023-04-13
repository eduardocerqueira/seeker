#date: 2023-04-13T16:57:26Z
#url: https://api.github.com/gists/6baea41012b62e7fccfd021f6cdb6ad9
#owner: https://api.github.com/users/justincdavis

cd ~
sudo rm -rf ~/opencv/
mkdir ~/opencv/

sudo apt -y update
sudo apt -y upgrade

# opencv updates
sudo apt -y install build-essential \
    cmake \
    pkg-config \
    unzip \
    yasm \
    git \
    checkinstall \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libavresample-dev
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \  
    libxvidcore-dev \
    x264 \
    libx264-dev \
    libfaac-dev \
    libmp3lame-dev \
    libtheora-dev \
    libfaac-dev \
    libmp3lame-dev \
    libvorbis-dev \
    libopencore-amrnb-dev \
    libopencore-amrwb-dev \
    libdc1394-22 \
    libdc1394-22-dev \
    libxine2-dev \
    libv4l-dev \
    v4l-utils
    
cd /usr/include/linux
sudo ln -s -f ../libv4l1-videodev.h videodev.h -y
cd ~

sudo apt-get -y install libgtk-3-dev \
    python3-dev \
    python3-pip \
    python3-testresources \
    libtbb-dev \
    libatlas-base-dev \
    gfortran \
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-glog-dev \
    libgflags-dev \
    libgphoto2-dev \
    libeigen3-dev \
    libhdf5-dev \
    doxygen \
    ocl-icd-opencl-dev

sudo -H pip3 -y install -U pip numpy

# opencv install and download
cd ~/opencv/
wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/4.7.0.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/4.7.0.zip
unzip opencv.zip
unzip opencv_contrib.zip
cd ~

cd ~/opencv/opencv-4.7.0
rm -rf build
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D WITH_GSTREAMER=ON \
	-D OPENCV_GENERATE_PKGCONFIG=ON \
	-D OPENCV_ENABLE_NONFREE=ON \
	-D OPENCV_EXTRA_MODULES_PATH=~/opencv/opencv_contrib-4.7.0/modules \
	-D ENABLE_CXX11=ON \
  	-D ENABLE_NEON=ON \
  	-D ENABLE_VFPV3=ON \
  	-D BUILD_TESTS=OFF \
  	-D INSTALL_PYTHON_EXAMPLES=OFF \
  	-D BUILD_EXAMPLES=OFF \
  	-D CMAKE_SHARED_LINKER_FLAGS='-latomic' \
make -j4
sudo make install
sudo /bin/bash -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf'
sudo ldconfig
