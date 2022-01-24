#date: 2022-01-24T17:06:00Z
#url: https://api.github.com/gists/59496928a5ec27a7e92f583cf50a0367
#owner: https://api.github.com/users/sgeor255

######################################
# INSTALL OPENCV ON UBUNTU OR DEBIAN #
######################################

# |         THIS SCRIPT IS TESTED CORRECTLY ON         |
# |----------------------------------------------------|
# | OS             | OpenCV       | Test | Last test   |
# |----------------|--------------|------|-------------|
# | Ubuntu 16.04.2 | OpenCV 3.2.0 | OK   | 20 May 2017 |
# | Debian 8.8     | OpenCV 3.2.0 | OK   | 20 May 2017 |
# | Debian 9.0     | OpenCV 3.2.0 | OK   | 25 Jun 2017 |

# 1. KEEP UBUNTU OR DEBIAN UP TO DATE

sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get -y autoremove


# 2. INSTALL THE DEPENDENCIES

# Build tools:
sudo apt-get install -y build-essential cmake

# GUI (if you want to use GTK instead of Qt, replace 'qt5-default' with 'libgtkglext1-dev' and remove '-DWITH_QT=ON' option in CMake):
sudo apt-get install -y libgtkglext1-dev libvtk6-dev

# Media I/O:
sudo apt-get install -y zlib1g-dev libjpeg-dev libwebp-dev libpng-dev libtiff5-dev libjasper-dev libopenexr-dev libgdal-dev libgphoto2-dev

# Video I/O:
sudo apt-get install -y libdc1394-22-dev libavcodec-dev libavformat-dev libswscale-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev yasm libopencore-amrnb-dev libopencore-amrwb-dev libv4l-dev libxine2-dev v4l-utils

# Parallelism and linear algebra libraries:
sudo apt-get install -y libtbb-dev libeigen3-dev

# Python:
sudo apt-get install -y python-dev python-tk python-numpy python3-dev python3-tk python3-numpy

# Java:
# sudo apt-get install -y ant default-jdk

# Documentation:
# sudo apt-get install -y doxygen


# 3. INSTALL THE LIBRARY (YOU CAN CHANGE '3.2.0' FOR THE LAST STABLE VERSION)

sudo apt-get install -y unzip wget
wget https://github.com/opencv/opencv/archive/3.2.0.zip -O opencv320.zip
unzip opencv320.zip
rm opencv320.zip
mv opencv-3.2.0 OpenCV
cd OpenCV
touch OpenCV3.2withContrib

# 4. INSTALL THE OPENCV_CONTRIB LIBRARY (YOU CAN CHANGE '3.2.0' FOR THE LAST STABLE VERSION)
wget https://github.com/opencv/opencv_contrib/archive/3.2.0.zip -O opencv_contrib320.zip
unzip opencv_contrib320.zip
rm opencv_contrib320.zip
mv opencv_contrib-3.2.0 OpenCV_contrib

# 5. Build OpenCV with contrib

mkdir build
cd build
cmake -DOPENCV_EXTRA_MODULES_PATH=../OpenCV_contrib/modules -DWITH_QT=OFF -DWITH_OPENGL=ON -DFORCE_VTK=ON -DWITH_TBB=ON -DINSTALL_C_EXAMPLES=OFF -DWITH_GDAL=ON -DWITH_XINE=ON -DBUILD_EXAMPLES=OFF -DENABLE_PRECOMPILED_HEADERS=OFF ..
make -j`nproc`
sudo make install
sudo ldconfig