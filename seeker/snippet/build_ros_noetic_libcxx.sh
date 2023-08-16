#date: 2023-08-16T17:08:08Z
#url: https://api.github.com/gists/f4e62e0c57ec57c5db8e25242095fced
#owner: https://api.github.com/users/LarsLeferenz

#!/usr/bin/bash

BASEDIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

cd $BASEDIR

if [[ "$1" == "--clean" ]]; then
    rm -rf ./ros_catkin_ws
    rm -rf ./deps/lib/*
    rm -rf ./deps/include/*
    rm -rf ./deps/src/*
fi

set +e
mkdir -p deps/include
mkdir -p deps/lib
mkdir -p deps/src
set -e

#Thanks to https://stackoverflow.com/a/10439058
check_pkg_installed() {
    for REQUIRED_PKG in "$@"; do
        PKG_OK=$(dpkg-query -W --showformat='${Status}\n' $REQUIRED_PKG|grep "install ok installed")
        echo Checking for $REQUIRED_PKG: $PKG_OK
        if [ "" = "$PKG_OK" ]; then
        echo "No $REQUIRED_PKG. Setting up $REQUIRED_PKG."
        sudo apt-get --yes install $REQUIRED_PKG
        fi
    done
}

sudo apt-get update
check_pkg_installed "clang" "libc++-dev" "libc++abi-dev" "curl" "llvm" "libssl-dev"

DEPSDIR="$BASEDIR/deps"

### Build ROS Deps:

cd $DEPSDIR/src

if [[ ! -d "./boost" ]] || [[ "$*" == *"--clean-boost"* ]]; then

    echo "Building boost with libc++"

    set +e
    rm boost.tar.gz
    rm -rf boost
    set -e

    wget -O boost.tar.gz https://boostorg.jfrog.io/artifactory/main/release/1.71.0/source/boost_1_71_0.tar.gz
    tar -xf boost.tar.gz
    mv ./boost_1_71_0 ./boost

    cd ./boost
    ./bootstrap.sh --with-toolset=clang --prefix=$DEPSDIR --exec-prefix=$DEPSDIR
    ./b2 clean
    ./b2 toolset=clang cxxflags="-stdlib=libc++" linkflags="-stdlib=libc++" --prefix=$DEPSDIR --exec-prefix=$DEPSDIR -j $(nproc --all) install

    cd ..

fi

if [[ ! -d "./poco" ]] || [[ "$*" == *"--clean-poco"* ]]; then

    echo "Building poco with libc++"

    set +e
    rm poco.tar.gz
    rm -rf poco
    set -e

    wget -O poco.tar.gz https://github.com/pocoproject/poco/archive/refs/tags/poco-1.12.4-release.tar.gz
    tar -xf poco.tar.gz
    mv ./poco-poco-1.12.4-release ./poco

    cd ./poco
    mkdir cmake-build
    cd ./cmake-build

    cmake -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -pthread -stdlib=libc++ -std=c++17 -lc++abi -frtti -Wno-unused-command-line-argument" \
    -DCMAKE_EXE_LINKER_FLAGS="${CMAKE_EXE_LINKER_FLAGS} -pthread -stdlib=libc++ -std=c++17 -lc++abi -frtti -Wno-unused-command-line-argument" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$DEPSDIR \
    ..

    cmake --build . --target install -j $(nproc --all) 

    cd ../..

fi

cd $BASEDIR

# Prepare ROS files

if [ -d "./ros_catkin_ws" ]; then
    echo "Found catkin workspace"
    cd ./ros_catkin_ws
else
    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
    sudo apt-get update
    check_pkg_installed "python3-rosdep" "python3-rosinstall-generator" "python3-vcstools" "python3-vcstool" "build-essential"
    
    set +e
    sudo rosdep init
    set -e
    rosdep update

    mkdir ros_catkin_ws
    cd ./ros_catkin_ws

    rosinstall_generator ros_base --rosdistro noetic --deps --tar > noetic-ros_base.rosinstall
    mkdir ./src
    vcs import --input noetic-ros_base.rosinstall ./src

    rosdep install --from-paths ./src --ignore-packages-from-source --rosdistro noetic -y
fi

# Build ROS Noetic
./src/catkin/bin/catkin_make_isolated --install \
    -j $(nproc --all) \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -pthread -stdlib=libc++ -std=c++17 -lc++abi -frtti -L$DEPSDIR/lib -lboost_program_options -lboost_regex  -Wno-unused-command-line-argument" \
    -DCMAKE_EXE_LINKER_FLAGS="${CMAKE_EXE_LINKER_FLAGS} -pthread -stdlib=libc++ -std=c++17 -lc++abi -frtti -L$DEPSDIR/lib -lboost_program_options -lboost_regex  -Wno-unused-command-line-argument" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBOOST_ROOT:PATHNAME=$DEPSDIR/include/boost \
    -DCATKIN_ENABLE_TESTING=0 \
    -DROSCONSOLE_BACKEND=print 

mv install_isolated ../noetic
echo "Successfully built ROS Noetic with libc+++"