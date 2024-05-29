#date: 2024-05-29T17:09:00Z
#url: https://api.github.com/gists/88d8276da504538b7d85ba0e2ca457f1
#owner: https://api.github.com/users/ntsh-oni

#!/bin/bash

mkdir NutshellEngine
cd NutshellEngine

# Clone NutshellEngine's repositories
git clone git@github.com:Team-Nutshell/NutshellEngine.git
git clone git@github.com:Team-Nutshell/NutshellEngine-Common.git
git clone git@github.com:Team-Nutshell/NutshellEngine-Module.git
git clone git@github.com:Team-Nutshell/NutshellEngine-Application.git
git clone git@github.com:Team-Nutshell/NutshellEngine-Editor.git
git clone git@github.com:Team-Nutshell/NutshellEngine-GraphicsModule.git
git clone git@github.com:Team-Nutshell/NutshellEngine-PhysicsModule.git
git clone git@github.com:Team-Nutshell/NutshellEngine-WindowModule.git
git clone git@github.com:Team-Nutshell/NutshellEngine-AudioModule.git
git clone git@github.com:Team-Nutshell/NutshellEngine-AssetLoaderModule.git

# Build NutshellEngine
cd NutshellEngine
mkdir build
cd build
cmake .. -DNTSHENGN_COMMON_PATH=../../NutshellEngine-Common
make -j
mkdir modules
cd ../..

# Build an application
cd NutshellEngine-Application
git fetch
git checkout application/camera-first-person
mkdir build
cd  build
mkdir modules
cmake .. -DNTSHENGN_COMMON_PATH=../../NutshellEngine-Common -DNTSHENGN_INSTALL_SCRIPTS_PATH=../../NutshellEngine/build
make -j install
cd ../..

# Setup all modules
cd NutshellEngine-GraphicsModule
git fetch
git checkout module/neige
mkdir build
cd build
cmake .. -DNTSHENGN_COMMON_PATH=../../NutshellEngine-Common -DNTSHENGN_MODULE_PATH=../../NutshellEngine-Module -DNTSHENGN_INSTALL_MODULE_PATH=../../NutshellEngine/build/modules
make -j install
cd ../..

cd NutshellEngine-PhysicsModule
git fetch
git checkout module/euler
mkdir build
cd build
cmake .. -DNTSHENGN_COMMON_PATH=../../NutshellEngine-Common -DNTSHENGN_MODULE_PATH=../../NutshellEngine-Module -DNTSHENGN_INSTALL_MODULE_PATH=../../NutshellEngine/build/modules
make -j install
cd ../..

cd NutshellEngine-WindowModule
git fetch
git checkout module/glfw
mkdir build
cd build
cmake .. -DNTSHENGN_COMMON_PATH=../../NutshellEngine-Common -DNTSHENGN_MODULE_PATH=../../NutshellEngine-Module -DNTSHENGN_INSTALL_MODULE_PATH=../../NutshellEngine/build/modules
make -j install
cd ../..

cd NutshellEngine-AudioModule
git fetch
git checkout module/openal-soft
mkdir build
cd build
cmake .. -DNTSHENGN_COMMON_PATH=../../NutshellEngine-Common -DNTSHENGN_MODULE_PATH=../../NutshellEngine-Module -DNTSHENGN_INSTALL_MODULE_PATH=../../NutshellEngine/build/modules
make -j install
cd ../..

cd NutshellEngine-AssetLoaderModule
git fetch
git checkout module/multi
mkdir build
cd build
cmake .. -DNTSHENGN_COMMON_PATH=../../NutshellEngine-Common -DNTSHENGN_MODULE_PATH=../../NutshellEngine-Module -DNTSHENGN_INSTALL_MODULE_PATH=../../NutshellEngine/build/modules
make -j install
cd ../..

# Launch base application
./NutshellEngine/build/NutshellEngine