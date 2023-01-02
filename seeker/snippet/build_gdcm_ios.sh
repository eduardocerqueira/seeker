#date: 2023-01-02T16:57:51Z
#url: https://api.github.com/gists/597e8ade3b9440e003eba18074ba7d75
#owner: https://api.github.com/users/hhool

#!/bin/bash

# Parameters
os_target_version=11
gdcm_tag=v2.8.9
install_dir=/usr/local/Frameworks/gdcm

# Directories
script_dir=$(cd $(dirname $0) || exit 1; pwd)
src_dir=${script_dir}/gdcm-src
install_lib_dir=${install_dir}/lib
arm64_install_dir=${install_dir}-arm64
x86_64_install_dir=${install_dir}-x86_64
arm64_install_lib_dir=${arm64_install_dir}/lib
x86_64_install_lib_dir=${x86_64_install_dir}/lib

# Clone
git -C ${src_dir} rev-parse &> /dev/null
if [ $? -eq 0 ]; then
  pushd ${src_dir} > /dev/null
  git checkout ${gdcm_tag}
  popd > /dev/null
else
  git clone https://github.com/malaterre/GDCM -b ${gdcm_tag} ${src_dir}
fi

set -e

function build {
  sdk=$1
  arch=$2
  build_dir=${script_dir}/gdcm-${arch}

  # Configure
  cmake -GXcode \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_TRY_COMPILE_PLATFORM_VARIABLES="CMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED" \
    -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO \
    -DCMAKE_XCODE_ATTRIBUTE_ENABLE_BITCODE=YES \
    -DCMAKE_XCODE_ATTRIBUTE_BITCODE_GENERATION_MODE=bitcode \
    -DCMAKE_OSX_SYSROOT=${sdk} \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=${os_target_version} \
    -DCMAKE_OSX_ARCHITECTURES=${arch} \
    -DCMAKE_IOS_INSTALL_COMBINED=NO \
    -DCMAKE_INSTALL_PREFIX=${install_dir}-${arch} \
    -DGDCM_BUILD_DOCBOOK_MANPAGES=0 \
    -S ${src_dir} \
    -B ${build_dir}

  # Build & Install
  rm -rf ${install_dir}-${arch} &> /dev/null
  cmake --build ${build_dir} --config Release --target install
}
build iphoneos arm64
build iphonesimulator x86_64

# Copy headers and remove libraries
echo "-- Copy headers to ${install_dir}"
rm -rf ${install_dir} &> /dev/null
cp -rf ${arm64_install_dir} ${install_dir}
rm -rf ${install_lib_dir}/*.a

# Combine libraries
for path in $(ls -1 ${arm64_install_lib_dir}/*.a | sort); do
  filename=$(basename ${path})
  arm64_lib=${arm64_install_lib_dir}/${filename}
  x86_64_lib=${x86_64_install_lib_dir}/${filename}
  combined_lib=${install_lib_dir}/${filename}
  echo "-- Create fat library ${combined_lib}"
  lipo -create ${arm64_lib} ${x86_64_lib} -output ${combined_lib}
done

rm -rf ${arm64_install_dir}
rm -rf ${x86_64_install_dir}
