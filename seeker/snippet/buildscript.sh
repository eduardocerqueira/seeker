#date: 2023-08-14T17:03:15Z
#url: https://api.github.com/gists/7df7d35bbc1187c257d04214d5324dc3
#owner: https://api.github.com/users/utkustnr

#!/bin/bash

if [ ! -f ./Makefile ]; then
  echo "Run this script inside kernel dir"
  exit
fi

if [ ! -d ../toolchain ]; then
	mkdir -P ../toolchain/clang-r383902b1
	git clone https://github.com/LineageOS/android_prebuilts_gcc_linux-x86_aarch64_aarch64-linux-android-4.9.git --depth=1 -b lineage-19.1 ../toolchain/aarch64-linux-android-4.9
	wget https://android.googlesource.com/platform/prebuilts/clang/host/linux-x86/+archive/refs/heads/android11-qpr3-release/clang-r383902b1.tar.gz -P ../toolchain
	tar xf ../toolchain/clang-r383902b1.tar.gz -C ../toolchain/clang-r383902b1
fi

export ARCH=arm64
export LLVM=1
export CLANG_PREBUILT_BIN=$(pwd)/../toolchain/clang-r383902b1/bin
export PATH=$PATH:$CLANG_PREBUILT_BIN
BUILD_CROSS_COMPILE=$(pwd)/../toolchain/aarch64-linux-android-4.9/bin/aarch64-linux-android-
KERNEL_LLVM_BIN=$(pwd)/../toolchain/clang-r383902b1/bin/clang
CLANG_TRIPLE=aarch64-linux-gnu-
KERNEL_MAKE_ENV="CONFIG_BUILD_ARM64_DT_OVERLAY=y"

mkdir out
make -j$(nproc --all) -C $(pwd) O=$(pwd)/out $KERNEL_MAKE_ENV ARCH=arm64 CROSS_COMPILE=$BUILD_CROSS_COMPILE REAL_CC=$KERNEL_LLVM_BIN CLANG_TRIPLE=$CLANG_TRIPLE CONFIG_SECTION_MISMATCH_WARN_ONLY=y vendor/a73xq_eur_open_defconfig
make -j$(nproc --all) -C $(pwd) O=$(pwd)/out $KERNEL_MAKE_ENV ARCH=arm64 CROSS_COMPILE=$BUILD_CROSS_COMPILE REAL_CC=$KERNEL_LLVM_BIN CLANG_TRIPLE=$CLANG_TRIPLE CONFIG_SECTION_MISMATCH_WARN_ONLY=y

if [ ! -f ../boot.img-ramdisk.cpio.gz ]; then
	curl https://gist.githubusercontent.com/utkustnr/3cd674fda61d58c967cc38f5b19e52a8/raw/0d70be6c93871e913b6473b1dd8191694bc2d700/boot.img-ramdisk.cpio.gz | base64 -d > ../boot.img-ramdisk.cpio.gz
fi

if [ ! -d ../mkbootimg ]; then
	git clone https://android.googlesource.com/platform/system/tools/mkbootimg ../mkbootimg
else
	../mkbootimg/mkbootimg.py --kernel out/arch/arm64/boot/Image --cmdline "console=null androidboot.hardware=qcom androidboot.memcg=1 lpm_levels.sleep_disabled=1 video=vfb:640x400,bpp=32,memsize=3072000 msm_rtb.filter=0x237 service_locator.enable=1 androidboot.usbcontroller=a600000.dwc3 swiotlb=0 loop.max_part=7 cgroup.memory=nokmem,nosocket firmware_class.path=/vendor/firmware_mnt/image pcie_ports=compat loop.max_part=7 iptable_raw.raw_before_defrag=1 ip6table_raw.raw_before_defrag=1 printk.devkmsg=on" --header_version 3 --os_patch_level 2023-06 --os_version 11.0.0 --output ../boot.img --pagesize 4096 --ramdisk ../boot.img-ramdisk.cpio.gz 
fi

rm -rf ../boot.img-ramdisk.cpio.gz 