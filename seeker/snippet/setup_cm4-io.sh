#date: 2022-12-19T16:28:11Z
#url: https://api.github.com/gists/3ab5d20947e8dcaeec5601f28255c05f
#owner: https://api.github.com/users/CaptChrisD

#!/bin/bash
set -e

mkdir cm4-io && cd cm4-io
git clone git://git.busybox.net/buildroot buildroot
git clone https://github.com/flatmax/buildroot.rockchip.git buildroot.rockchip.ext
cat >./cm4-io.patch <<EOF
diff --git a/configs/cm3_defconfig b/configs/cm3_defconfig
index 7dbb7d4..614a549 100644
--- a/configs/cm3_defconfig
+++ b/configs/cm3_defconfig
@@ -30,7 +30,7 @@ BR2_TARGET_UBOOT_USE_DEFCONFIG=y
 BR2_TARGET_UBOOT_CUSTOM_GIT=y
 BR2_TARGET_UBOOT_CUSTOM_REPO_URL="https://github.com/radxa/u-boot.git"
 BR2_TARGET_UBOOT_CUSTOM_REPO_VERSION="26d3b6963ed2d2215348f1baba8b9646ed3dc6ea"
-BR2_TARGET_UBOOT_BOARD_DEFCONFIG="radxa-cm3-io-rk3566"
+BR2_TARGET_UBOOT_BOARD_DEFCONFIG="radxa-cm3-rpi-cm4-io-rk3566"
 BR2_TARGET_UBOOT_CUSTOM_MAKEOPTS="BL31=../rkbin-7d631e0d5b2d373b54d4533580d08fb9bd2eaad4/bin/rk35/rk3568_bl31_v1.24.elf spl/u-boot-spl.bin u-boot.dtb u-boot.itb"
 BR2_TARGET_UBOOT_NEEDS_PYLIBFDT=y
 BR2_TARGET_UBOOT_FORMAT_BIN=y
@@ -48,7 +48,7 @@ BR2_LINUX_KERNEL_CUSTOM_REPO_URL="https://github.com/radxa/kernel.git"
 BR2_LINUX_KERNEL_CUSTOM_REPO_VERSION="f0b4c3d6f86f433280662a6158e0bc1b4d83503a"
 BR2_LINUX_KERNEL_USE_DEFCONFIG=y
 BR2_LINUX_KERNEL_DEFCONFIG="rockchip_linux"
-BR2_LINUX_KERNEL_INTREE_DTS_NAME="rockchip/rk3566-radxa-cm3-io"
+BR2_LINUX_KERNEL_INTREE_DTS_NAME="rockchip/rk3566-radxa-cm3-rpi-cm4-io"
 BR2_KERNEL_HEADERS_4_19=y

 #BR2_LINUX_KERNEL_LATEST_VERSION=y
EOF
cd buildroot.rockchip.ext
git apply ../cm4-io.patch
cd ..
mkdir buildroot.dl
cd buildroot && git checkout 2022.08.2
cd ..
source buildroot.rockchip.ext/setup.cm3.sh ./buildroot
##### MANUAL STEPS TO COMPILE
# cd cm4-io/buildroot
# make