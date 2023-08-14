#date: 2023-08-14T17:00:49Z
#url: https://api.github.com/gists/97aa141a389e9edbce11e48ddc6585b3
#owner: https://api.github.com/users/utkustnr

#!/bin/bash

git subtree add --prefix arch/arm64/boot/dts/vendor/ 				https://github.com/AdarshGrewal/android_kernel_qcom_devicetree.git 		131cb8768239bd4b960af22a023b1af10ae18659
git subtree add --prefix arch/arm64/boot/dts/vendor/qcom/camera/ 	https://github.com/MotorolaMobilityLLC/kernel-camera-devicetree.git 	19431df3ca6fcf35cf2a962f94e970c47481b88b
git subtree add --prefix arch/arm64/boot/dts/vendor/qcom/display/ 	https://github.com/MotorolaMobilityLLC/kernel-display-devicetree.git 	3d5509a108052daad9d4c96e9341388ec705d8ee
git subtree add --prefix drivers/staging/fw-api/				https://git.codelinaro.org/clo/la/platform/vendor/qcom-opensource/wlan/fw-api.git 				LA.UM.9.14.r1-21000-LAHAINA.QSSI12.0
git subtree add --prefix drivers/staging/qca-wifi-host-cmn/ 	https://git.codelinaro.org/clo/la/platform/vendor/qcom-opensource/wlan/qca-wifi-host-cmn.git 	LA.UM.9.14.r1-21000-LAHAINA.QSSI12.0
git subtree add --prefix drivers/staging/qcacld-3.0/			https://git.codelinaro.org/clo/la/platform/vendor/qcom-opensource/wlan/qcacld-3.0.git 			LA.UM.9.14.r1-21000-LAHAINA.QSSI12.0
git subtree add --prefix techpack/audio/			https://git.codelinaro.org/clo/la/platform/vendor/opensource/audio-kernel.git 			LA.UM.9.14.r1-21000-LAHAINA.QSSI12.0
git subtree add --prefix techpack/camera/			https://git.codelinaro.org/clo/la/platform/vendor/opensource/camera-kernel.git 			945d74685ef27aecbda23f7919f04e025c7055c5
git subtree add --prefix techpack/dataipa/			https://git.codelinaro.org/clo/la/platform/vendor/opensource/dataipa.git 				LA.UM.9.14.r1-21000-LAHAINA.QSSI12.0
git subtree add --prefix techpack/datarmnet-ext/	https://git.codelinaro.org/clo/la/platform/vendor/qcom/opensource/datarmnet-ext.git 	LA.UM.9.14.r1-21000-LAHAINA.QSSI12.0
git subtree add --prefix techpack/datarmnet/		https://git.codelinaro.org/clo/la/platform/vendor/qcom/opensource/datarmnet.git 		LA.UM.9.14.r1-21000-LAHAINA.QSSI12.0
git subtree add --prefix techpack/display/			https://git.codelinaro.org/clo/la/platform/vendor/opensource/display-drivers.git 		LA.UM.9.14.r1-21000-LAHAINA.QSSI12.0
git subtree add --prefix techpack/video/			https://git.codelinaro.org/clo/la/platform/vendor/opensource/video-driver.git 			LA.UM.9.14.r1-21000-LAHAINA.QSSI12.0