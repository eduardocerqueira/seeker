#date: 2024-03-07T18:19:18Z
#url: https://api.github.com/gists/a08eca2bba3e5e23bda2b3f7d7506ab0
#owner: https://api.github.com/users/jmcerrejon

#!/bin/bash

function reinstall_vulkan_driver() {
    readonly BUILD_MESA_VULKAN_DRIVER_DIR="/home/ulysess/mesa_vulkan/build"

    if [[ ! -d $BUILD_MESA_VULKAN_DRIVER_DIR ]]; then
        echo "Vulkan driver not found. Exiting..."
        exit 1
    fi

    cd $BUILD_MESA_VULKAN_DRIVER_DIR || exit 1
    echo "Reinstalling Vulkan driver..."
    sudo ninja install
    echo "Vulkan driver reinstalled!."
}
reinstall_vulkan_driver
