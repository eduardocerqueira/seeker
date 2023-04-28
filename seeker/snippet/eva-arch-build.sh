#date: 2023-04-28T17:06:03Z
#url: https://api.github.com/gists/b08bd1cc7bab10ba7c403f49ceda1100
#owner: https://api.github.com/users/Dreamail

#!/bin/bash

clear

set -eux

setup_export() {
    export LC_ALL=C
    export SOURCE_PATH=$PWD
    export CLANG_PATH=$SOURCE_PATH/prebuilts-master/clang/host/linux-x86/clang-r450784d/
    export CONFIG_KERNELSU=true     # Optional values: true/false
    export KERNELSU_TAG=main        # Optional values: main/NULL/'v\d+(\.\d+)*'
    export CONFIG_DOCKER=false      # Optional values: true/false
    export CONFIG_ROOTGUARD=false   # Optional values: true/false
    export DEVICE_NAME=9RT          # Optional values: LEMONADE/9RT/LUNAA
    export CONFIG_LTO=thin          # Optional values: none/thin
}

sync_source() {
    cd $SOURCE_PATH
    test -d ./.repo || repo init -u https://github.com/mvaisakh/android_kernel_manifest -b eva-oneplus-5.4 --depth=1
    test ! -e ./kernel/msm-5.4 || rm -rf ./kernel/msm-5.4
    test ! -e ./out || rm -rf ./out
    repo sync --no-tags --no-clone-bundle -j$(nproc --all)
}

setup_environment() {
    cd $SOURCE_PATH
    sudo pacman -S zstd tar wget curl base-devel --noconfirm
    wget -O ./lib32-ncurses.pkg.tar.zst https://archlinux.org/packages/multilib/x86_64/lib32-ncurses/download/
    wget -O ./lib32-readline.pkg.tar.zst https://archlinux.org/packages/multilib/x86_64/lib32-readline/download/
    wget -O ./lib32-zlib.pkg.tar.zst https://archlinux.org/packages/multilib/x86_64/lib32-zlib/download/
    sudo pacman -U ./lib32-ncurses.pkg.tar.zst ./lib32-readline.pkg.tar.zst ./lib32-zlib.pkg.tar.zst --noconfirm
    rm ./*.zst
    yay -S lineageos-devel python2-bin --noconfirm
    if [ ! -d $CLANG_PATH ]; then
      mkdir -p $CLANG_PATH
      cd $CLANG_PATH
      bash <(curl -s "https://raw.githubusercontent.com/Neutron-Toolchains/antman/main/antman") -S
    fi
}

setup_kernelsu() {
    cd $SOURCE_PATH/kernel/msm-5.4
    curl -LSs "https://raw.githubusercontent.com/tiann/KernelSU/main/kernel/setup.sh" | bash -s "$KERNELSU_TAG"
    # Apply patch for clang-17
    wget https://gist.githubusercontent.com/natsumerinchan/cebf0d64ea10a5deecec74fb7803d72a/raw/f9b6539e81ec6e97792f16c4a68b25752e43ba47/0001-kallsyms-strip-LTO-suffixes-from-static-functions.patch
    git apply ./0001-kallsyms-strip-LTO-suffixes-from-static-functions.patch
    # Enable Kprobe
    about_kprobe="
        CONFIG_MODULES=y
        CONFIG_KPROBES=y
        CONFIG_HAVE_KPROBES=y
        CONFIG_TRACING_SUPPORT=y
        CONFIG_FTRACE=y
        CONFIG_HAVE_REGS_AND_STACK_ACCESS_API=y
        CONFIG_KPROBE_EVENTS=y
    "
    for config_name in $about_kprobe
    do
        printf "\n$config_name\n" >> "arch/arm64/configs/vendor/lahaina_NQGKI.config"
    done
}

docker_support() {
    cd $SOURCE_PATH/kernel/msm-5.4
    wget https://gist.githubusercontent.com/natsumerinchan/da5bcec1c13395b3f92efcc232b0c237/raw/ad143a697e02662a4f3f815cf8253fc4eb5290ce/0002-kernel-Show-the-real-proc-config.gz.patch
    git apply ./0002-kernel-Show-the-real-proc-config.gz.patch
    about_docker="
        CONFIG_NAMESPACES=y
        CONFIG_NET_NS=y
        CONFIG_PID_NS=y
        CONFIG_IPC_NS=y
        CONFIG_UTS_NS=y
        CONFIG_CGROUPS=y
        CONFIG_CGROUP_CPUACCT=y
        CONFIG_CGROUP_DEVICE=y
        CONFIG_CGROUP_FREEZER=y
        CONFIG_CGROUP_SCHED=y
        CONFIG_CPUSETS=y
        CONFIG_MEMCG=y
        CONFIG_KEYS=y
        CONFIG_VETH=y
        CONFIG_BRIDGE=y
        CONFIG_BRIDGE_NETFILTER=y
        CONFIG_IP_NF_FILTER=y
        CONFIG_IP_NF_TARGET_MASQUERADE=y
        CONFIG_NETFILTER_XT_MATCH_ADDRTYPE=y
        CONFIG_NETFILTER_XT_MATCH_CONNTRACK=y
        CONFIG_NETFILTER_XT_MATCH_IPVS=y
        CONFIG_NETFILTER_XT_MARK=y
        CONFIG_IP_NF_NAT=y
        CONFIG_NF_NAT=y
        CONFIG_POSIX_MQUEUE=y
        CONFIG_NF_NAT_IPV4=y
        CONFIG_NF_NAT_NEEDED=y
        CONFIG_CGROUP_BPF=y
        CONFIG_USER_NS=y
        CONFIG_SECCOMP=y
        CONFIG_SECCOMP_FILTER=y
        CONFIG_CGROUP_PIDS=y
        CONFIG_MEMCG_SWAP=y
        CONFIG_MEMCG_SWAP_ENABLED=y
        CONFIG_IOSCHED_CFQ=y
        CONFIG_CFQ_GROUP_IOSCHED=y
        CONFIG_BLK_CGROUP=y
        CONFIG_BLK_DEV_THROTTLING=y
        CONFIG_CGROUP_PERF=y
        CONFIG_PAGE_COUNTER=y
        CONFIG_HUGETLB_PAGE=y
        CONFIG_CGROUP_HUGETLB=y
        CONFIG_NET_CLS_CGROUP=y
        CONFIG_CGROUP_NET_PRIO=y
        CONFIG_CFS_BANDWIDTH=y
        CONFIG_FAIR_GROUP_SCHED=y
        CONFIG_SCHED_WALT=n
        CONFIG_RT_GROUP_SCHED=y
        CONFIG_IP_NF_TARGET_REDIRECT=y
        CONFIG_IP_VS=y
        CONFIG_IP_VS_NFCT=y
        CONFIG_IP_VS_PROTO_TCP=y
        CONFIG_IP_VS_PROTO_UDP=y
        CONFIG_IP_VS_RR=y
        CONFIG_SECURITY_SELINUX=y
        CONFIG_SECURITY_APPARMOR=y
        CONFIG_EXT4_FS=y
        CONFIG_EXT4_FS_POSIX_ACL=y
        CONFIG_EXT4_FS_SECURITY=y
        CONFIG_VXLAN=y 
        CONFIG_VLAN_8021Q=y
        CONFIG_BRIDGE_VLAN_FILTERING=y
        CONFIG_CRYPTO=y 
        CONFIG_CRYPTO_AEAD=y
        CONFIG_CRYPTO_GCM=y
        CONFIG_CRYPTO_SEQIV=y
        CONFIG_CRYPTO_GHASH=y 
        CONFIG_XFRM=y
        CONFIG_XFRM_USER=y
        CONFIG_XFRM_ALGO=y
        CONFIG_INET_ESP=y
        CONFIG_INET_XFRM_MODE_TRANSPORT=y
        CONFIG_IPVLAN=y
        CONFIG_MACVLAN=y
        CONFIG_DUMMY=y
        CONFIG_NF_NAT_FTP=y
        CONFIG_NF_CONNTRACK_FTP=y
        CONFIG_NF_NAT_TFTP=y
        CONFIG_NF_CONNTRACK_TFTP=y
        CONFIG_AUFS_FS=y
        CONFIG_BTRFS_FS=y
        CONFIG_BTRFS_FS_POSIX_ACL=y
        CONFIG_BLK_DEV_DM=y
        CONFIG_DM_THIN_PROVISIONING=y
        CONFIG_OVERLAY_FS=y
    "
    for config_name in $about_docker
    do
        printf "\n$config_name\n" >> "arch/arm64/configs/vendor/lahaina_NQGKI.config"
    done
}

compile_rootguard() {
    cd $SOURCE_PATH/kernel/msm-5.4/drivers
    git clone https://github.com/natsumerinchan/RootGuard.git rootguard
    mv ./rootguard/Makefile.bak ./rootguard/Makefile
    echo "obj-m += rootguard/" >>"./Makefile"
}

build_kernel() {
    cd $SOURCE_PATH/kernel/msm-5.4
    wget https://gist.githubusercontent.com/natsumerinchan/77d5ad9ea42b5a1b4667de9f54c69d8e/raw/03cbe567e798cef5261f551668310067a878ffef/0003-Makefile-Use-CCACHE-for-faster-compilation.patch
    git apply ./0003-Makefile-Use-CCACHE-for-faster-compilation.patch
    cd $SOURCE_PATH
    sed -i s/build-user/mvaisakh/g build/_setup_env.sh
    sed -i s/build-host/statixos/g build/_setup_env.sh
    time CCACHE="/usr/bin/ccache" BUILD_CONFIG=kernel/msm-5.4/build.config.msm.lahaina VARIANT=nqgki DEVICE=$DEVICE_NAME LTO=$CONFIG_LTO POLLY=1 BUILD_KERNEL=1 build/build.sh 2>&1 | tee compile.log
}

make_anykernel3_zip() {
    cd $SOURCE_PATH
    test -d ./backup || mkdir ./backup
    ls | grep zip && mv ./*.zip ./backup
    cp out/msm-5.4-lahaina-nqgki/dist/Image ak3/
    cat out/msm-5.4-lahaina-nqgki/dist/*.dtb > ak3/dtb
    cp out/msm-5.4-lahaina-nqgki/dist/dtbo.img ak3/
    cd ak3/ && zip -r9 eva-martini-$(/bin/date -u '+%Y%m%d%I%M').zip * -x .git README.md ./*/placeholder
    mv *.zip ../
    rm Image dtb dtbo.img
    if test "$CONFIG_ROOTGUARD" == "true"; then
        cd $SOURCE_PATH
        test -d ./rootguard || git clone https://github.com/natsumerinchan/Kernel_Module_Loader.git rootguard
        cd ./rootguard
        git pull
        cp ../out/msm-5.4-lahaina-nqgki/kernel/msm-5.4/drivers/rootguard/RootGuard.ko ./kernel_module
        sed -i 's/Kernel_Module_Loader/RootGuard/g' ./module.prop
        sed -i 's/Kernel Module Loader/RootGuard/g' ./module.prop
        sed -i 's/natsumerinchan/Ylarod/g' ./module.prop
        sed -i '/description/d' ./module.prop
        printf "description=一个防止格机的内核模块，支持内核4.19-5.4" >> ./module.prop
        zip -r rootguard-$(/bin/date -u '+%Y%m%d%I%M').zip * -x .git ./kernel_module/.placeholder .gitattributes .gitgnore
        mv ./*.zip ../
        rm ./kernel_module/*.ko
    fi
    cd $SOURCE_PATH
}

setup_export

sync_source 

if test -e $CLANG_PATH/env_is_setup; then
   echo [INFO]Environment have been setup!
else
   setup_environment
   touch $CLANG_PATH/env_is_setup
fi

if test "$CONFIG_KERNELSU" == "true"; then
   setup_kernelsu
fi

if test "$CONFIG_DOCKER" == "true"; then
   docker_support
fi

if test "$CONFIG_ROOTGUARD" == "true"; then
   compile_rootguard
fi

build_kernel

make_anykernel3_zip

unset LC_ALL