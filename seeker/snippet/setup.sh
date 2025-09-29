#date: 2025-09-29T17:09:47Z
#url: https://api.github.com/gists/c6f61d829f75a7383fd0dc0a09c82bc9
#owner: https://api.github.com/users/CmdBlockZQG

#!/bin/bash

run_cmd() {
    local cmd=("$@")
    local retries=3
    local attempt=1

    while [ $attempt -le $retries ]; do
        echo "Running command: ${cmd[*]}"
        "${cmd[@]}"
        local exit_code=$?

        if [ $exit_code -eq 0 ]; then
            return 0
        else
            if [ $attempt -lt $retries ]; then
                local next_attempt=$((attempt + 1))
                echo "Command failed, retrying ..."
            fi
        fi
        attempt=$((attempt + 1))
    done

    echo "Command '${cmd[*]}' failed $retries times, exiting ..."
    exit 1
}

setup_env() {
    local commands_to_run=(
        # tuna mirror
        "sudo sed -Ei 's/(gb.)?archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list"
        # install packages
        "sudo apt update"
        "sudo apt upgrade -y"
        "sudo apt install -y git vim curl openjdk-17-jdk \
            gcc g++ gdb make build-essential autoconf \
            python-is-python3 help2man perl flex bison ccache \
            libreadline-dev libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev \
            g++-riscv64-linux-gnu llvm llvm-dev"
        # fix compile error using riscv64-linux-gnu
        "sudo sed -i 's|^# include <gnu/stubs-ilp32.h>|//# include <gnu/stubs-ilp32.h>|' /usr/riscv64-linux-gnu/include/gnu/stubs.h"
        # install verilator
        "git clone https://gitee.com/mirrors/Verilator.git /tmp/verilator"
        "cd /tmp/verilator"
        "git checkout stable"
        "autoconf"
        "./configure"
        "make -j$(nproc)"
        "sudo make install"
        "cd -"
        "rm -rf /tmp/verilator"
    )

    for cmd_str in "${commands_to_run[@]}"; do
        local cmd_array=()
        eval "cmd_array=($cmd_str)"
        run_cmd "${cmd_array[@]}"
    done

    echo "Environment setup completed."
}

setup_repo() {
    # clone repo
    run_cmd git clone --depth 1 -b $1 https://github.com/sashimi-yzh/ysyx-submit-test.git ysyx-workbench
    YSYX_HOME=$(pwd)/ysyx-workbench
    # create activate.sh
    echo "export YSYX_HOME=$(pwd)/ysyx-workbench" > activate.sh
    echo "export NEMU_HOME=\$YSYX_HOME/nemu" >> activate.sh
    echo "export AM_HOME=\$YSYX_HOME/abstract-machine" >> activate.sh
    echo "export NAVY_HOME=\$YSYX_HOME/navy-apps" >> activate.sh
    echo "export NPC_HOME=\$YSYX_HOME/npc" >> activate.sh
    echo "export NVBOARD_HOME=\$YSYX_HOME/nvboard" >> activate.sh
    echo "export PATH=\$PATH:$(pwd)/bin" >> activate.sh
    # cd into workbench
    cd $YSYX_HOME
    # disable git tracer
    echo ".git_commit:" >> Makefile
    echo -e "\t@echo git tracer is disabled" >> Makefile
    # clone other repos
    run_cmd git clone --depth 1 https://github.com/NJU-ProjectN/am-kernels
    run_cmd git clone --depth 1 https://github.com/NJU-ProjectN/rt-thread-am
    run_cmd git clone --depth 1 https://github.com/NJU-ProjectN/nvboard
    run_cmd git clone --depth 1 -b ysyx6 https://github.com/OSCPU/ysyxSoC
    # apply patches
    cd $YSYX_HOME/rt-thread-am
    git am $YSYX_HOME/patch/rt-thread-am/*
    cd $YSYX_HOME/ysyxSoC
    git am $YSYX_HOME/patch/ysyxSoC/*
    # clean up
    make -C $YSYX_HOME/nemu clean
    make -C $YSYX_HOME/am-kernels clean-all
    make -C $YSYX_HOME/npc clean
    # install mill
    mkdir -p $YSYX_HOME/../bin
    MILL_VERSION=0.11.13
    if [[ -e $YSYX_HOME/npc/.mill-version ]]; then
        MILL_VERSION=`cat $YSYX_HOME/npc/.mill-version`
    fi
    echo "Downloading mill with version $MILL_VERSION"
    run_cmd sh -c "curl -L https://github.com/com-lihaoyi/mill/releases/download/$MILL_VERSION/$MILL_VERSION > $YSYX_HOME/../bin/mill"
    chmod +x $YSYX_HOME/../bin/mill
    # generate verilog for ysyxSoC
    sed -i -e 's+git@github.com:+https://github.com/+' $YSYX_HOME/ysyxSoC/.gitmodules
    run_cmd make -C $YSYX_HOME/ysyxSoC dev-init
    run_cmd make -C $YSYX_HOME/ysyxSoC verilog
}

if [ -z "$1" ]; then
    echo "Error: No argument specified."
    echo "Usage: $0 {env|repo|clean}"
    exit 1
fi

case "$1" in
    env)
        setup_env
        ;;
    repo)
        setup_repo $2 
        ;;
    clean)
        rm -rf ysyx-workbench bin activate.sh
        ;;
    *)
        echo "Error: Unknown argument '$1'."
        echo "Usage: $0 {env|repo|clean}"
        exit 1
        ;;
esac
