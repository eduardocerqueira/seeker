#date: 2024-11-26T16:55:58Z
#url: https://api.github.com/gists/87c0fb590c4063397fd1ca8fb3a1df6e
#owner: https://api.github.com/users/YieldRay

#!/usr/bin/env bash
# 使用errexit, nounset, pipefail选项，启用nullglob扩展
set -o errexit -o nounset -o pipefail
shopt -s nullglob
# 设置btrfs路径和cgroup
btrfs_path='/var/bocker' && cgroups='cpu,cpuacct,memory'
# 解析命令行参数，将`--key=value`转换为`BOCKER_key=value`，`--key`转换为`BOCKER_key=x`
[[ $# -gt 0 ]] && while [ "${1:0:2}" == '--' ]; do
    OPTION=${1:2}
    [[ $OPTION =~ = ]] && declare "BOCKER_${OPTION/=*/}=${OPTION/*=/}" || declare "BOCKER_${OPTION}=x"
    shift
done

# 检查btrfs子卷是否存在
function bocker_check() {
    # 使用grep -q静默搜索，存在返回0，否则返回1
    btrfs subvolume list "$btrfs_path" | grep -qw "$1" && echo 0 || echo 1
}

# 从目录创建镜像
function bocker_init() {
    #HELP Create an image from a directory:\nBOCKER init <directory>
    uuid="img_$(shuf -i 42002-42254 -n 1)" # 生成随机uuid，范围42002-42254
    if [[ -d "$1" ]]; then                 # 判断目录是否存在
        # 如果uuid已存在，则运行容器
        [[ "$(bocker_check "$uuid")" == 0 ]] && bocker_run "$@"
        # 创建btrfs子卷
        btrfs subvolume create "$btrfs_path/$uuid" >/dev/null
        # 复制目录内容到子卷，使用reflink优化
        cp -rf --reflink=auto "$1"/* "$btrfs_path/$uuid" >/dev/null
        # 记录镜像来源目录
        [[ ! -f "$btrfs_path/$uuid"/img.source ]] && echo "$1" >"$btrfs_path/$uuid"/img.source
        echo "Created: $uuid"
    else
        echo "No directory named '$1' exists"
    fi
}

# 从Docker Hub拉取镜像
function bocker_pull() {
    #HELP Pull an image from Docker Hub:\nBOCKER pull <name> <tag>
    # 获取Docker Hub的认证token
    token="$(curl -sL -o /dev/null -D- -H 'X-Docker-Token: "**********"://index.docker.io/v1/repositories/$1/images" | tr -d '\r' | awk -F ': *' '$1 == "X-Docker-Token" { print $2 }')"
    registry='https://registry-1.docker.io/v1' # Docker Registry API地址
    # 获取镜像ID
    id="$(curl -sL -H "Authorization: "**********"
    [[ "${#id}" -ne 64 ]] && echo "No image named '$1:$2' exists" && exit 1 # 校验镜像ID长度
    # 获取镜像层级信息
    ancestry="$(curl -sL -H "Authorization: "**********"
    # 使用IFS分隔ancestry字符串为数组
    IFS=',' && ancestry=(${ancestry//[\[\] \"]/}) && IFS=' \n\t'
    tmp_uuid="$(uuidgen)" && mkdir /tmp/"$tmp_uuid" # 创建临时目录
    for id in "${ancestry[@]}"; do                  # 循环下载每一层镜像
        # 下载镜像层
        curl -#L -H "Authorization: "**********"
        # 解压镜像层
        tar xf /tmp/"$tmp_uuid"/layer.tar -C /tmp/"$tmp_uuid" && rm /tmp/"$tmp_uuid"/layer.tar
    done
    echo "$1:$2" >/tmp/"$tmp_uuid"/img.source               # 记录镜像来源
    bocker_init /tmp/"$tmp_uuid" && rm -rf /tmp/"$tmp_uuid" # 使用临时目录初始化镜像并删除临时目录
}

# 删除镜像或容器
function bocker_rm() {
    #HELP Delete an image or container:\nBOCKER rm <image_id or container_id>
    [[ "$(bocker_check "$1")" == 1 ]] && echo "No container named '$1' exists" && exit 1 # 检查容器是否存在
    btrfs subvolume delete "$btrfs_path/$1" >/dev/null                                   # 删除btrfs子卷
    cgdelete -g "$cgroups:/$1" &>/dev/null || true                                       # 删除cgroup，忽略错误
    echo "Removed: $1"
}

# 列出镜像
function bocker_images() {
    #HELP List images:\nBOCKER images
    echo -e "IMAGE_ID\t\tSOURCE"
    for img in "$btrfs_path"/img_*; do
        img=$(basename "$img")                                 # 获取文件名
        echo -e "$img\t\t$(cat "$btrfs_path/$img/img.source")" # 打印镜像ID和来源
    done
}

# 列出容器
function bocker_ps() {
    #HELP List containers:\nBOCKER ps
    echo -e "CONTAINER_ID\t\tCOMMAND"
    for ps in "$btrfs_path"/ps_*; do
        ps=$(basename "$ps")
        echo -e "$ps\t\t$(cat "$btrfs_path/$ps/$ps.cmd")" # 打印容器ID和运行命令
    done
}

# 创建并运行容器
function bocker_run() {
    #HELP Create a container:\nBOCKER run <image_id> <command>
    # 生成随机的容器ID
    uuid="ps_$(shuf -i 42002-42254 -n 1)"
    # 检查镜像是否存在
    [[ "$(bocker_check "$1")" == 1 ]] && echo "No image named '$1' exists" && exit 1
    # 处理UUID冲突
    [[ "$(bocker_check "$uuid")" == 0 ]] && echo "UUID conflict, retrying..." && bocker_run "$@" && return
    # 获取命令，计算IP和MAC地址
    cmd="${@:2}" && ip="$(echo "${uuid: -3}" | sed 's/0//g')" && mac="${uuid: -3:1}:${uuid: -2}"
    # 创建veth pair虚拟网络接口
    ip link add dev veth0_"$uuid" type veth peer name veth1_"$uuid"
    # 设置veth0_"$uuid"
    ip link set dev veth0_"$uuid" up
    # 将veth0_"$uuid"添加到bridge0网桥
    ip link set veth0_"$uuid" master bridge0
    # 创建netns网络命名空间
    ip netns add netns_"$uuid"
    # 将veth1_"$uuid"移到netns_"$uuid"命名空间
    ip link set veth1_"$uuid" netns netns_"$uuid"
    # 在netns_"$uuid"命名空间内配置网络
    ip netns exec netns_"$uuid" ip link set dev lo up                                  # 启用lo环回接口
    ip netns exec netns_"$uuid" ip link set veth1_"$uuid" address 02:42:ac:11:00"$mac" # 设置MAC地址
    ip netns exec netns_"$uuid" ip addr add 10.0.0."$ip"/24 dev veth1_"$uuid"          # 设置IP地址
    ip netns exec netns_"$uuid" ip link set dev veth1_"$uuid" up                       # 启用veth1_"$uuid"
    ip netns exec netns_"$uuid" ip route add default via 10.0.0.1                      # 设置默认网关

    btrfs subvolume snapshot "$btrfs_path/$1" "$btrfs_path/$uuid" >/dev/null # 创建btrfs快照
    echo 'nameserver 8.8.8.8' >"$btrfs_path/$uuid"/etc/resolv.conf           # 设置DNS
    echo "$cmd" >"$btrfs_path/$uuid/$uuid.cmd"                               # 保存运行命令
    # 创建cgroup
    cgcreate -g "$cgroups:/$uuid"
    # 设置CPU份额
    : "${BOCKER_CPU_SHARE:=512}" && cgset -r cpu.shares="$BOCKER_CPU_SHARE" "$uuid"
    # 设置内存限制
    : "${BOCKER_MEM_LIMIT:=512}" && cgset -r memory.limit_in_bytes="$((BOCKER_MEM_LIMIT * 1000000))" "$uuid"

    cgexec -g "$cgroups:$uuid" \ # 在cgroup中执行
    ip netns exec netns_"$uuid" \ # 在netns中执行
    unshare -fmuip --mount-proc \ # 取消共享文件系统、用户、进程、网络命名空间和ipc命名空间，并挂载proc
    chroot "$btrfs_path/$uuid" \ # 切换根目录
    /bin/sh -c "/bin/mount -t proc proc /proc && $cmd" \ # 执行命令
    2>&1 | tee "$btrfs_path/$uuid/$uuid.log" || true # 记录日志
    # 清理网络配置
    ip link del dev veth0_"$uuid"
    ip netns del netns_"$uuid"
}

# 在容器中执行命令
function bocker_exec() {
    #HELP Execute a command in a running container:\nBOCKER exec <container_id> <command>
    [[ "$(bocker_check "$1")" == 1 ]] && echo "No container named '$1' exists" && exit 1 # 检查容器是否存在
    # 获取容器init进程的PID
    cid="$(ps o ppid,pid | grep "^$(ps o pid,cmd | grep -E "^\ *[0-9]+ unshare.*$1" | awk '{print $1}')" | awk '{print $2}')"
    # 检查容器是否运行
    [[ ! "$cid" =~ ^\ *[0-9]+$ ]] && echo "Container '$1' exists but is not running" && exit 1
    # 进入容器的命名空间并执行命令
    nsenter -t "$cid" -m -u -i -n -p chroot "$btrfs_path/$1" "${@:2}"
}

# 查看容器日志
function bocker_logs() {
    #HELP View logs from a container:\nBOCKER logs <container_id>
    [[ "$(bocker_check "$1")" == 1 ]] && echo "No container named '$1' exists" && exit 1 # 检查容器是否存在
    cat "$btrfs_path/$1/$1.log"                                                          # 打印日志文件
}

# 将容器提交为镜像
function bocker_commit() {
    #HELP Commit a container to an image:\nBOCKER commit <container_id> <image_id>
    [[ "$(bocker_check "$1")" == 1 ]] && echo "No container named '$1' exists" && exit 1 # 检查容器是否存在
    [[ "$(bocker_check "$2")" == 0 ]] && bocker_rm "$2"                                  # 如果镜像ID已存在则删除
    #从容器创建快照
    btrfs subvolume snapshot "$btrfs_path/$1" "$btrfs_path/$2" >/dev/null
    echo "Created: $2"
}

# 显示帮助信息
function bocker_help() {
    #HELP Display this message:\nBOCKER help
    sed -n "s/^.*#HELP\\s//p;" <"$1" | sed "s/\\\\n/\n\t/g;s/$/\n/;s!BOCKER!${1/!/\\!}!g" # 提取#HELP注释并格式化输出
}

[[ -z "${1-}" ]] && bocker_help "$0"                                                 # 如果没有参数，显示帮助信息
case $1 in                                                                           # 根据第一个参数执行相应函数
pull | init | rm | images | ps | run | exec | logs | commit) bocker_"$1" "${@:2}" ;; # 执行bocker命令
*) bocker_help "$0" ;;                                                               # 默认显示帮助信息
esac
${@:2}" ;; # 执行bocker命令
*) bocker_help "$0" ;;                                                               # 默认显示帮助信息
esac
