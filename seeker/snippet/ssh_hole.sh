#date: 2024-12-03T17:00:12Z
#url: https://api.github.com/gists/2547efcaf6f73958d3eb125da12f6a9b
#owner: https://api.github.com/users/ayasechan

#!/bin/bash

# 检查是否以 root 身份运行

if [[ $EUID -ne 0 ]]; then

	echo "此脚本需要以 root 身份运行。"	exit 1

fi

# 配置

SET_NAME="blocked_ips" # ipset 集合名称

PORT=22                # 监听的端口

# 初始化 ipset 集合

if ! ipset list "$SET_NAME" &>/dev/null; then

	echo "创建 ipset 集合 $SET_NAME..."

	ipset create "$SET_NAME" hash:ip

	iptables -A INPUT -m set --match-set "$SET_NAME" src -j DROP

fi

echo "开始监听 22 端口的连接..."

# 使用 nc 监听端口

while true; do

	# 提取连接的 IP 地址

	IP=$(nc -v -l -p $PORT 2>&1 >/dev/null | grep -oP '^\d[^\:]+')

	echo "检测到连接，源 IP: $IP"

	if [[ -n "$IP" ]]; then

		echo "屏蔽 IP 地址 $IP..."

		# 添加到 ipset 并记录

		ipset add "$SET_NAME" "$IP"

	fi

done