#date: 2026-02-09T17:47:05Z
#url: https://api.github.com/gists/7e68e6843c8cd4a131f755c59a69d5a6
#owner: https://api.github.com/users/enihsyou

source /koolshare/scripts/base.sh
NEW_PATH=$(echo $PATH | sed 's/:\/opt\/bin//g' | sed 's/:\/opt\/sbin//g' | sed 's/:\/opt\/usr\/bin//g'| sed 's/:\/opt\/usr\/sbin//g')
export PATH="${NEW_PATH}"
eval $(dbus export tailscale_)
alias echo_date='echo 【$(TZ=UTC-8 date -R +%Y年%m月%d日\ %X)】:'
config_path="/jffs/softcenter/etc/tailscale"
LOG_FILE=/tmp/upload/tailscale_log.txt
LOCK_FILE=/var/lock/tailscale.lock
SNAT_FLAG=1
BASH=${0##*/}
ARGS=$@

run(){
	env -i PATH=${PATH} "$@"
}


set_lock(){
	exec 233>${LOCK_FILE}
	flock -n 233 || {
		# bring back to original log
		http_response "$ACTION"
		# echo_date "$BASH $ARGS" | tee -a ${LOG_FILE}
		exit 1
	}
}

unset_lock(){
	flock -u 233
	rm -rf ${LOCK_FILE}
}

__valid_ip4() {
	local format_4=$(echo "$1" | grep -Eo "([0-9]{1,3}[\.]){3}[0-9]{1,3}$")
	if [ -n "${format_4}" ]; then
		echo "${format_4}"
		return 0
	else
		echo ""
		return 1
	fi
}

__valid_ip6() {
	local format_6=$(echo "$1" | grep -Eo '^\s*((([0-9A-Fa-f]{1,4}:){7}([0-9A-Fa-f]{1,4}|:))|(([0-9A-Fa-f]{1,4}:){6}(:[0-9A-Fa-f]{1,4}|((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3})|:))|(([0-9A-Fa-f]{1,4}:){5}(((:[0-9A-Fa-f]{1,4}){1,2})|:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3})|:))|(([0-9A-Fa-f]{1,4}:){4}(((:[0-9A-Fa-f]{1,4}){1,3})|((:[0-9A-Fa-f]{1,4})?:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){3}(((:[0-9A-Fa-f]{1,4}){1,4})|((:[0-9A-Fa-f]{1,4}){0,2}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){2}(((:[0-9A-Fa-f]{1,4}){1,5})|((:[0-9A-Fa-f]{1,4}){0,3}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){1}(((:[0-9A-Fa-f]{1,4}){1,6})|((:[0-9A-Fa-f]{1,4}){0,4}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|(:(((:[0-9A-Fa-f]{1,4}){1,7})|((:[0-9A-Fa-f]{1,4}){0,5}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:)))(%.+)?\s*')
	if [ -n "${format_6}" ]; then
		echo "${format_6}"
		return 0
	else
		echo ""
		return 1
	fi
}

close_in_five() {
	echo_date "插件将在5秒后自动关闭！！"
	local i=5
	while [ $i -ge 0 ]; do
		sleep 1
		echo_date $i
		let i--
	done
	stop_tailscale
	dbus set tailscale_enable=0
	sync
	echo_date "插件已关闭！！"
	echo_date =================================================================
	unset_lock
	exit
}

stop_tailscale(){
	# stop first
	local TS_PID=$(pidof tailscale)
	if [ -n "${TSD_PID}" ];then
		echo_date "关闭tailscale进程！"
		killall tailscale >/dev/null 2>&1
	fi

	local TSD_PID=$(pidof tailscaled)
	if [ -n "${TSD_PID}" ];then
		echo_date "关闭tailscaled进程！"
		kill -9 ${TS_PID} >/dev/null 2>&1
		killall tailscaled >/dev/null 2>&1
		echo_date "一些清理工作！"
		tailscaled -cleanup >/dev/null 2>&1
	fi

	#rm -rf /tmp/upload/tailscaled_log.txt
	dbus remove tailscale_ipv4
	dbus remove tailscale_ipv6
	
	del_fw_rule
}

start_tailscale(){
	# 0. prepare, 
	echo_date "开启Tailscale服务..."
	mkdir -p /koolshare/configs/tailscale
	local IP_CIDR=$(ip addr show br0 2>/dev/null|grep -E "inet " | awk '{print $2}')
	local IP_ADDR=${IP_CIDR%/*}
	local IP_ADDR=${IP_ADDR%.*}.0
	local IP_MASK=${IP_CIDR#*/}
	local IP_CIDR_2="${IP_ADDR}/${IP_MASK}"
	local TSUP_LOG=/tmp/upload/tailscale_up_log.txt
	rm -rf /tmp/upload/tailscaled_log.txt
	rm -rf /tmp/tailscale*.pid
	rm -rf ${TSUP_LOG}

	# 1. stop first
	stop_tailscale >/dev/null 2>&1

	# 2. del rule first
	del_fw_rule >/dev/null 2>&1

	# 3. insert module
	local TU=$(lsmod |grep -w tun)
	local CM=$(lsmod | grep xt_comment)
	local OS=$(uname -r)
	if [ -z "${TU}" ];then
		echo_date "加载tun内核模块！"
		modprobe tun >/dev/null 2>&1
	fi
	if [ -z "${CM}" -a -f "/lib/modules/${OS}/kernel/net/netfilter/xt_comment.ko" ];then
		echo_date "加载xt_comment.ko内核模块！"
		insmod /lib/modules/${OS}/kernel/net/netfilter/xt_comment.ko >/dev/null 2>&1
	fi

	# 4. start tailscaled process
	echo_date "启动tailscaled进程..."
	if [ -z "${tailscale_port}" ];then
		local ext_arg=""
	else
		local ext_arg="-port ${tailscale_port}"
	fi
	run tailscaled -cleanup >/dev/null 2>&1
	TSD_LOG=/tmp/upload/tailscaled_log.txt
	rm -rf /tmp/tailscaled.pid
	rm -rf ${TSD_LOG}
	run start-stop-daemon --start --quiet --make-pidfile --pidfile /tmp/tailscaled.pid --background --startas /bin/sh -- -c "exec /koolshare/bin/tailscaled -state /koolshare/configs/tailscale/tailscaled.state > ${TSD_LOG} 2>&1"
	local TSPID
	local i=20
	until [ -n "$TSPID" ]; do
		i=$(($i - 1))
		TSPID=$(pidof tailscaled)
		if [ "$i" -lt 1 ]; then
			echo_date "tailscaled进程启动失败！"
			echo_date "关闭插件！"
			close_in_five
		fi
		usleep 250000
	done
	echo_date "tailscaled进程启动成功，pid：${TSPID}"

	# 5. wait for tailscaled run for a while till get health status
	# 进程启动成功后，需要不断的去读取日志：/tmp/upload/tailscaled_log.txt 中的health字段来获取状态，本来应该用tailscale status去获取，但是会多运行一次golang程序，所以不用这个
	echo_date "🔴连接到Tailscale网络..."
	local TAILSD_FLAG
	local j=240
	while : ; do
		usleep 250000
		j=$(($j - 1))
		local CAS=$(echo $j|awk '{for(i=1;i<=NF;i++)if(!($i%5))print $i}')
		local HEALTH_FLAG=$(cat /tmp/upload/tailscaled_log.txt | grep "health" | awk -F": " '{print $NF}')
		
		if [ -n "${HEALTH_FLAG}" ];then
			TAILSD_FLAG="${HEALTH_FLAG}"
			echo_date "🟢成功连接到Tailscale网络！"
			break
		fi
		
		if [ -n "${CAS}" ];then
			echo_date "🔴连接到Tailscale网络..."
		fi
	done
		
	# 6. check for status using tailscale status
	local LOGED=$(cat /koolshare/configs/tailscale/tailscaled.state 2>/dev/null | grep -Eo "_profiles")
	if [ -n "${LOGED}" ];then
		# 已经授权过，登录后设置一次参数
		if [ "${tailscale_advertise_routes}" == "1" ];then
			echo_date "🆗开启宣告路由表（--advertise-routes），路由表网段：${IP_CIDR_2}！"
			run tailscale set --advertise-routes ${IP_CIDR_2}
		else
			echo_date "⛔️宣告路由表（--advertise-routes）未开启！"
			run tailscale set --advertise-routes ""
		fi
		if [ "${tailscale_accept_routes}" == "1" ];then
			echo_date "🆗开启接受路由表（--accept-routes）"
			run tailscale set --accept-routes
		else
			echo_date "⛔️接受路由表（--accept-routes）未开启"
			run tailscale set --accept-routes=false
		fi
		if [ "${tailscale_exit_node}" == "1" ];then
			echo_date "🆗开启互联网出口（--advertise-exit-node）"
			run tailscale set --advertise-exit-node
		else
			echo_date "⛔️互联网出口（--advertise-exit-node）未开启"
			run tailscale set --advertise-exit-node=false
		fi

		# 7. check tailscale ONLINE status
		echo_date "检测tailscale的IP地址..."
		local j=20
		until [ -n "${IPV4}" ]; do
			usleep 250000
			j=$(($j - 1))
			local IPV4=$(run tailscale ip -4)
			local IPV4=$(__valid_ip4 ${IPV4})
			
			if [ "$j" -lt 1 ]; then
				echo_date "tailscale在10s内没有获取到IP地址！请检查你的路由器网络是否畅通！"
				echo_date "在网络较差的情况下，可能需要等更久的时间，才能得到IP地址！"
				echo_date "插件将继续运行，运行完毕后，请注意插件界面的网口状态是否获取到IP地址！"
				dbus remove tailscale_ipv4
				break
			fi
		done

		local k=20
		until [ -n "${IPV6}" ]; do
			usleep 250000
			k=$(($k - 1))
			local IPV6=$(run tailscale ip -6)
			local IPV6=$(__valid_ip6 ${IPV6})
			
			if [ "$k" -lt 1 ]; then
				echo_date "tailscale在10s内没有获取到IP地址！请检查你的路由器网络是否畅通！"
				echo_date "在网络较差的情况下，可能需要等更久的时间，才能得到IP地址！"
				echo_date "插件将继续运行，运行完毕后，请注意插件界面的网口状态是否获取到IP地址！"
				dbus remove tailscale_ipv6
				break
			fi
		done	

		if [ -n "${IPV4}" -a -n "${IPV6}" ];then
			echo_date "成功连接tailscale网络:"
			echo_date "分配到IPV4地址：${IPV4}"
			echo_date "分配到IPV6地址：${IPV6}"
			dbus set tailscale_ipv4=${IPV4}
			tailscale_ipv4=${IPV4}
			dbus set tailscale_ipv6=${IPV6}
			tailscale_ipv6=${IPV6}
		fi

		# 8. check others
		echo_date "检测Subnets..."
		local SUBNET=$(cat /koolshare/configs/tailscale/tailscaled.state | jq -r '.[keys[] | select(contains("profile-"))]' | base64 -d | jq -r '.AdvertiseRoutes[0]')
		if [ -n "${SUBNET}" ];then
			echo_date "成功配置了Subnets: ${SUBNET}"
			echo_date "如希望远程设备能访问${SUBNET}局域网内的设备，请在控制台内允许该Subnets设定！"
		fi

		# 9. firewall
		sleep 2
		add_fw_rules

		# finish
		echo_date "Tailscale 插件启动完毕！"
	else
		# 未授权用户
		echo_date "准备加入tailnet网络..."

		local ARGS=""
		local ARGS="${ARGS} --snat-subnet-routes=false"
		if [ "${tailscale_advertise_routes}" == "1" ];then
			#echo_date "开启宣告路由表（--advertise-routes），路由表网段：${IP_CIDR_2}！"
			local ARGS="${ARGS} --advertise-routes=${IP_CIDR_2}"
		#else
			#echo_date "宣告路由表（--advertise-routes）未开启！"
		fi
		if [ "${tailscale_accept_routes}" == "1" ];then
			#echo_date "开启接受路由表（--accept-routes）"
			local ARGS="${ARGS} --accept-routes=true"
		#else
			#echo_date "接受路由表（--accept-routes）未开启"
		fi
		if [ "${tailscale_exit_node}" == "1" ];then
			#echo_date "开启互联网出口（--advertise-exit-node）"
			local ARGS="${ARGS} --advertise-exit-node"
		#else
			#echo_date "互联网出口（--advertise-exit-node）未开启"
		fi
		local ARGS="${ARGS} --accept-dns=false"

		run tailscale up $ARGS >/dev/null 2>&1 &
		echo_date "检测到你尚未授权，获取授权链接..."
		local AUTH_URL
		local j=40
		until [ -n "${AUTH_URL}" ]; do
			usleep 250000
			j=$(($j - 1))
			local AUTH_URL=$(cat ${TSD_LOG} | grep -E "AuthURL is" | tail -n1 | grep -Eo "https.*")
			
			if [ "$j" -lt 1 ]; then
				echo_date "Tailscale在10s内没有获得授权链接！请检查你的路由器网络是否畅通！"
				echo_date "关闭插件！"
				close_in_five
			fi
		done
		
		echo_date "请访问以下链接进行授权："
		echo
		echo "                  ${AUTH_URL}"
		echo
		echo XU6J03M6
		
		# 日志结束了，但是插件本质上还要继续等待
		# 后台需要一直等待用户完成授权的状态，等待60分钟
		local k=3600
		while : ; do
			sleep 1
			k=$(($k - 1))
			local IPV4=$(run tailscale ip -4)
			local IPV4=$(__valid_ip4 ${IPV4})
			if [ -n "${IPV4}" ];then
				echo_date "授权成功，继续..."
				break
			fi
			
			if [ "$k" -lt 1 ]; then
				echo_date "你在30分钟内未进行授权，插件将自行关闭！"
				echo_date "如果你已经授权了，请重新开启插件即可！"
				echo_date "-----------------------------------------"
				close_in_five
			fi
		done

		dbus set tailscale_ipv4=${IPV4}
		tailscale_ipv4=${IPV4}

		
		# check others
		echo_date "检测Subnets..."
		local SUBNET=$(cat /koolshare/configs/tailscale/tailscaled.state | jq -r '.[keys[] | select(contains("profile-"))]' | base64 -d | jq -r '.AdvertiseRoutes[0]')
		if [ -n "${SUBNET}" ];then
			echo_date "成功配置了Subnets: ${SUBNET}"
			echo_date "如希望远程设备能访问${SUBNET}局域网内的设备，请在控制台内允许该Subnets设定！"
		fi

		# firewall
		sleep 2
		add_fw_rules
	fi
}

del_fw_rule(){
	local IPTSV4=$(iptables -t filter -S | grep -w "tailscale_rule" | sed 's/-A/iptables -t filter -D/g')
	if [ -n "${IPTSV4}" ];then
		echo_date "关闭本插件的ipv4防火墙规则！"
		iptables -t filter -S | grep -w "tailscale_rule" | sed 's/-A/iptables -t filter -D/g' > /tmp/clean4.sh
		chmod +x /tmp/clean4.sh
		sh /tmp/clean4.sh > /dev/null 2>&1
		rm /tmp/clean4.sh
	fi
	local IPTSV6=$(ip6tables -t filter -S | grep -w "tailscale_rule" | sed 's/-A/ip6tables -t filter -D/g')
	if [ -n "${IPTSV6}" ];then
		echo_date "关闭本插件的ipv6防火墙规则！"
		ip6tables -t filter -S | grep -w "tailscale_rule" | sed 's/-A/ip6tables -t filter -D/g' > /tmp/clean6.sh
		chmod +x /tmp/clean6.sh
		sh /tmp/clean6.sh > /dev/null 2>&1
		rm /tmp/clean6.sh
	fi
}

add_fw_rules(){
	# 1. write DNAT, allow other visit 100.x.x.x
	echo_date "设置DNAT规则，以便tailnet中其它客户端通过 http://${IPV4} 访问本路由器..."
	local LANADDR=$(ifconfig br0|grep -Eo "inet addr.+"|awk -F ":| " '{print $3}' 2>/dev/null)
	local MATCH=$(iptables -t nat -S PREROUTING|grep tailscale_rule|grep ${tailscale_ipv4})
	if [ -n "${LANADDR}" -a -n "${tailscale_ipv4}" -a -z "${MATCH}" ];then
		iptables -t nat -A PREROUTING -d ${tailscale_ipv4} -j DNAT --to-destination ${LANADDR} -m comment --comment "tailscale_rule"
	fi

	# snat
	if [ -n "${LANADDR}" ];then
		iptables -t nat -A POSTROUTING ! -s ${LANADDR}/32 -o tailscale0 -j MASQUERADE -m comment --comment "tailscale_rule"
	fi
	
	# 2. alow incoming
	local DEVICE=$(ifconfig | grep tailscale|awk '{print $1}')
	if [ -n "${DEVICE}" ];then
		echo_date "设置防火墙规则，放行访问网卡：【${DEVICE}】的流量..."
		if [ "${tailscale_ipv4_enable}" == "0" ];then
			iptables -I INPUT -i ${DEVICE} -j DROP -m comment --comment "tailscale_rule" >/dev/null 2>&1
			#iptables -I OUTPUT -o ${DEVICE} -j DROP -m comment --comment "tailscale_rule" >/dev/null 2>&1
		else
			iptables -I INPUT -i ${DEVICE} -j ACCEPT -m comment --comment "tailscale_rule" >/dev/null 2>&1
			#iptables -I OUTPUT -o ${DEVICE} -j ACCEPT -m comment --comment "tailscale_rule" >/dev/null 2>&1
		fi
		
		if [ "${tailscale_ipv6_enable}" == "0" ];then
			ip6tables -I INPUT -i ${DEVICE} -j DROP -m comment --comment "tailscale_rule" >/dev/null 2>&1
			#ip6tables -I OUTPUT -o ${DEVICE} -j DROP -m comment --comment "tailscale_rule" >/dev/null 2>&1
		else
			ip6tables -I INPUT -i ${DEVICE} -j ACCEPT -m comment --comment "tailscale_rule" >/dev/null 2>&1
			#ip6tables -I OUTPUT -o ${DEVICE} -j ACCEPT -m comment --comment "tailscale_rule" >/dev/null 2>&1
		fi
		
		iptables -I FORWARD -i ${DEVICE} -j ACCEPT -m comment --comment "tailscale_rule" >/dev/null 2>&1
		ip6tables -I FORWARD -i ${DEVICE} -j ACCEPT -m comment --comment "tailscale_rule" >/dev/null 2>&1
	fi

	local PORTS_V6=$(netstat -nlp|grep tailscale|grep -E "^udp"|awk '{print $4}'|grep ":::"|sed -n 's/.*:\(\w\+\).*/\1/p'|head -n1)
	if [ -n "${PORTS_V6}" ];then
		for PORT_V6 in ${PORTS_V6}
		do
			if [ "${tailscale_ipv6_enable}" == "0" ];then
				echo_date "添加防火墙入站规则，关闭tailscale ipv6端口：${PORT_V6}"
				#ip6tables -I INPUT -p tcp --dport ${PORT_V6} -j DROP -m comment --comment "tailscale_rule" >/dev/null 2>&1
				ip6tables -I INPUT -p udp --dport ${PORT_V6} -j DROP -m comment --comment "tailscale_rule" >/dev/null 2>&1
			else
				echo_date "添加防火墙入站规则，打开tailscale ipv6端口：${PORT_V6}"
				#ip6tables -I INPUT -p tcp --dport ${PORT_V6} -j ACCEPT -m comment --comment "tailscale_rule" >/dev/null 2>&1
				ip6tables -I INPUT -p udp --dport ${PORT_V6} -j ACCEPT -m comment --comment "tailscale_rule" >/dev/null 2>&1
			fi
		done
	fi

	local PORTS_V4=$(netstat -nlp|grep tailscale|grep -E "^udp"|awk '{print $4}'|grep "0.0.0.0"|sed -n 's/.*:\(\w\+\).*/\1/p'|head -n1)
	if [ -n "${PORTS_V4}" ];then
		for PORT_V4 in ${PORTS_V4}
		do
			if [ "${tailscale_ipv4_enable}" == "0" ];then
				echo_date "添加防火墙入站规则，关闭tailscale ipv4端口：${PORT_V4}"
				#iptables -I IPUT -p tcp --dport ${PORT_V4} -j DROP -m comment --comment "tailscale_rule" >/dev/null 2>&1
				iptables -I INPUT -p udp --dport ${PORT_V4} -j DROP -m comment --comment "tailscale_rule" >/dev/null 2>&1
			else
				echo_date "添加防火墙入站规则，打开tailscale ipv4端口：${PORT_V4}"
				#iptables -I INPUT -p tcp --dport ${PORT_V4} -j ACCEPT -m comment --comment "tailscale_rule" >/dev/null 2>&1
				iptables -I INPUT -p udp --dport ${PORT_V4} -j ACCEPT -m comment --comment "tailscale_rule" >/dev/null 2>&1
			fi
		done
	fi
}

case $1 in
start)
	if [ "${tailscale_enable}" == "1" ]; then
		logger "[软件中心-开机自启]: tailscale自启动开启！"
		start_tailscale | tee -a ${LOG_FILE}
	else
		logger "tailscale插件未开启，跳过！"
	fi
	;;
start_nat)
	if [ "${tailscale_enable}" == "1" ]; then
		logger "[软件中心]-[${0##*/}]，NAT重启触发：打开tailscale防火墙端口！"
		del_fw_rule >/dev/null 2>&1
		add_fw_rules
	else
		logger "[软件中心]-[${0##*/}]，NAT重启触发：tailscale插件未开启，跳过！"
	fi
	;;
stop)
	stop_tailscale | tee -a ${LOG_FILE}
	;;
esac

case $2 in
web_submit)
	set_lock
	true > ${LOG_FILE}
	http_response "$1"
	# 调试
	# echo_date "$BASH $ARGS" | tee -a ${LOG_FILE}
	if [ "${tailscale_enable}" == "1" ]; then
		start_tailscale | tee -a ${LOG_FILE}
	else
		echo_date "停止tailscale！" | tee -a ${LOG_FILE}
		stop_tailscale | tee -a ${LOG_FILE}
	fi
	echo XU6J03M6 | tee -a ${LOG_FILE}
	unset_lock
	;;
esac
