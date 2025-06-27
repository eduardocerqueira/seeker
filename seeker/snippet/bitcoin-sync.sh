#date: 2025-06-27T17:12:20Z
#url: https://api.github.com/gists/a1308b0267ba583243c4125df4e95734
#owner: https://api.github.com/users/eugenesan

#!/bin/bash

# Interacive Bitcoin Daemon / Electrum Server sync manager for Linux (MacOS should work with tweaks)
sversion=4.17

# Usage: ${0} [electrumx|electrs|fulcrum|testnet]
# Default is no electrum server (bitcoin daemon only)

# Enable fancy debug output
export PS4=$'+\e[33m\t ${LINENO} ${FUNCNAME[0]:-MAIN}() \e[0m+  '

# Runtime options (during pauses):
# [p]eriodic   - set periodic mode (default)
# [c]ontinuous - set continuous mode
# [s]kip       - set periodic mode and skip current sync cycle
# [f]orce      - set periodic mode and force sync once
# [q]uit       - stop daemons and quit
# [a]udio      - togle audio notifications (default on)


### Definitions

# Color output dfinitions
C='\033[0;33m'
B='\033[0;36m'
E='\033[0;35m'
R='\033[0;31m'
N='\033[0m'

startdelay=5		# timeout for user input prompt
daemondelay=5		# delay for newly started services
stopdelay=600		# timeout when trying to stop/kill services
checkdelay=60		# delay between status check during sync
cycledelay=10800	# delay between sync cycles
retrydelay=300		# delay before retrying cycle due to runtime conditions
freshblock=1200		# latest block max age
anouncefreq=5		# how frequently to anounce status in min

runmode="periodic"	# continuous is another option
audiomode="on"		# enable audio notifications
onacpower="true"	# set to false to run on battery power
runload="false"		# enable to simulate system load when running testnet
audioprog="play"	# festival

skiponce="false"	# internal runtime variable
forceonce="false"	# internal runtime variable

nice="nice -n 15 ionice -t -c 3"	# prefix command to start service with niceness

# Set Bitcoin Daemon parameters based on variant installed (Snap/Native/Knots)
bitcoindhome="${HOME}/snap/bitcoin-core/common/.bitcoin"
test -d "${bitcoindhome}" || bitcoindhome="${HOME}/.bitcoin"
bitcoindbin="bitcoind"
bitcoindcmd="bitcoin-core.daemon"
which "${bitcoindcmd}" || bitcoindcmd="bitcoind"
bitcoindrpc="bitcoin-core.cli"
which "${bitcoindrpc}" || bitcoindrpc="bitcoin-cli"
bitcoindinitmsg="init message: Done loading" # initload thread exit msg
bitcoindwallettxmsg="AddToWallet" # wallet changed msg
#bitcoindopt="-disablewallet" # optional bitcoin daemon options (ex. disable bitcoin wallets when running with electrum server)

# Check if testnet was requested
if [ "${1}" = "testnet" ]; then
	bitcoindnet="-testnet4"
	bitcoindlog="${bitcoindhome}/testnet4/debug.log"
	bitcoinsynclog="${bitcoindhome}/testnet4/sync.log"
else
	bitcoindlog="${bitcoindhome}/debug.log"
	bitcoinsynclog="${bitcoindhome}/sync.log"
fi

# Choose electrum server
if [ "${1}" = "electrs" ]; then
	electrumdhome="${HOME}/.electrs"
	electrumdbin="electrs"
	electrumdrpc=""
elif [ "${1}" = "electrumx" ]; then
	electrumdhome="${HOME}/.electrumx"
	electrumdbin="electrumx_server"
	electrumdrpc="electrumx_rpc"
	electrumddb="leveldb"
elif [ "${1}" = "fulcrum" ]; then
	electrumdhome="${HOME}/.fulcrum"
	electrumdbin="Fulcrum"
	electrumdrpc="FulcrumAdmin"
	electrumdrpcopt="-p 8000"
else
	electrumdhome=""
	electrumdbin=""
	electrumdrpc=""
	electrumdrpcopt=""
fi

electrumdlog="${electrumdhome}/debug.log"

# Define script requirements
reqs=( "${bitcoindcmd}" "${bitcoindrpc}" "${electrumdbin}" "${electrumdrpc}" "${audioprog}" "7z" "on_ac_power" "nm-online" "wget" )

### Functions

aprint() {
	if [[ "${audiomode}" = "on" || "${2}" = "R" ]]; then
		if [ "${audioprog}" = "festival" ]; then
			# Filter msg for voice anouncement
			vmsg=$(echo -en "${1}" | sed 's/[Bb]itcoin\ [Cc]ore/B\ C,/' | sed 's/[Ee]lectrum\ [Ss]erver/E\ S,/' | sed 's/\[.*\]//')
			# Speak the msg
			echo -en "${vmsg}" | festival --tts &
		elif [ "${audioprog}" = "play" ]; then
			# Incoming sequence tone
			if [ "${2}" = "E" ]; then
				play -q -n -c1 synth sin %-12 sin %-9 sin %-15 sin %-12 fade h 0.1 0.2 vol 0.33
				#play -q -n -c1 synth 0.1 sin 480 vol 0.25
			elif  [ "${2}" = "R" ]; then
				play -q -n -c1 synth sin %40 sin %9 sin %15 sin %-12 fade h 0.1 0.2 vol 0.50
				echo -en "\007\007\007\007"
			fi

			# Main tone
			#echo -n '='
			play -q -n -c1 synth sin %-2 sin %-9 sin %-15 sin %-12 fade h 0.1 0.2 vol 0.33
			#play -q -n -c1 synth 0.1 sin 280 vol 0.25

			# Outgoing sequence tone
			if [ "${2}" = "F" ]; then
				#echo -n '<'
				play -q -n -c1 synth sin %-12 sin %-9 sin %-15 sin %-12 fade h 0.1 0.2 vol 0.33
				#play -q -n -c1 synth 0.1 sin 480 vol 0.25
			elif  [ "${2}" = "R" ]; then
				play -q -n -c1 synth sin %-5 sin %9 sin %-15 sin %-20 fade h 0.1 0.2 vol 0.50
				echo -en "\007\007\007\007"
			fi
		fi
	fi
}

iprint() {
	case ${2} in
		A)
		aprint "${1}" "${2}"
	;;
		B)
		echo -en "${B}"
		echo -e "[$(date "+%Y-%m-%d %H-%M-%S")] ${1}${N}"
	;;
		E|F)
		echo -en "${E}"
		echo -e "[$(date "+%Y-%m-%d %H-%M-%S")] ${1}${N}"
	;;
		R)
		echo -en "${R}"
		echo -e "[$(date "+%Y-%m-%d %H-%M-%S")] ${1}${N}"
	;;
		C|*)
		echo -en "${C}"
		echo -e "[$(date "+%Y-%m-%d %H-%M-%S")] ${1}${N}"
	;;
	esac

	if [ "${3}" = "A" ]; then
		aprint "${1}" "${2}"
	fi
}

ipause() {
	# Save pause start time and initial runmode
	pausestart=$(date -u +%s)
	irunmode="${runmode}"

	# Output prompt if provided
	if [ -n "${2}" ]; then
		iprint "${2} [${1}s]..." B
	fi

	while : ; do
		pauseremain=$((${1} - ($(date -u +%s) - pausestart)))

		read -t "${pauseremain}" -s -N 1 input

		case "${input}" in
			c)
				iprint "Continuous mode" B A
				runmode="continuous"
				audiomode="off"
			;;
			p)
				iprint "Periodic mode" B A
				runmode="periodic"
				audiomode="on"
				(( syncdelay=cycledelay ))
			;;
			s)
				iprint "Periodic mode, Skip once" B A
				runmode="periodic"
				skiponce="true"
				forceonce="false"
				(( syncdelay=cycledelay ))
			;;
			f)
				iprint "Periodic mode, Force once" B A
				runmode="periodic"
				forceonce="true"
				skiponce="false"
			;;
			q|x)
				iprint "Quiting..." E A
				istop
				cycleduration=$(($(date -u +%s) - cyclestart))
				iprint "Bitcoin Daemon and Electrum Server updated, fetched ${bdiffblocks} blocks in $((cycleduration / 60)) minutes." E A
				exit 0
			;;
			a)
				if [ "${audiomode}" = "on" ]; then
					audiomode="off"
				else
					audiomode="on"
				fi
				iprint "Audio ${audiomode}." E A
			;;
		esac

		# Break pause if time is up or runmode was changed to requested value
		if [[ $(($(date -u +%s) - pausestart)) -ge ${1} || ("${runmode}" != "${irunmode}") || "${forceonce}" = "true" || "${skiponce}" = "true" ]]; then
			break
		fi
	done
}

irotate() {
	if [ -s ${1} ]; then
		logdate="$(date '+%Y-%m-%d_%H-%M-%S' --date @$(stat -c %Y ${1}))"
		mv -f "${1}" "${1}.${logdate}"
		7z -t7z -slp -m0=lzma2 -myx=9 -mx=9 -mfb=256 -md=256m -mlc=4 -sdel -bd a ${1}.7z ${1}.${logdate} > /dev/null
	fi
}

ionline() {
	if nm-online -q -t 5; then
		if wget -q --spider -T 5 http://google.com; then
			iprint "Internet (google.com) is reachable." E
			return 0
		else
			iprint "Internet (google.com) is unreachable." R
			return 1
		fi
	else
		iprint "Internet (google.com) is unreachable." R
		return 1
	fi
}

ibitcoind() {
	# Rotate log
	irotate "${bitcoindlog}"

	# Run Bitcoin Daemon
	MALLOC_ARENA_MAX=1 ${nice} ${bitcoindcmd} ${bitcoindnet} ${bitcoindopt} &
}

ielectrs() {
	# Install Electrs: cargo install elecrs

	# Rotate log
	irotate "${electrumdlog}"

	# Run Electrs
	${nice} ${electrumdbin} 2>&1 | tee "${electrumdlog}" &
}

ifulcrum() {
	# Install Fulcrum
	# git clone https://github.com/cculianu/Fulcrum.git
	# sed 's/x86_64\)/x86_64x\)' Fulcrum.pro
	# /usr/include/rocksdb/compression_type.h
	# sed -i 's/kNoCompression/kZSTD/g' src/Storage.cpp
	# sed 's/kNoCompression/kSnappyCompression/g' src/Storage.cpp
	# mkdir -p build; cd build
	# qmake ../Fulcrum.pro "CONFIG-=debug" "CONFIG+=release" QMAKE_LINK=clang++ QMAKE_CXX=clang++ QMAKE_CC=clang
	# make -j4
	# https://github.com/cculianu/Fulcrum/blob/master/doc/fulcrum-example-config.conf

	# Rotate log
	irotate "${electrumdlog}"

	# Run Fulcrum
	${nice} ${electrumdbin} "${electrumdhome}/fulcrum.conf" 2>> ${electrumdlog}| tee ${electrumdlog} &
}

ielectrumx() {
	# Install ElectrumX
	# git clone https://github.com/spesmilo/electrumx.git
	# pip3 install .[plyvel,rocksdb,ujson]

	# Configure ElectrumX
	export COIN=Bitcoin
	#export DAEMON_URL=http://bitcoin:bitcoin@localhost:8332
	export DAEMON_URL=http://$(cat ${bitcoindhome}/.cookie)@localhost:8332
	export NET=mainnet
	export CACHE_MB=512 # 256
	export MAX_SESSIONS=4 # 16
	#export INITIAL_CONCURRENT=16
	#export DB_ENGINE=leveldb # rocksdb
	#export DB_DIRECTORY=${electrumdhome}/bitcoin-leveldb # -rocksdb
	export DB_ENGINE=${electrumddb}
	export DB_DIRECTORY=${electrumdhome}/bitcoin-${electrumddb}
	export SSL_CERTFILE=${electrumdhome}/certs/server.crt
	export SSL_KEYFILE=${electrumdhome}/certs/server.key
	#export BANNER_FILE=${electrumdhome}/banner
	export DONATION_ADDRESS=bc1thisaddressisnotrealdonotsendfundstoit
	export ANON_LOGS=freeloadersarenotwelcome
	#xport LOG_LEVEL=debug
	export SERVICES="rpc://localhost,tcp://localhost:8335,ssl://localhost:8336"
	#export FORCE_PROXY=yes
	#export TOR_PROXY_HOST=localhost
	#export TOR_PROXY_PORT=9050
	export PEER_DISCOVERY=self
	export PEER_ANNOUNCE=

	# Rotate log
	irotate "${electrumdlog}"

	# Run ElectrumX
	${nice} ${electrumdbin} 2>> "${electrumdlog}" | tee "${electrumdlog}" &
}

ielectrumd_start() {
	if [ -z "${electrumdbin}" ]; then
		return
	fi

	if [ -z "${electrumdrpc}" ]; then
		ielectrs
	elif [ -z "${electrumdrpcopt}" ]; then
		ielectrumx
	else
		ifulcrum
	fi
}

ielectrumd_stop() {
	if [ -z "${electrumdbin}" ]; then
		return
	fi

	if [ -z "${electrumdrpc}" ]; then
		pkill -ex --signal 15 ${electrumdbin:0:15}
	#elif [ -z "${electrumdrpcopt}" ]; then
	#	${electrumdrpc} ${electrumdrpcopt} stop
	else
		${electrumdrpc} ${electrumdrpcopt} stop
	fi
}

ielectrumd_status() {
	if [ -z "${electrumdbin}" ]; then
		return ${1}
	fi

	if [ -z "${electrumdrpc}" ]; then
			# {"id":0,"jsonrpc":"2.0","result":["electrs/0.10.1","1.4"]}
			test -n "$(echo '{"jsonrpc": "2.0", "method": "server.version", "params": ["", "1.4"], "id": 0}' | nc -N 127.0.0.1 8335 | grep result | cut -d'[' -f2 | cut -d'"' -f2)"
			local edstat=$?
	else
			${electrumdrpc} ${electrumdrpcopt} getinfo &> /dev/null
			local edstat=$?
	fi

	return ${edstat}
}

ielectrumd_last() {
	local ed_last
	if [ -z "${electrumdbin}" ]; then
		echo "${1}"
		return
	fi

	if [ -z "${electrumdrpc}" ]; then
		# {"id":0,"jsonrpc":"2.0","result":{"height":815895,"hex":"0060be...224f10"}}
		ed_last="$(echo '{"jsonrpc": "2.0", "method": "blockchain.headers.subscribe", "params": ["", "1.4"], "id": 0}' | nc -N 127.0.0.1 8335 | grep result | cut -d'{' -f3 | cut -d',' -f1 | cut -d':' -f2)"
	#elif [ -z "${electrumdrpcopt}" ]; then
	#	ielectrumx
	else
		ed_last="$(${electrumdrpc} ${electrumdrpcopt} getinfo | grep "height" | tail -n1 | cut -d":" -f2 | cut -d"," -f1 | xargs)"
	fi

	# Return 0 if result is empty
	if [ -z "${ed_last}" ]; then
		echo "0"
	else
		echo "${ed_last}"
	fi
}

istart() {
	# Stop daemons just in case
	iprint "Stopping Bitcoin Daemon and Electrum Server..."
	istop

	# Start bitcoin
	iprint "Starting Bitcoin Daemon..."
	ibitcoind
	iprint "Waiting for Bitcoin Daemon to start up..."
	ibfirstblock=$(${bitcoindrpc} ${bitcoindnet} -rpcwait getblockchaininfo | grep '\"pruneheight\"' | cut -d":" -f2 | cut -d"," -f1 | xargs)
	[ -z "${ibfirstblock}" ] && ibfirstblock=0
	iblastblock=$(${bitcoindrpc} ${bitcoindnet} -rpcwait getblockchaininfo | grep '\"blocks\"' | cut -d":" -f2 | cut -d"," -f1 | xargs)
	iprint "Bitcoin Daemon is up. Local blocks: [${ibfirstblock}..${iblastblock}]."

	# Start electrumx
	iprint "Starting Electrum Server..." C
	ielectrumd_start

	# Let daemons a chance to settle
	sleep ${daemondelay}

	esdelay=0
	until ielectrumd_status 0; do
		if [ ${esdelay} -gt 11 ]; then
			(( esdelay=11 ))
		else
			(( esdelay++ ))
		fi

		ipause $((esdelay * daemondelay)) "Waiting for Electrum Server to start up"

		# For fulcrum check Prometheus Monitoring status
		#[[ "${electrumdbin}" = "Fulcrum" && -n "printf 'GET /stats HTTP/1.1\r\nHost: localhost\r\n\r\n' | nc localhost 8001 | grep '"error"' | grep 'time period'" ]]
		if [ ${esdelay} -gt 3 ]; then
			iprint "Electrum Server might be doing ininital sync."
		fi

		cblastblock=$(${bitcoindrpc} ${bitcoindnet} -rpcwait getblockchaininfo | grep '\"blocks\"'| head -n 1  | cut -d":" -f2 | cut -d"," -f1 | xargs)
		cbclastblock=$(${bitcoindrpc} ${bitcoindnet} -rpcwait getchaintips | grep '\"height\"' | head -n 1 | cut -d":" -f2 | cut -d"," -f1 | xargs)
		iprint "Bitcoin Daemon last block is [${cblastblock} of ${cbclastblock}]."
	done

	ielastblock=$(ielectrumd_last ${iblastblock})
	iprint "Electrum Server is up. Last block [${ielastblock}]."
}

istop() {
	# Stop electrum server
	if  [ -n "${electrumdbin}" ] && pgrep -x ${electrumdbin:0:15} > /dev/null; then
		iprint "Stopping Electrum Server RPC..."
		ielectrumd_stop

		sleep ${daemondelay}
		while ielectrumd_status 1; do
			iprint "Waiting for Electrum Server RPC to shutdown..."
			sleep ${daemondelay}
		done
		iprint "Electrum Server RPC is down."

		# Waiting until electrumx server fully shuts down
		start_ts=$EPOCHSECONDS
		if [ ${electrumdbin} = "electrumx_server" ]; then
			iprint "Waiting for ElectrumX Server to shutdown (${stopdelay}s)..."
			while ! grep "^INFO" ${electrumdlog} | tail -n1 | grep "ElectrumX server terminated normally" > /dev/null; do
				if (( EPOCHSECONDS-start_ts > stopdelay )); then
					iprint "Timeout, waiting for ElectrumX Server to shutdown..."
					break;
				fi
				sleep ${daemondelay}
			done
		elif [ ${electrumdbin} = "Fulcrum" ]; then
			iprint "Waiting for Fulcrum Server to shutdown (${stopdelay}s)..."
			while ! tail -n1 "${electrumdlog}" | grep "Shutdown complete" > /dev/null; do
				if (( EPOCHSECONDS-start_ts > stopdelay )); then
					iprint "Timeout, waiting for Fulcrum Server to shutdown..."
					break;
				fi
				sleep ${daemondelay}
			done
		fi

		# Make sure electrum server is really down
		if pgrep -x ${electrumdbin:0:15} > /dev/null; then
			iprint "Electrum Server is still up. Attempt to terminate Electrum Server..."
			pkill -ex ${electrumdbin:0:15}
			iprint "Waiting for Electrum Server to terminate (${stopdelay}s)..."
			while pgrep -x ${electrumdbin:0:15} > /dev/null; do
				if (( EPOCHSECONDS-start_ts > stopdelay )); then
					iprint "Timeout, waiting for Electrum Server to terminate..."
					break;
				fi
				sleep ${daemondelay}
			done
		fi

		if pgrep -x ${electrumdbin:0:15} > /dev/null; then
			iprint "Electrum Server is still up." E A
			iprint "Attempt to kill Electrum Server..."
			pkill -ex -9 ${electrumdbin:0:15}
			iprint "Waiting for Electrum Server to die (${stopdelay}s)..."
			while pgrep -x ${electrumdbin:0:15} > /dev/null; do
				if (( EPOCHSECONDS-start_ts > stopdelay )); then
					iprint "Timeout, waiting for Electrum Server to die..."
					break;
				fi
				sleep ${daemondelay}
			done
		fi

		if pgrep -x ${electrumdbin:0:15} > /dev/null; then
			iprint "Something is wrong. Electrum Server is still up." R A
		fi
	fi
	iprint "Electrum Server is down."

	# Stop bitcoin daemon
	if pgrep -x ${bitcoindbin:0:15} > /dev/null; then
        start_ts=$EPOCHSECONDS
		iprint "Waiting for Bitcoin Daemon to finish init (${stopdelay}s)..."
		while ! grep "${bitcoindinitmsg}" "${bitcoindlog}" > /dev/null; do
			if (( EPOCHSECONDS-start_ts > stopdelay )); then
				iprint "Timeout, Waiting for Bitcoin Daemon to finish init..."
				break;
			fi
			sleep ${daemondelay}
		done

		iprint "Stopping Bitcoin Daemon..."
		${bitcoindrpc} ${bitcoindnet} stop
		sleep ${daemondelay}

		start_ts=$EPOCHSECONDS
		while ${bitcoindrpc} ${bitcoindnet} -getinfo &> /dev/null; do
			iprint "Waiting for Bitcoin Daemon RPC to shutdown..."
			if (( EPOCHSECONDS-start_ts > stopdelay )); then
				iprint "Timeout, waiting for Bitcoin Daemon RPC to shutdown..."
				break;
			fi
			sleep ${daemondelay}
		done

		iprint "Bitcoin Daemon RPC is down."

		# Waiting until daemon fully shuts down
		iprint "Waiting for Bitcoin Daemon to fully shutdown..."
		start_ts=$EPOCHSECONDS
		while ! tail -n1 "${bitcoindlog}" | grep "Shutdown: done" > /dev/null; do
			if (( EPOCHSECONDS-start_ts > stopdelay )); then
				iprint "Timeout, waiting for Bitcoin Daemon to shutdown..."
				break;
			fi
			sleep ${daemondelay}
		done

		# Make sure bitcoin is really down
		if pgrep -x ${bitcoindbin:0:15} > /dev/null; then
			iprint "Bitcoin Daemon is still up. Attempting to terminate..."
			pkill -ex ${bitcoindbin:0:15}
			while pgrep -x ${bitcoindbin:0:15} > /dev/null; do
				if (( EPOCHSECONDS-start_ts > stopdelay )); then
					iprint "Timeout, waiting for Bitcoin Daemon to terminate..."
					break;
				fi
				sleep ${daemondelay}
			done

			iprint "Bitcoin Daemon is still up. Attempting to kill..."
			pkill -ex -9 ${bitcoindbin:0:15}
			while pgrep -x ${bitcoindbin:0:15} > /dev/null; do
				if (( EPOCHSECONDS-start_ts > stopdelay )); then
					iprint "Timeout, waiting for Bitcoin Daemon to die..."
					break;
				fi
				sleep ${daemondelay}
			done

			if pgrep -x ${bitcoindbin:0:15} > /dev/null; then
				iprint "Something is wrong. Bitcoin Daemon is still up. Aborting..." R A
				exit 1
			fi
		fi
	fi
	iprint "Bitcoin Daemon is down."
}

icheckreqs() {
	# Check availability of required binaries
	for req in ${reqs[@]}; do
		if [ -z "$(which ${req})" ]; then
			iprint "Missing (${req})." R
			missingreq="true"
		#else
		#	iprint "Found (${req})." B
		fi
	done

	# Abort if requirements are not available
	if [ "${missingreq}" = "true" ]; then
		iprint "Missing requirements, aborting..." R
		exit 1
	fi
}


### Main

echo -e "${N}"
clear

# Rotate log
irotate "${bitcoinsynclog}"

# Log all output
exec > >(tee "${bitcoinsynclog}") 2>&1

# Welcome
iprint "Welcome to Bitcoin Daemon [${bitcoindcmd}] and Electrum Server [${electrumdbin}] sync manager v${sversion}."

# Check requirements
icheckreqs

while true; do
	# Save cycle starting time
	cyclestart=$(date -u +%s)
	iblastblock=0
	bdiffblocks=0

	if ( [ "${onacpower}" = "true" ] && ! on_ac_power ) || (! ionline) ; then
		skiponce=true
		(( syncdelay=retrydelay ))
		iprint "Skipping sync cycle due to missing AC-Power or Internet. Will retry in $(( syncdelay / 60 ))m. To run anyway, use [f]orce option." B
	else
		(( syncdelay=cycledelay ))
		iprint "Running with AC-Power and Internet. Set sync cycle to [$(( syncdelay / 60 ))]m." E
	fi

	iprint "Available runtime options [p]eriodic, [c]ontinuous, [s]kip, [f]orce, [q]uit, [a]udio." B
	ipause ${startdelay}

	if [ "${skiponce}" != "true" ]; then
		# Anounce sync start
		iprint "Initiating Bitcoin Daemon and Electrum Server ${runmode} sync." E A

		# Start daemons
		istart
	else
		# Anounce sync skip
		iprint "Skipping Bitcoin Daemon and Electrum Server ${runmode} sync once." E A
	fi

	# Check if daemon and electrum server are synced and shutdown them until next re-sync
	syncloop=0
	bitcoindwallettx=""
	while [ "${skiponce}" != "true" ]; do
		# Enable skiponce if forced
		if [ "${skiponce}" = "force" ]; then
			skiponce="true"
		fi

		# Check bitcoin daemon last block status
		iprint "Fetching Bitcoin Daemon status..."
		cbsize=$(${bitcoindrpc} ${bitcoindnet} -rpcwait getblockchaininfo | grep '\"size_on_disk\"'| head -n 1  | cut -d":" -f2 | cut -d"," -f1 | xargs)
		cbfirstblock=$(${bitcoindrpc} ${bitcoindnet} -rpcwait getblockchaininfo | grep '\"pruneheight\"'| head -n 1  | cut -d":" -f2 | cut -d"," -f1 | xargs)
		[ -z "${cbfirstblock}" ] && cbfirstblock=0
		blocktime=$(${bitcoindrpc} ${bitcoindnet} -rpcwait getblockchaininfo | grep '\"time\"' | head -n 1 | cut -d":" -f2 | cut -d"," -f1 | xargs)
		timediff=$(($(date -u +%s) - blocktime))
		cycleduration=$(($(date -u +%s) - cyclestart))
		cbclastblock=$(${bitcoindrpc} ${bitcoindnet} -rpcwait getchaintips | grep '\"height\"' | head -n 1 | cut -d":" -f2 | cut -d"," -f1 | xargs)
		cblastblock=$(${bitcoindrpc} ${bitcoindnet} -rpcwait getblockchaininfo | grep '\"blocks\"'| head -n 1  | cut -d":" -f2 | cut -d"," -f1 | xargs)
		bdiffblocks=$((cblastblock - iblastblock))
		bcdiffblocks=$((cbclastblock - iblastblock))

		# Inform of current sync cycle state
		anounceaudio="$([[ ${syncloop} = 1 || $((syncloop % anouncefreq)) = 0 ]] && echo 'A')"
		iprint "Bitcoin Daemon fetched [${bdiffblocks}/${bcdiffblocks}] blocks in $((cycleduration / 60 )) minutes. Local blocks: [${cbfirstblock}..${cblastblock}] are $((timediff / 60 ))m old]. On-chain: ${cbclastblock}. Size: $((cbsize / 1024 / 1024 / 1024 ))GB." C "${anounceaudio}"

		# Fetch sync status between bitcoin daemon and electrum server
		iprint "Fetching Electrum Server status..."
		celastblock=$(ielectrumd_last "${cblastblock}")
		ediffblocks=$((cblastblock - celastblock))
		iprint "Electrum server [${celastblock}] is ${ediffblocks} blocks behind."

		# Check last bitcoin daemon block freshness
		if [[ (${timediff} -le ${freshblock}) || ( ( ${timediff} -le $((freshblock * 2)) ) && (${cblastblock} = "${cbclastblock}") ) ]]; then
				bcfreshblock=true
		else
				bcfreshblock=false
		fi

		if [ "${onacpower}" = "true" ] && [ "${forceonce}" = "false" ] && ! on_ac_power; then
			skiponce=true
			(( syncdelay=retrydelay ))
			iprint "Aborting sync cycle due to missing AC power. Will retry in $(( syncdelay / 60 ))m" B
		fi

		bitcoindwallettx_new="$(grep ${bitcoindwallettxmsg} ${bitcoindlog})"
		if [ "${bitcoindwallettx}" != "${bitcoindwallettx_new}" ]; then
			iprint "Bitcoin Daemon found changes in loaded wallet" R A
			bitcoindwallettx="${bitcoindwallettx_new}"
		fi

		# If last block is fresh&synced or mode/skip requires it, stop daemons
		if [[ ("${bcfreshblock}" = "true" && "${ediffblocks}" = "0" && "${runmode}" = "periodic") || "${skiponce}" = "true" ]]; then
			iprint "Bitcoin Daemon and Electrum Server both fetched fresh block [${cblastblock} = ${celastblock}]."

			# Stop daemons
			iprint "Stopping Bitcoin Daemon and Electrum Server..." C A
			istop

			skiponce="false"
			break
		else
			if [ "${runmode}" = "periodic" ]; then
				iprint "Not yet synced, will re-check in ${checkdelay} seconds..."
			else
				iprint "Continuous mode, will report status every ${checkdelay} seconds..."
			fi

			ipause ${checkdelay}

			# Force another cycle to stop daemons
			if [ "${skiponce}" = "true" ]; then
				skiponce="force"
			fi
		fi

		(( syncloop++ ))
	done

	# Check if need to stop daemons and reset state
	if [ "${skiponce}" = "true" ]; then
		# Stop daemons
		iprint "Stopping Bitcoin Daemon and Electrum Server..." C A
		istop
	fi
	skiponce="false"

	# Anounce end of cycle
	synctime=$(($(date -u +%s) + syncdelay))
	cycleduration=$(($(date -u +%s) - cyclestart))

	if [ "${runload}" = "true" ]; then
		iprint "7zip load started." E A
		7z a -mx=1 runload.7z ~/.local/bin
		cycleduration2=$(($(date -u +%s) - ${cyclestart}))
		iprint "7zip load finished in $(( (${cycleduration2} - ${cycleduration}) / 60))m." F A
	fi

	iprint "Bitcoin Daemon and Electrum Server are synced [${cbfirstblock}..${cblastblock}]/[${cbclastblock}]. Fetched [${bdiffblocks}] blocks in $((cycleduration / 60))m. Next update at $(date --date="@${synctime}" "+%_H:%M") [$(date --date="@${synctime}" "+%Y-%m-%d %H-%M-%S")]." F A

	while true; do
		ipause ${checkdelay}
		timeleft=$((synctime - $(date -u +%s)))
		if [[ ${timeleft} -le 0 || "${runmode}" = "continuous" || "${forceonce}" = "true" ]]; then
			forceonce="false"
			break
		fi
	done
done
