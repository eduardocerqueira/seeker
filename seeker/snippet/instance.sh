#date: 2023-02-21T16:48:25Z
#url: https://api.github.com/gists/f4eeae46d7c7505bd84a979e6563c327
#owner: https://api.github.com/users/listvin

#!/bin/bash
cd /opt/Telegram

function check-vpn {
	echo 'vpn?'
	nordvpn status | grep Connected
	return $?
}

function check-single {
	echo other telegrams?
	cnt=$(ps axu | grep Telegram | wc -l)
	if [[ $cnt -gt 1 ]]; then
		ps axu | grep --color=always Telegram
		return 1
	else
		return 0
	fi
}

function rem-tor {
	echo 'do not forget tor ;)'
}

function specials {
	case $1 in
		vaxton)
			check-vpn || exit 1
			check-single || exit 1
			rem-tor
			;;
		*)
			echo 'no special requirements'
			;;
	esac
}

if [ $# -lt 1 ]; then
	echo "expected profile name (~/.TelegramDesktop<ProfileName>)"
	echo "possible:"
	ls -a1 ~ | grep TelegramDesktop | cut -c 17- | grep -E '^[^\s]+'
	echo "  or default"
	exit 1
fi

if [ "$1" == "-c" ]; then
	c=1
	shift
else
	c=0
fi

profile=$1

specials $1

if [[ "$profile" == "default" ]]; then
	profile=""
fi

path=~/.TelegramDesktop$profile

if ! [ -d $path ]; then
	if [ $c -eq 1 ]; then
		mkdir "$path"
	else
		echo "not found '$path', to create add -c in the beginning"
		exit 2
	fi
fi

screen -dmS Telegram-$profile ./Telegram -many -workdir $path
