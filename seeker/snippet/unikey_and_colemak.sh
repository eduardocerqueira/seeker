#date: 2021-12-03T16:54:55Z
#url: https://api.github.com/gists/867228d65fb9ce484f3d68af5441ed96
#owner: https://api.github.com/users/Tarrasch

#!/bin/bash -xe

# Usage:
#   Step 1: Test that commands work when running manually
#
#      example: /home/rouhani/unikey_and_colemak.sh enable_unikey
#
#   Step 2:
#
#      Map Desktop environment custom shortcuts to these commnads
#
#  /home/rouhani/unikey_and_colemak.sh enable_unikey
#  /home/rouhani/unikey_and_colemak.sh try_reset
#  /home/rouhani/unikey_and_colemak.sh set_ibus_engine_to_eng
#  /home/rouhani/unikey_and_colemak.sh replace_ibus_deamon

case $1 in
	enable_unikey)				
		ibus engine Unikey
		setxkbmap -variant colemak
		;;
	try_reset)				
		xkbcomp /home/rouhani/repos/keyboard-layout-gist/keymap.xkb :0 || xkbcomp /home/rouhani/repos/keyboard-layout-gist/keymap.xkb :1
		;;
	set_ibus_engine_to_eng)				
		ibus engine xkb:us:colemak:eng
		;;
	replace_ibus_deamon)
		ibus-daemon --daemonize --replace --xim
		;;
	*)
		echo 'unknown command noooo'
		;;
esac