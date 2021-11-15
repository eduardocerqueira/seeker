#date: 2021-11-15T16:56:29Z
#url: https://api.github.com/gists/7981dd32505b7846d8b007f16ececd08
#owner: https://api.github.com/users/swathinattuva

#!/bin/bash

if [ $# -eq 0 ] ; then
echo clone source-dir target-dir num hour/min/sec
else

echo Cloning "$1" to "$2"
vmx=`find "$1" -name '*vmx'`
if [ "$vmx" == "" ] ; then
		echo Could not find VMX
else
echo "$1" vmx is "$vmx"

if [ "$3" != "" ] ; then
declare -i sec
sec=0
		case "$4" in
		"hour")
				sec=`expr "$3" \* 3600`
		;;
		"min")
				sec=`expr "$3" \* 60`
		;;
		"sec")
				sec="$3"
		;;
		esac

		echo waiting for $sec
		sleep $sec
fi

declare -i start
declare -i now
start=`date +%s`
suspend="y"
state="`vmware-cmd "$vmx" getstate`"
if [ "$state" == "getstate() = suspended" ] ; then
   suspend="n"
fi

if [ "$suspend" == "y" ] ; then
if [ "suspend() = 1" == "`vmware-cmd "$vmx" suspend`" ] ; then
 suspend="n"
fi
fi

if [ "$suspend" == "n" ] ; then

while [ "getstate() = on" == "`vmware-cmd "$vmx" getstate`" ] ; do
		vmware-cmd "$vmx" getstate
		echo Waiitng for VM to suspend
		sleep 5
done

now=`date +%s`
echo Suspend took `expr $now - $start` seconds
start=`date +%s`

#create new dir
mkdir "$2"
cp -vr "$1"/* "$2"

now=`date +%s`
echo Copy took `expr $now - $start` seconds

vmware-cmd "$vmx" start

echo Clone complete
else
		echo Suspend failed
fi
fi
fi