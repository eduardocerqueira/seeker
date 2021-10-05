#date: 2021-10-05T17:07:45Z
#url: https://api.github.com/gists/8865865aac4613c1ace492aa9f04231a
#owner: https://api.github.com/users/imneonizer

#!/bin/bash

export ANDROID_SERIAL=`adb devices -l | grep -i oneplus | awk '{print $1}'`
export HOST_PORT=8080
export IPWEBCAM_PORT=8080
export V4L2_DEVICE=/dev/video0

# start screen mirror service
if [[ $1 == '-s' ]];then
    if [ ! `pgrep scrcpy` ];then
        echo 'starting screen mirroring'
        scrcpy
    fi
    exit
fi


# kill already running ffmpeg pipe
if [[ $1 == '-k' ]];then
    PID=`pgrep -f 'ipwebcam.*-f'`
    if [[ $PID ]];then
        kill -9 $PID
        kill -9 `pgrep -f 'ffmpeg.*v4l2'`
        echo 'ffmpeg-v4l2loopback killed'
    fi
    exit
fi

# start ffmpeg to write to v4l2loopback device
if [[ $1 == '-f' ]];then
    while :
    do
        if [ ! `pgrep -f 'ffmpeg.*v4l2'` ];then
            echo 'starting ffmpeg'
            # forward port from android to host system
            adb forward tcp:$HOST_PORT tcp:$IPWEBCAM_PORT
            modprobe v4l2loopback
            ffmpeg -i http://localhost:$HOST_PORT/videofeed -vf format=yuv420p -f v4l2 $V4L2_DEVICE
        else
            echo 'ffmpeg-v4l2loopback already running'
            exit
        fi
        sleep 1
    done
    exit
fi


# print help message
function print_help(){
    echo "usage: $0 [-s|-f|-k]"
    echo "-s    start screen mirroring"
    echo "-f    start ffmpeg-v4l2loopback"
    echo "-k    kill ffmpeg-v4l2loopback"
}

if [[ $1 == '-h' ]];then
    print_help
else
    print_help
fi