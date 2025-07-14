#date: 2025-07-14T16:58:49Z
#url: https://api.github.com/gists/787e2b4af1e2420a9f9b6629489266c6
#owner: https://api.github.com/users/ForwardFeed

#! /bin/env bash

# I use the default microphone, 
# but at the end of the script, the default microphone will change (at least on my end)
# to the virt microphone for some reason that I may fix oneday
REAL_MIC=$(pactl get-default-source)
MIX_MIC="mix_mic"
VIRT_MIC_NAME="virt_mic"
VIRT_MIC_PATH="/tmp/$VIRT_MIC_NAME"

# On your real microphone
# create a sink
pactl load-module module-null-sink \
    sink_name=$MIX_MIC \
    sink_properties=device.description="Mix-for-Virtual-Microphone"
# create a loopback
pactl load-module module-loopback \
    sink=$MIX_MIC \
    latency_msec=20 \
    source="$REAL_MIC"
# On the virtual microphone
# create a source
pactl load-module module-pipe-source \
    source_name=$VIRT_MIC_NAME \
    file=$VIRT_MIC_PATH \
    format=s16le rate=16000 channels=1
# adds a loopback like with the real microphone
pactl load-module module-loopback \
    sink=$MIX_MIC \
    latency_msec=20 \
    source=$VIRT_MIC_NAME
# For the mix microphone
# combine both
pactl load-module module-combine-sink \
    sink_name=virtual-microphone-and-speakers \
    slaves=$MIX_MIC,$(pactl get-default-sink)
# and remap stuff, I don't understand that part but it works
pactl load-module module-remap-source \
    master=$MIX_MIC.monitor \
    source_properties=device.description=mixed-mic

# if you want to play a sound into the virtual microphone, i only managed with ffmpeg, but aplay or paplay exists too
# ffmpeg -re -i $INPUT_FILE -f s16le -ar 16000 -ac 1 - > $VIRT_MIC_PATH

function cleanup(){
    pactl unload-module module-pipe-source
    pactl unload-module module-null-sink
    pactl unload-module module-remap-source
    pactl unload-module module-loopback
    pactl unload-module module-combine-sink
}
