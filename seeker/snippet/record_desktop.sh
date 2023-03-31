#date: 2023-03-31T16:51:46Z
#url: https://api.github.com/gists/9b80d5f9e9608914a24e3506b06ea483
#owner: https://api.github.com/users/sug0

#!/bin/sh

set -e

main() {
    parse_args $@
    capture_screen
}

parse_args() {
    while true; do
        case $1 in
        -select)
            selection=1
            ;;
        -audio)
            audio=1
            ;;
        *)
            output="${1:-output.mp4}"
            if [ -n "$audio" ]; then
                audio_args="-f pulse -i default -ac 2"
            fi
            if [ -z "$selection" ]; then
                start_pos=0,0
            else
                selection=$(slop -f "%wx%h %x,%y")
                video_size="-video_size $(echo $selection | cut -d' ' -f1)"
                start_pos=$(echo $selection | cut -d' ' -f2)
            fi
            unset audio
            unset fullscreen
            unset selection
            return
            ;;
        esac
        shift
    done
}

capture_screen() {
    exec ffmpeg $audio_args \
        -f x11grab \
        -framerate 60 \
        $video_size \
        -i ${DISPLAY}+${start_pos} \
        -vaapi_device /dev/dri/renderD128 \
        -vcodec h264_vaapi \
        -vf format='nv12|vaapi,hwupload' \
        "$output"
}

main $@