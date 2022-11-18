#date: 2022-11-18T17:07:06Z
#url: https://api.github.com/gists/0f6661be04e43062f4f6d6034a6582fb
#owner: https://api.github.com/users/recursivecodes

ffmpeg \
    -rtsp_transport \
    tcp \
    -i rtsp://demo:AmazonIVS@192.168.86.34:554 \
    -f image2 \
    -c:v libx264 \
    -b:v 6000K \
    -maxrate 6000K \
    -pix_fmt yuv420p \
    -r 30 \
    -s 1920x1080 \
    -profile:v main \
    -preset veryfast \
    -g 120 \
    -x264opts "nal-hrd=cbr:no-scenecut" \
    -acodec aac \
    -ab 160k \
    -ar 44100 \
    -f flv \
    $DEMO_STREAM_INGEST_ENDPOINT/$DEMO_STREAM_KEY