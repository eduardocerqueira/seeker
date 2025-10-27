#date: 2025-10-27T17:04:29Z
#url: https://api.github.com/gists/6e576ede58090509a96c52fb5a174e01
#owner: https://api.github.com/users/greasycat

ffmpeg -fflags nobuffer -flags low_delay   -input_format mjpeg -video_size 1280x720 -framerate 30   -i /dev/video0   -map 0:v -f v4l2 -vcodec rawvideo -pix_fmt yuv420p /dev/video10   -map 0:v -vf scale=1280x720:flags=fast_bilinear -f v4l2 -vcodec rawvideo -pix_fmt yuv420p /dev/video11
