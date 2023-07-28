#date: 2023-07-28T17:06:08Z
#url: https://api.github.com/gists/53056a9f52fa7c1d69ef998e67a51bb3
#owner: https://api.github.com/users/SuaYoo

# Software encoding, more options, slower
ffmpeg \
  -i input.mov \
  -hwaccel auto \
  -c:v libx264 \
  -c:a aac_at \
  -b:a 320k \
  -crf 17 \
  -preset slow \
  -movflags +faststart \
  -vf format=yuv420p \
  -tune film \
  output.mp4

# -crf 17                               Constant Rate Factor (0 lossless to 51 worst, 17-18 visually lossless)
# -hwaccel auto                         Hardware accelerated decoding
# -tune film                            Lower deblocking, preserve details
# -c:a aac_at, -b:a 320k                Apple audiotoolbox + Constant Bit Rate (HD stereo)
# -vf format=yuv420p                    Quicktime/broader player support



# 4K -> 2K
ffmpeg \
  -i input.mov \
  -hwaccel auto \
  -c:v libx264 \
  -c:a aac_at \
  -b:a 320k \
  -crf 17 \
  -minrate 20M \
  -preset slow \
  -movflags +faststart \
  -vf format=yuv420p,scale=2560:-1 \
  -tune film \
  output.mp4

# -minrate 20M                          Minimum of 20Mbps bit rate (suggested)
# -vf scale=2560:-1                     Width 2560, keep aspect ratio




# Hardware encoding, less options, faster
ffmpeg \
  -i input.mov \
  -c:v h264_videotoolbox \
  -q:v 75 \
  -c:a aac_at \
  -b:a 320k \
  -movflags +faststart \
  -prio_speed false \
  -vf format=yuv420p \
  output.mp4
  
# -c:v h264_videotoolbox       Apple Silicon hardware acceleration
# -q:v 75                      Quality (0 worst - 100 is best, 75 seems acceptable)