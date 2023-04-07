#date: 2023-04-07T16:38:48Z
#url: https://api.github.com/gists/f735077aff0850cbe0b531c52c1e3fcf
#owner: https://api.github.com/users/amiantos

# Install gource and ffmpeg
brew install gource ffmpeg

# Export activity list
gource --output-custom-log repo-activity.txt

# Then clean up names in repo-activity.txt
code repo-activity.txt

# Then pick a grouce command based on your needs

# 30 fps at normal speed for small repos with h264 encoding for wide compatability
gource repo-activity.txt \
    -s 1 \
    -1280x720 \
    --auto-skip-seconds .1 \
    --multi-sampling \
    --stop-at-end \
    --highlight-users \
    --hide mouse \
    --hide filenames \
    --font-size 25 \
    --output-ppm-stream - \
    --output-framerate 30 \
    | ffmpeg -y -r 30 -f image2pipe -vcodec ppm -i - -vcodec libx264 -preset medium -pix_fmt yuv420p -crf 1 -threads 0 -bf 0 gource.mp4
    
# 60 fps at fast speed with h265 encoding
# Good for repos that have been around for years
gource repo-activity.txt \
  -s .03 \
  -1280x720 \
  --auto-skip-seconds .1 \
  --multi-sampling \
  --stop-at-end \
  --highlight-users \
  --hide mouse \
  --hide filenames \
  --font-size 25 \
  --output-ppm-stream - \
  --output-framerate 60 \
  | ffmpeg -y -r 60 -f image2pipe -vcodec ppm -i - -vcodec libx265 -preset medium -pix_fmt yuv420p -crf 1 -threads 0 -bf 0 gource.mp4
