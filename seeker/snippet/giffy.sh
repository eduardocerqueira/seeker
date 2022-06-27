#date: 2022-06-27T17:01:06Z
#url: https://api.github.com/gists/3de08bfe6b3d8c26c08890c2325089ae
#owner: https://api.github.com/users/discatte

# Gif optimization based on
# https://cassidy.codes/blog/2017/04/25/ffmpeg-frames-to-gif-optimization/
# http://blog.pkh.me/p/21-high-quality-gif-with-ffmpeg.html
# https://github.com/intgr/keep-on-giffing

ffmpeg -i  "my_frames_%03d.png" \
       -vf "palettegen=stats_mode=diff:max_colors=256" \
	     "my_palette.png"
       
ffmpeg -i "my_frames_%03d.png" \
       -i "my_palette.png" \
       -lavfi "paletteuse=dither=bayer:diff_mode=rectangle" \
	     "my_animation.gif"