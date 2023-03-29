#date: 2023-03-29T17:40:04Z
#url: https://api.github.com/gists/2884b744afc17af4884d5baecb0d1c43
#owner: https://api.github.com/users/bigforcegun

gource --hide dirnames,filenames --seconds-per-day 0.1 --auto-skip-seconds 1 -1280x720 -o - | ffmpeg -y -r 30 -f image2pipe -vcodec ppm -i - -vcodec libx264 -preset ultrafast -pix_fmt yuv420p -crf 1 -threads 0 -bf 0 gource.mp4

# дожимаем если надо (можно сразу так в ffmpeg из gource)

ffmpeg -i gource.mp4 -vcodec libx265 -crf 28 output.mp4\