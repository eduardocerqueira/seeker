#date: 2022-09-26T17:17:15Z
#url: https://api.github.com/gists/5910b9d1e84e98d79625d7522a240a2d
#owner: https://api.github.com/users/oclero

for i in *.wav; do ffmpeg -i "$i" -codec:a libmp3lame -ar 44100 -b:a 320k "${i%.*}.mp3"; done