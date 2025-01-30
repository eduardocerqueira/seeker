#date: 2025-01-30T16:54:19Z
#url: https://api.github.com/gists/c2be6be341238e76849f9ca49f35c1f8
#owner: https://api.github.com/users/Windovvsill

for f in "$@"
do
	filename="${f%.*}"
    date=$(date +"%Y-%m-%dT%H-%M-%S")
	framerate=30
    out="/Users/steve.vezeau/Desktop/wowCoolGif-$date.gif"
    echo "$f"
    filesize=$(stat -f%z "$f")
	if [ "$filesize" -gt 1000000 ]; then
      framerate=20
	fi
    if [ "$filesize" -gt 10000000 ]; then
      framerate=10
	fi
    echo "$filesize"
    echo $framerate
	echo "did"
	/opt/homebrew/bin/ffmpeg -y -i "$f" -vf fps=30,scale=320:-1:flags=lanczos,palettegen palette.png
	/opt/homebrew/bin/ffmpeg -y -i "$f" -i palette.png -filter_complex "fps=$framerate,scale=0:-1:flags=lanczos[x];[x][1:v]paletteuse" "$out"
	rm palette.png
	osascript -e 'display notification "'"$out"' created by Automator script" with title "GIF Created"'
    osascript -e "set the clipboard to (POSIX file \"$out\")"
    echo 'based on work by https://www.ehfeng.com/gif-screen-recordings-on-macos/'
done