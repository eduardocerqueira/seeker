#date: 2025-02-19T16:43:14Z
#url: https://api.github.com/gists/83702eb23425548eb79b2646eb5bc785
#owner: https://api.github.com/users/mtmn

copy-playlist-tracks() {
    local basepath="$HOME/misc/xld.out"
    local playlistname="${1?}"
    [ -d "$basepath"/"$playlistname" ] || mkdir "$basepath"/"$playlistname" || exit
    sed "s/#.*//g" < "$playlistname" | sed "/^$/d" | while read line; do cp "${line}" "$basepath"/"$playlistname"/; done
    cd "$basepath"/"$playlistname" || exit
    parallel xld {} -f aif --samplerate 44100 --bit 16 ::: *.flac
    rm ./*.flac
}