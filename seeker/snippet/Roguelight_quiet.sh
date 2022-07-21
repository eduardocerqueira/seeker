#date: 2022-07-21T16:51:42Z
#url: https://api.github.com/gists/c847136564f7b8caf4b06439f02d8aa2
#owner: https://api.github.com/users/qguv

#!/bin/sh
set -e

exe="$1"
zip="$2"
re="/tmp/re/ResourcesExtract.exe"

function usage() {
    printf 'Usage: %s EXE ZIP\n' "$0"
    printf '%s\n' "Removes in-game background music from Roguelight.exe, producing a .zip containing a new .exe and its dependencies."
    exit 2
}

# $1...: binaries on $PATH
function check_deps() {
    ok=true
    while [ $# -ne 0 ]; do
        if ! hash "$1" 2>/dev/null; then
            printf 'missing dependency: %s\n' "$1" 2>&1
            ok=false
        fi
        shift
    done
    if [ "$ok" = false ]; then
        exit 1
    fi
}

# $1: output directory
function download_re_to_dir() {
    mkdir "$1"
    (
        cd "$1"
        curl -Lo re.zip https://www.nirsoft.net/utils/resourcesextract.zip
        unzip re.zip
        rm re.zip
    )
}

# $1: unix path
function winpath() {
    printf '%s' 'Z:'
    realpath "$1" | tr '/' '\\'
}

# $1: path to exe (unix format)
# $2: path to output file
function exe2cab() {
    resources_dir="$(mktemp -d)"
    wine "$re" \
        /Source "$(winpath "$1")" \
        /DestFolder "$(winpath "$resources_dir")" \
        /ExtractBinary 1 \
        /ExtractAVI 0 \
        /ExtractIcons 0 \
        /OpenDestFolder 0
    mv "$resources_dir"/Roguelight_CABINET_10.bin "$2"
    rm -rf "$resources_dir"
}

# $1: path to cab file
# $2: output zip
function repack_cab() {
    infile="$(realpath "$1")"

    zipname="$(basename "$2")"
    zipname="${zipname%*.zip}"

    workdir="$(mktemp -d)"
    (
        cd "$workdir"
        mkdir "$zipname"
        gcab -C "$zipname" -x "$infile"
        rm "$zipname"/music_main.mp3
        zip -r out.zip "$zipname"
    )
    mv "$workdir"/out.zip "$2"
    rm -rf "$workdir"
}

if ! [ $# = 2 ]; then
    usage
fi

if ! [ -f "$exe" ]; then
    printf '%s\n' "can't find Roguelight.exe" 2>&1
    usage
fi

if [ -f "$re" ]; then
    check_deps wine gcab zip
else
    check_deps wine gcab zip curl unzip
    download_re_to_dir "$(dirname "$re")"
fi

cab="$(mktemp)"
exe2cab "$exe" "$cab"
repack_cab "$cab" "$zip"
rm "$cab"
