#date: 2024-02-13T17:04:18Z
#url: https://api.github.com/gists/a8c62387cd9346e65637cb0cb34951dd
#owner: https://api.github.com/users/AkBKukU

#!/usr/bin/env bash

# Usage
# mderive source (TXT of dir paths or dir) destination

# Conversion Format
ext="mp3"
args="-c:a mp3 -b:a 320k"
over="-y" # Overwrite (-n/-y)

origin="$1"
dest="$2"

origin_list=()

function convert ()
{
    # Final conversion step
    ffmpeg $over -i "$1" $args "$2"
}

function origin_dir ()
{
    # Convert single directory to an array element
    echo "Dir to list: $1"
    origin_list+=("$1")
}


function origin_file ()
{
    # Read list of folders into array
    readarray -t origin_list < "$1"
}

function dir_walk ()
{
    # Recursive direcyory scanning for media files and artwork
    base="$1" # Current dir path
    echo "Working in [$base]"
    folders=( "$base"/* ) # Contents of current path
    for dir in "${folders[@]}"
    do
        # Iterate over contents of current path
        if [[ -d "$dir" ]]
        then
            # Recursively call self if a subdir has been found
            if [[ "$dest" != "" ]]
            then
                # Build destination structure if path set
                mkdir -p "$dest/$dir"
            fi
            dir_walk "$dir"
        else
            # Handle files
            if [[ "$(file "$dir" | grep audio)" != "" ]]
            then
                # Convert audio files
                echo "Converting [$dir]"

                if [[ "$dest" != "" ]]
                then
                    convert "$dir" "$dest/${dir%.*}.$ext"
                fi
            fi

            if [[ "$(file "$dir" | grep image)" != "" ]]
            then
                # Copy album art
                echo "Copying Art [$dir]"

                if [[ "$dest" != "" ]]
                then
                    cp "$dir" "$dest/$dir"
                fi
            fi
        fi
    done
}


# Main

if [[ -d "$origin" ]]
then
    echo "Origin is directory"
    origin_dir "$origin"
elif [[ -f "$origin" ]]
then
    echo "Origin is file"
    origin_file "$origin"
else
    echo "Origin is invald"
    echo -e "Usage: \n\tmderive [source] [destination]\n\t(Source may be folder or TXT file of folders)"
    exit 1
fi

# Iterate over all folders for origin with recursion
for odir in "${origin_list[@]}"
do
    dir_walk "$odir"
done