#date: 2022-12-12T17:04:35Z
#url: https://api.github.com/gists/582a2e6395ccf9507e787bdfa5a1f33e
#owner: https://api.github.com/users/wernersbacher

#!/bin/bash
# Extract subtitles from each MKV file in the given directory

# If no directory is given, work in local dir
if [ "$1" = "" ]; then
  DIR="."
else
  DIR="$1"
fi

# Get all the MKV files in this dir and its subdirs
find "$DIR" -type f -name '*.mkv' | while read filename
do
  # Find out which tracks contain the subtitles
  mkvmerge -J $filename | jq -r '
    .tracks |
      map((.id | tostring) + " " + .properties.language + .properties.track_name + " " + .codec) |
      join("\n")' | grep 'SubRip' | while read subline
  do
    # Grep the number of the subtitle track
    tracknumber=`echo $subline | cut -d' ' -f1`
    
    language=`echo $subline | cut -d' ' -f2`

    # Get base name for subtitle
    subtitlename="${filename%.*}_$language"

    # add language check if you like e.g.:
    # if [ "$language" == "ger" ] || [ "$language" == "chiSimplified" ]; then
    `mkvextract tracks "$filename" $tracknumber:"$subtitlename.srt" > /dev/null 2>&1`
    `chmod g+rw "$subtitlename.srt"`
  done
done