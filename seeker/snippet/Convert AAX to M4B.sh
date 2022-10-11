#date: 2022-10-11T17:09:49Z
#url: https://api.github.com/gists/099309b4a0cbe4a06ec62d6f22b1440c
#owner: https://api.github.com/users/Pamblam

# Converting Audible AAX files to M4B Audiobook files, split into chapters and decrypted
# Get activation bits here for now: https://audible-converter.ml

SHORTNAME='BeautifulYou'
AAXFILE='/Users/rob/Downloads/BeautifulYou_ep6.aax'
ACTIVATION_BYTES='2758110a'

# These files don'e exist yet, we just need to tell the script where to create them.
DATAFILE='/Users/rob/Desktop/audible-conv/data.json'
METADATAFILE='/Users/rob/Desktop/audible-conv/data.tmp'

# generate the picture
ffmpeg -y -i "$AAXFILE" "$SHORTNAME.png"

# Create a JSON file with chapter breaks and read the file data into arrays
ffprobe -loglevel error -i "$AAXFILE" -print_format json -show_chapters -loglevel error -sexagesimal > "$DATAFILE"
IDS=($(jq -r '.chapters[].id' "$DATAFILE"))
START_TIMES=($(jq -r '.chapters[].start_time' "$DATAFILE"))
END_TIMES=($(jq -r '.chapters[].end_time' "$DATAFILE"))
TITLES=()
while IFS= read -r line; do
   TITLES+=("$line")
done < <(jq -c -r '.chapters[].tags.title' "$DATAFILE")

# create a ffmpeg metadata file to extract additional metadata lost in splitting files
ffmpeg -loglevel error -i "$AAXFILE" -f ffmetadata "$METADATAFILE"
artist_sort=$(sed 's/.*=\(.*\)/\1/' <<<$(cat "$METADATAFILE" |grep -m 1 ^sort_artist))
album_sort=$(sed 's/.*=\(.*\)/\1/' <<<$(cat "$METADATAFILE" |grep -m 1 ^sort_album))

mkdir -p "$SHORTNAME"

for i in ${!IDS[@]}; do
  let trackno=$i+1
  outname="$SHORTNAME/${TITLES[$i]}.m4b"
  outname=$(sed 's/:/_/g' <<< $outname)
  outname=$(sed 's/ /_/g' <<< $outname)
  ffmpeg -loglevel error -y -activation_bytes $ACTIVATION_BYTES \
  			-i "$AAXFILE" -vn -c copy \
            -ss ${START_TIMES[$i]} -to ${END_TIMES[$i]} \
            -metadata title="${TITLES[$i]}" \
            -metadata track=$trackno \
            -map_metadata 0 -id3v2_version 3 \
            "$outname"
  AtomicParsley "$outname" \
            --artwork "$SHORTNAME.png" --overWrite \
            --sortOrder artist "$artist_sort" \
            --sortOrder album "$album_sort" \
            > /dev/null
done