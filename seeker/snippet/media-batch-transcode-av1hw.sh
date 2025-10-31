#date: 2025-10-31T16:59:07Z
#url: https://api.github.com/gists/05d5a9a85236c7a6e8e1f13fa90a3f5a
#owner: https://api.github.com/users/surveilled

#!/usr/bin/env bash
DEBUG=0 # set to 1 for dry running with stdout verbosity
        # will still populate INTERFOLDER

        ### README
    ## DEPENDENCIES
# ffmpeg
# libheif-examples
# exiftool
# INSTALL VIA apt install ffmpeg libheif-examples libimage-exiftool-perl -y

#################################################################
### START OF CONFIG #############################################

        ### CONFIG OF FOLDERS
# tmp folder
INTERFOLDER="~/.mediatmp"
# drop folder, if not specified FULL PATH in SECOND parameter
DROPFOLDER="~/Videos/transcode/in"
# out folder, can be specified FULL PATH by FIRST parameter
OUTFOLDER="~/Videos/transcode/out"

        ### CONFIG OF FFMPEG (OBLIGATORY)

    ## VIDEOS
# minimum byte-size (not kB/MB/GB!) of file to process (smaller will be skipped)
SMALLVID=15728640
# api for scaling. "scale" if on CPU, "scale_vaapi" if on GPU via VAAPI, check documentation for other apis scaling hook
VIDSCALEAPI="scale_vaapi"
# video target bitrate. you can use e.g. "512k" or "4M"
VIDEOBITS="4M"
# target size in pixels on LONGER side
VIDEOSIZE=1920
# main function to tune
VIDFFMPEG() {
    ffmpeg -hide_banner \
        -hwaccel vaapi -hwaccel_output_format vaapi \
        -i "$1" $VF \
        -c:v av1_vaapi \
        -extbrc 1  -look_ahead_depth 40 \
        -b:v "$VIDEOBITS" \
        -c:a copy \
        "$OUTFOLDER/$2.mp4"
    }

    ## IMAGES
# api for scaling. "scale" if on CPU, "scale_vaapi" if on GPU via VAAPI, check documentation for other apis scaling hook
PICSCALEAPI="scale"
# quality factor - lower is better (i.e. bigger byte-size)
IMAGECRF=20
# target size in pixels on SHORTER side
IMAGESIZE=2160
# main function to tune
PICFFMPEG() {
    ffmpeg -hide_banner -i "$1" $VF \
        -c:v av1_vaapi -crf "$IMAGECRF" \
        -cpu-used 0 "$OUTFOLDER/${1%.*}.avif"
    }


##### OPTIONAL config below #####################################

        ### FILETYPE ARRAYS
    ## generally no need to change that.
# "good" are usually well-compressed to begin with and will be COPIED
goodVariant=( "avif" "AVIF" "webp" "WEBP" )
# ffmpeg chokes on HEIF, thus conversion out of that format
heifVariant=( "heic" "heif" "HEIC" "HEIF" )
# jpegs, due to poor renrot compatibility
jpegVariant=( "jpeg" "jpg" "JPEG" "JPG" )
# the rest, compatible with ffmpeg
otherVariant=( "jxl" "JXL" "tif" "tiff" "TIF" "TIFF" "png" "PNG" "bmp" "BMP" )
# videos
vidVariant=( "mov" "MOV" "mp4" "MP4" "mkv" "MKV" )

        ### EXIF TYPES
    ## choose which tags to move
# for images
picExifMigrate=( "datetimeoriginal" "focallength" "gpslatitude" "gpslongitude" )
# for videos
vidExifMigrate=( "createdate" "trackcreatedate" "mediacreatedate" )

### END OF CONFIG ###############################################
#################################################################



# PREP
if [[ ! -d "$INTERFOLDER" ]]; then mkdir -p "$INTERFOLDER" ; fi
if [[ ! -z $1 ]]; then OUTFOLDER="$1" ; fi
if [[ ! -z $2 ]]; then DROPFOLDER="$2" ; fi

if [[ `ls "$DROPFOLDER" | wc -c` -eq 0 ]]; then
    echo "Drop folder empty or absent. (Be sure to use FULL PATHs, or tilde expansion.)"
    exit
fi


shopt -s extglob nullglob
cd $DROPFOLDER



#################################################################
### PART 1: PICTURES
## SECTION 0: GOOD ENOUGH
# No need to do any modifications, but maybe if EXIF is missing...

for img in "${goodVariant[@]}" ; do
    test $DEBUG = 1 && echo "# copying $img files"
    cp *.$img "$OUTFOLDER/"
done


## SECTION 1: EXIFTOOL
# try to use common Date/Time Original tag

function NAMEFROMEXIF {
    TIMESTAMP=$(exiftool -datetimeoriginal "$1")
    test $DEBUG = 1 && echo "$TIMESTAMP"
    IMGNEWNAME="${TIMESTAMP:34:4}${TIMESTAMP:39:2}${TIMESTAMP:42:2}_${TIMESTAMP:45:2}${TIMESTAMP:48:2}${TIMESTAMP:51:2}"
    if [[ ${IMGNEWNAME:0:1} -eq "_" ]]; then
        echo "# name couldn't be established from EXIF!"
        IMGNEWNAME="${1%.*}"
    else
        echo "# new name $IMGNEWNAME"
    fi
    echo "$1|$IMGNEWNAME.avif" >> "$DROPFOLDER/.picExifMigrate.tmp"
    }


## SECTION 2: HEICs
# they choke ffmpeg, need transposition to intermediate format

for heimg in "${heifVariant[@]}"; do
    for h in *.$heimg ; do
        test $DEBUG = 1 && echo "# converting HEIC $h"
        NAMEFROMEXIF "$h"
        heif-convert -q 100 "$h" "$INTERFOLDER/$IMGNEWNAME.y4m"
    done
done

## SECTION 3: THE REST, RENAME IF POSSIBLE
# if EXIF can set the date, then it should

for rest in "${jpegVariant[@]}" "${otherVariant[@]}" ; do
    for i in *.$rest; do
        if [[ ! $i =~ [0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]* ]]; then
            test $DEBUG = 1 && echo "# attempting rename of $i"
            NAMEFROMEXIF "$i"
        else
            IMGNEWNAME="${i%.*}"
            echo "$i|$IMGNEWNAME.avif" >> "$DROPFOLDER/.picExifMigrate.tmp"
        fi
        test $DEBUG = 1 && echo "# moving $i under name $IMGNEWNAME"
        cp "$i" "$INTERFOLDER/$IMGNEWNAME.${i#*.}"
    done
done

## SECTION 4: TRANSCODE
# aspect ratio sometimes affected by EXIF

function IMAGETRANSCODE {
    if [[ $(ffprobe "$1" 2>&1 | grep Video:) =~ ([0-9]{3,4})x([0-9]{3,4}) ]]; then
        width="${BASH_REMATCH[1]}"
        height="${BASH_REMATCH[2]}"

        if [[ $width -le $IMAGESIZE ]] && [[ $height -le $IMAGESIZE ]]; then
            test $DEBUG = 1 && echo "# resizing $1 not necessary"
            VF=""
        elif [[ $height -gt $width ]]; then   # vertical aspect
            VF="-vf $PICSCALEAPI=$IMAGESIZE:-2"
        else
            echo "# checking orientation of $1"
            exiftool -orientation "$1" | grep Rotate
            if [[ $? -eq 0 ]]; then # it's rotated
                echo "# $1 found rotation by exif"
                VF="-vf $PICSCALEAPI=$IMAGESIZE:-2"
            else
                VF="-vf $PICSCALEAPI=-2:$IMAGESIZE"
            fi
        fi

        if [[ $DEBUG = 1 ]]; then echo "FFMPEG $1 with res '$VF'"
        else
            PICFFMPEG "$1" || echo "$1 crashed ffmpeg!" >> "$DROPFOLDER/.errors.log"
        fi

    else # ffmpeg could't read file
        echo "# ffprobe couldn't read the file $1." >> "$DROPFOLDER/.errors.log"
    fi
    }

cd "$INTERFOLDER"
for ready in * ; do
    IMAGETRANSCODE "$ready"
done
test $DEBUG = 1 || rm "$INTERFOLDER/"*




#################################################################
### PART 2: VIDEOS
## SECTION 1: DATES

function DATEFROMPROBE {
    test $DEBUG = 1 && echo "# renaming $1"
    VIDSTAMP=$(ffprobe "$1" 2>&1 | grep -E ':\s[0-9]{4}-[0-9]{2}-[0-9]{2}T' | head -n 1)
    if [[ -z $VIDSTAMP ]]; then
        test $DEBUG = 1 && echo "# unable to find creationdate"
        VIDNEWNAME="${1%.*}"
        return 1
    else
        if [[ "$VIDSTAMP" =~ ([0-9]{4}-[0-9]{2}-[0-9]{2}T.*) ]]; then
            VIDNEWNAME="${BASH_REMATCH:0:4}${BASH_REMATCH:5:2}${BASH_REMATCH:8:2}_${BASH_REMATCH:11:2}${BASH_REMATCH:14:2}${BASH_REMATCH:17:2}"
            echo "# Renaming $1 to $VIDNEWNAME.mp4."
        fi
    fi
    }


## SECTION 2: SCALING
# and hook up ffmpeg

function VIDEOTRANSCODE {
    if [[ $(ffprobe "$v" 2>&1 | grep Video:) =~ ([0-9]{3,4})x([0-9]{3,4}) ]]; then
        width="${BASH_REMATCH[1]}"
        height="${BASH_REMATCH[2]}"

        if [[ $width -le $VIDEOSIZE ]] && [[ $height -le $VIDEOSIZE ]]; then
            test $DEBUG = 1 && echo "# resizing $v not necessary"
            VF=""
        else
            if [[ $height -gt $width ]]; then   # vertical aspect
                VF="-vf $VIDSCALEAPI=-2:$VIDEOSIZE"
            else                                # horizontal aspect
                VF="-vf $VIDSCALEAPI=$VIDEOSIZE:-2"
            fi
        fi

        if [[ $DEBUG = 1 ]]; then echo "FFMPEG $v with res '$VF' and name $VIDNEWNAME"
        else
            VIDFFMPEG "$v" "$VIDNEWNAME" || echo "$v crashed ffmpeg!" >> "$DROPFOLDER/.errors.log"
        fi

     else # ffmpeg could't read file
        echo "# ffprobe couldn't read the file $1." >> "$DROPFOLDER/.errors.log"
    fi
    }


## SECTION 3: ITERATE
# set second parameter for new filename if available

cd "$DROPFOLDER"
for vform in ${vidVariant[@]}; do
    for v in *.$vform; do
        if [[ ! "$v" =~ [0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]* ]]; then
            DATEFROMPROBE "$v"
        else
            VIDNEWNAME="${v%.*}"
        fi

        # skip if small file
        if [[ $(stat "$v" | grep Size: | cut -c9-19) -le $SMALLVID ]]; then
            echo "# just moving $v (small size) under name $VIDNEWNAME.${v##*.}"
            test $DEBUG = 1 || cp "$v" "$OUTFOLDER/$VIDNEWNAME.${v##*.}"
        else
            VIDEOTRANSCODE
            echo "$v|$VIDNEWNAME.mp4" >> "$DROPFOLDER/.vidExifMigrate.tmp"
        fi
    done
done




#################################################################
### PART 3: EXIFS & CLEANUP
# don't do when debugging
test $DEBUG = 1 && exit
cd "$OUTFOLDER"

## SECTION 1: pictures
while IFS="|" read -r FROM INTO; do
    for tag in ${picExifMigrate[@]}; do
        exiftool -tagsfromfile "$DROPFOLDER/$FROM" "-$tag>$tag" "$OUTFOLDER/$INTO"
    done
done < "$DROPFOLDER/.picExifMigrate.tmp"


## SECTION 2: videos generally
while IFS="|" read -r FROM INTO; do
    for tag in ${vidExifMigrate[@]}; do
        exiftool -tagsfromfile "$DROPFOLDER/$FROM" "-$tag>$tag" "$OUTFOLDER/$INTO"
    done
done < "$DROPFOLDER/.vidExifMigrate.tmp"

## SECTION 3: potentially missing items
for item in *.avif; do
    if [[ -z $(exiftool -datetimeoriginal "$item") ]] && [[ $item =~ [0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]* ]]; then
        echo "# engraining EXIF based on filename into $item"
        exiftool -datetimeoriginal="${BASH_REMATCH:0:4}:${BASH_REMATCH:4:2}:${BASH_REMATCH:6:2} ${BASH_REMATCH:9:2}:${BASH_REMATCH:11:2}:${BASH_REMATCH:13:2}" "$item"
    fi
done

for item in *.mp4; do
    if [[ -z $(exiftool -mediacreatedate "$item") ]] && [[ $item =~ [0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]* ]]; then
        echo "# engraining EXIF based on filename into $item"
        exiftool -datetimeoriginal="${BASH_REMATCH:0:4}:${BASH_REMATCH:4:2}:${BASH_REMATCH:6:2} ${BASH_REMATCH:9:2}:${BASH_REMATCH:11:2}:${BASH_REMATCH:13:2}" "$item"
    fi
done

## SECTION Z: cleanup
rm *_original
rm "$DROPFOLDER/.picExifMigrate.tmp" 2>/dev/null
rm "$DROPFOLDER/.vidExifMigrate.tmp" 2>/dev/null
if [[ ! -z $(wc -l "$DROPFOLDER/.errors.log") ]]; then
    date >> "$DROPFOLDER/.errors.log"
    echo "------------------------------" >> "$DROPFOLDER/.errors.log"
fi
