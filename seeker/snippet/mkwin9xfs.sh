#date: 2022-03-04T17:13:47Z
#url: https://api.github.com/gists/e56286add9bc36db0f3490aa8b236613
#owner: https://api.github.com/users/christopherwoodall

#!/usr/bin/env bash
hash parted 2>/dev/null || { echo >&2 "Program requires parted but it's not installed. Exit."; exit 1; }
hash numfmt 2>/dev/null || { echo >&2 "Program requires numfmt but it's not installed. Exit."; exit 1; }
hash truncate 2>/dev/null || { echo >&2 "Program requires truncate but it's not installed. Exit."; exit 1; }
hash mcopy 2>/dev/null || { echo >&2 "Program requires mtools but it's not installed. Exit."; exit 1; }
hash sed 2>/dev/null || { echo >&2 "Program requires sed but it's not installed. Exit."; exit 1; }
#hash udisksctl 2>/dev/null || { echo >&2 "Program requires udisksctl but it's not installed. Exit."; exit 1; }
#hash fdisk 2>/dev/null || { echo >&2 "Program requires fdisk but it's not installed. Exit."; exit 1; }
#hash mkdosfs 2>/dev/null || { echo >&2 "Program requires mkdosfs but it's not installed. Exit."; exit 1; }
#hash mattrib 2>/dev/null || { echo >&2 "Program requires mtools but it's not installed. Exit."; exit 1; }
#hash dd 2>/dev/null || { echo >&2 "Program requires dd but it's not installed. Exit."; exit 1; }


ME=$(basename "$0")

function usage(){
    cat >&2 <<ENDOFHELP
Usage: $ME [-h|--help] [-f|--fat32] [-c|--clone IMG] SIZE NAME

$ME creates a single partition FAT (CHS)
filesystem, already formated and outputs the
the dosbox imgmount and the linux mount commands.
FAT 16 is the default (dosbox readable).

The size will be restricted between ≃35MB and ≃1GB
for FAT16 and between ≃35MB and ≃7.84GB for FAT32.
Do not be surprised if a large drive gives you less
than expected space in the OS, because the cluster
size limits that enable that extra space, make the
OS waste space on larger drives because all files
must occupy at least 1 cluster, so if you have many
small files (like in the OS) it's better to use a
small drive just for it.

You can use the strings 'MIN' and 'MAX' as size.
You can use a human readable suffix for the SIZE
in SI or IEC-I standards. SI is the modern hd size
standard with powers of 10 and IEC-I is the older
hd size standard with powers of two. Bytes will be
assumed if no suffix is given.

SI    goes KB,  MB,  GB,  TB...
IEC-I goes KiB, MiB, GiB, TiB...

    -h show this help
    -f force FAT32 type
    -c IMG clone IMG file primary partition to
       the new image, preserving DOS attributes,
       errors if there is not enough space, does
       not work with bootable IMG
ENDOFHELP
}

#silent mode, h or f or c or - followed by a 'argument' to handle long options 
#(notice that all of these require - already, including '-')
while getopts ":hfc-:" opt; do
  case $opt in
    -)
        case "${OPTARG}" in
            fat32) FAT=32; ;;
            clone) CLONE="${!OPTIND}"; OPTIND=$(( OPTIND + 1 )); ;;
            help) usage; exit 0; ;;
            *) usage; exit 1; ;;
        esac;;
    f) FAT=32;  ;;
    c) CLONE="${!OPTIND}"; OPTIND=$(( OPTIND + 1 )); ;;
    h) usage; exit 0; ;;
    *) usage; exit 1; ;;
  esac
done
shift $((OPTIND-1))
(( "$#" != 2 ))   && { usage; exit 1; }
[[ ! -d "$PWD" ]] && { echo  >&2 "$ME: could not retrieve current directory. Exit"; exit 1; }

#variable checks, leave CLONE unset if unset
export NAME=$(realpath -- "$2") #deleted on error, needs to created after options parsing
export TMPFILE=$(mktemp -u)     #deleted on exit
BASE=$(basename "$NAME")
FAT=${FAT-16}                   #if unset set to 16
SIZE="$1"                       #currently a string which may have suffixes (MB etc), needs validation
MIN_FAT=35000000
#while FAT16 is supposed to support 2gb, mformat tries to place 64kb clusters above this 
#which aren't win95/98/DOS compatible, so this value came from trial and error 
#(-c doesn't upper bound, only lower bound, so you can't force it above this)
MAX_FAT16=1074000000
MAX_FAT32=8414461440 #derived from http://web.allensmith.net/Storage/HDDlimit/Int13h.htm but with 1023 heads (for a DOS hang bug)
[[ -d "$NAME" ]] && { echo  >&2 "$ME: output file is a directory. Exit"; exit 1; }
[[ -f "$NAME" ]] && { echo  >&2 "$ME: output file already exists. Exit"; exit 1; }

if [[ "$SIZE" == MIN ]]; then
    SIZE=$MIN_FAT
elif [[ "$SIZE" == MAX ]]; then
    if (( "$FAT" == 32 )); then
        SIZE=$MAX_FAT32
    else
        SIZE=$MAX_FAT16
    fi
else
    SIZE=$(LC_ALL=C numfmt --from=auto --suffix="B" "$SIZE" 2>/dev/null) \
        || { echo  >&2 "$ME: could not convert size input to bytes. Exit."; exit 1; }
    SIZE=${SIZE%B}
fi

if (( FAT == 16 )); then
    (( "$SIZE" >= "$MIN_FAT" )) && (( "$SIZE" <= "$MAX_FAT16" )) \
        || { echo  >&2 "$ME: requested size for FAT 16 image not in [35MB-2GB] range. Exit"; exit 1; }
else
    (( "$SIZE" >= "$MIN_FAT" )) && (( "$SIZE" <= "$MAX_FAT32" )) \
        || { echo  >&2 "$ME: requested size for FAT 32 image not in [35MB-8GB] range. Exit"; exit 1; }
fi

#cleanup code attached here (previous exits should NOT delete name)
function finish(){
    EXIT=$?
    rm -f "$TMPFILE" 2>/dev/null
    (( "$EXIT" != 0 )) && { rm -f "$NAME" 2>/dev/null; }
    exit $EXIT
}
trap finish INT TERM EXIT

OFFSET=$((2048*512)) #1024KiB or 2048 sectors, offsets partition
# align to next MB (not sure if it should be aligned to linux blocks or the windows page file boundaries)
SIZE=$(( (SIZE + OFFSET - 1)/OFFSET * OFFSET ))
# Number of Sectors in the filesystem that parted will give
SECTORS=$(( (SIZE-OFFSET) / 512 ))

#create fs name based on name limited to 11 chars
DNAME="${NAME##*/}"
DNAME="${DNAME%.*}"
DNAME="${DNAME:0:11}"
DNAME="${DNAME^^}"

#sparse file
truncate -s "$SIZE" "$NAME"

##create the bootable CHS partition with optimal alignment
parted "$NAME" -s -a optimal mklabel msdos mkpart primary fat$FAT "${OFFSET}B" "100%" set 1 lba off set 1 boot off 2>/dev/null  \
        || { echo  >&2 "$ME: could not partition filesystem. Exit"; exit 1; }

if (( FAT == 32 )); then #doesn't have a else on purpose
    FLAG="-F"
fi

#tries to create a optimal csh by just specifying sectors and limiting max size
MTOOLSRC=<( echo "drive c: file=\"$NAME\" partition=1" ) \
        mformat -i "${NAME}"@@"${OFFSET}" -T $SECTORS -v "$DNAME" $FLAG \
        || { echo  >&2 "$ME: error formating the filesystem. Exit"; exit 1; }

#ms-sys/bin/ms-sys --mbr95b -f "$NAME" \
#        || { echo  >&2 "$ME: could not add Windows Boot volume record to filesystem. Exit"; exit 1; }

#LOOP=$(udisksctl loop-setup -f "$NAME") \
#        || { echo  >&2 "$ME: could not mount filesystem. Exit"; exit 1; }
#LOOP="${LOOP##* }"
#LOOP="${LOOP%.}"
#sudo ms-sys/bin/ms-sys --fat$FAT -f "${LOOP}p1" \
#        || { echo  >&2 "$ME: could not add Windows Boot volume record to filesystem. Exit"; exit 1; }
#udisksctl unmount -b "${LOOP}p1"

INFO=$(MTOOLSRC=<( echo "drive c: file=\"$NAME\" partition=1" ) \
        minfo -i "${NAME}"@@"${OFFSET}")

#parse CHS https://stackoverflow.com/a/30872943
HEA=$( sed -En '0,/^heads: ([0-9]+)$/{s/^heads: ([0-9]+)$/\1/p}' <<< "$INFO" )
SEC=$( sed -En '0,/^sectors per track: ([0-9]+)$/{s/^sectors per track: ([0-9]+)$/\1/p}' <<< "$INFO" )
CYL=$( sed -En '0,/^cylinders: ([0-9]+)$/{s/^cylinders: ([0-9]+)$/\1/p}' <<< "$INFO" )

if [[ -r "$CLONE" && -f "$CLONE" ]]; then #exists and is readable and is regular file
    #mtools supplies a option to provide a 'third' config file which we're using here to redefine c and d
    #without nuking or creating config files (-i is not enough for two images). This keeps attributes


    MTOOLSRC=<( echo "drive c: file=\"$NAME\" partition=1"; echo "drive d: file=\"$CLONE\" partition=1" ) \
        mcopy -bspmQ "d:" "c:"  >/dev/null \
        || { echo  >&2 "$ME: error transfering the filesystem from $CLONE. Exit"; exit 1; }

elif [[ -v CLONE ]]; then #is not file or is not readable and was assigned
    echo  >&2 "$ME: given clone is not readable. Exit"; exit 1;
fi

if (( FAT == 32 )); then
cat >&2 << ENDOFHELP
Success - FAT32 is only used for booting in dosbox

mount as c (3 is d) for boot in dosbox:

imgmount 2 "$BASE" -size 512,$SEC,$HEA,$CYL -nofs

mount in linux, doesn't set DOS-unique attributes if written to:

udisksctl loop-setup -f "$BASE"

unmount in linux with DEV the device of the previous command:

udisksctl unmount -b "DEVp1"

clone/shrink/expand image: use the -c flag, it preserves attributes
ENDOFHELP
else
cat >&2 << ENDOFHELP
Success!

mount as c in dosbox:

imgmount c "$BASE" -size 512,$SEC,$HEA,$CYL

mount in linux, doesn't set DOS-unique attributes if written to:

udisksctl loop-setup -f "$BASE"

unmount in linux with DEV the device of the previous command:

udisksctl unmount -b "DEVp1"

clone/shrink/expand image: use the -c flag, it preserves attributes
ENDOFHELP
fi