#date: 2022-03-03T16:54:43Z
#url: https://api.github.com/gists/ec5adb68cbbd461ab11fa5993711b99c
#owner: https://api.github.com/users/christopherwoodall

#!/bin/sh

#duplicate smbios and change system info
#Version : 1.6.4
#https://gist.github.com/nnn-dev/

_help()
{
cat <<_end_
This program duplicate smbios and change system info.
Syntax: $0 [-f device] [-B] [-D opt=value] [-o opt] out
Options:
 -f : select device (default /sys/firmware/dmi/tables/* or /dev/mem)
 -B : batch mode (no prompt)
 -D : indicate a value. Opt can be:
  MANUFACTURER : change Manufacturer
  PRODUCT      : change Product Name 
  VERSION      : change Version
  SERIAL       : change Serial Number
  UUID         : change UUID (if supported). Must be hexadecimal
  FAMILY       : change Family (if supported).
  SKU          : change SKU Number (if supported).
 -o : indicate a option. Opt can be:
  dmi          : create dmi entries (out will be a folder)
_end_
}

help()
{
if [ "$HELP" = '1' ]; then
 _help
 return 0
else
 _help >&2
 return "$HELP"
fi
}

#awk support posix option?
if `awk --posix 2>&1 | grep -q 'not an option: --posix'`; then
 awk_posix_opt=''
else
 awk_posix_opt='--posix'
fi

VERBOSE=0
HELP=0
#OPT_MANUFACTURER
#OPT_PRODUCT
#OPT_VERSION
#OPT_SERIAL
#OPT_UUID
#OPT_FAMILY
#OPT_SKU
BATCH=0
OPT_DMI=0

if [ -e "/sys/firmware/dmi/tables/smbios_entry_point" ]; then
 FILE='/sys/firmware/dmi/tables/*'
else
 FILE=/dev/mem
fi

#RANDOMIZESOURCE=/dev/zero
#RANDOMIZESOURCE=/dev/random
RANDOMIZESOURCE=/dev/urandom

while getopts "vh?BD:f:o:" opts ; do
case $opts in
 'v') VERBOSE=$((VERBOSE + 1));;
 'h|?') HELP=1;;
 'B') BATCH=1;;
 'D') n=$(echo "${OPTARG}" | cut -f1 -d"=");
 q="'"; dq='"' 
 v=$(echo "${OPTARG}" | cut -f2- -d'='|sed '1,$s/'"$q"'/'"$q$dq$q$dq$q"'/g') 
 eval "OPT_${n}='$v'" ;;
 'o') if [ "${OPTARG}" = 'dmi' ]; then
 OPT_DMI=1
 fi;;
 'f') FILE="${OPTARG}";;
 *) HELP=2;;
esac
done


shift $((OPTIND-1))


if [ "$HELP" != '0' ]; then
 help $HELP
 exit $?
fi

if `echo "${FILE}" | grep -q '/sys/firmware/dmi/tables'`; then
  FILEENTRY="/sys/firmware/dmi/tables/smbios_entry_point" 
  FILEDMI="/sys/firmware/dmi/tables/DMI"
  RANDOMIZESOURCE=/dev/zero
 else
  FILEDMI="${FILE}" 
  FILEENTRY="${FILE}"
 fi 

if [ ! -e "$FILEENTRY" ]; then
 echo "$FILEENTRY not found" >&2
 exit 3
fi


if [ ! -e "$FILEDMI" ]; then
 echo "$FILEDMI not found" >&2
 exit 3
fi


if [ -n "$1" ]; then
 OUT=$1
 if [ "$OPT_DMI" -eq '1' ]; then
  if [ -f "$OUT" ]; then
   help 0
   exit $?
  fi
  [ -d "$OUT" ] && rm -Rf $OUT/tables $OUT/entries
  mkdir -p $OUT/tables
  mkdir -p $OUT/entries
  OUT_ENTRIES=$OUT/entries
  OUT_HEADER=$OUT/tables/smbios_entry_point
  OUT_DMI=$OUT/tables/DMI  
 else
  [ -f "$OUT" ] && rm "$OUT"
  OUT_HEADER=$OUT
  OUT_DMI=$OUT
 fi
else
 help 0
 exit $?
fi

CURSOR=0
CURSORO=0
unset R
unset S

fn_read_byte()
{
 R=$(od -v -j $CURSOR -N 1 -t u1 "$FILE" | sed '1,$s/^[0-9]* *//g')
 CURSOR=$((CURSOR + 1))
 S=$R
}

fn_write_byte()
{
 echo "$1" |LANG=C awk '{ printf("%c",$1); }' >>"$OUT"
 CURSORO=$((CURSORO + 1))
}

fn_read_word()
{
 R=$(od -v -j $CURSOR -N 2 -t u2 "$FILE"  | sed '1,$s/^[0-9]* *//g')
 S=0; for i in 0 1; do
   S=$((S + $(od -v -j $((CURSOR + i)) -N 1 -t u1 "$FILE"  | sed '1,$s/^[0-9]* *//g')))
 done
 CURSOR=$((CURSOR + 2))
}

fn_write_word()
{
echo "$1" | LANG=C awk '{ t=sprintf("%04x",$1)
for(i=length(t)-1;i>0;i-=2){
 c=substr(t,i,2)
 print "0x"c
}
}' | LANG=C awk ${awk_posix_opt} '{ printf("%d\n",$1) }' | LANG=C awk '{ printf("%c",$1) }' >>"$OUT"
CURSORO=$((CURSORO + 2))
}

fn_read_dword()
{
 R=$(od -v -j $CURSOR -N 4 -t u4 "$FILE"  | sed '1,$s/^[0-9]* *//g')
 S=0; for i in 0 1 2 3; do
   S=$((S + $(od -v -j $((CURSOR + i)) -N 1 -t u1 "$FILE"  | sed '1,$s/^[0-9]* *//g')))
 done
 CURSOR=$((CURSOR + 4))
}

fn_write_dword()
{
echo "$1" | LANG=C awk '{ t=sprintf("%08x",$1)
for(i=length(t)-1;i>0;i-=2){
 c=substr(t,i,2)
 print "0x"c
}
}' | LANG=C awk ${awk_posix_opt} '{ printf("%d\n",$1) }' | LANG=C awk '{ printf("%c",$1) }' >>"$OUT"

CURSORO=$((CURSORO + 4))
}


# $1 = nb chars
fn_read_chars()
{
 R=$(od -v -j $CURSOR -N "$1" -c "$FILE"  | sed '1,$s/^[0-9]* *//g')
 Rd=$(od -v -j $CURSOR -N "$1" -t u1 "$FILE"  | sed '1,$s/^[0-9]* *//g' | tr '\n' ' ')
 S=0; i=0
 while [ "$i" -lt "$1" ]; do
   S=$((S + $(od -v -j $((CURSOR + i)) -N 1 -t u1 "$FILE"  | sed '1,$s/^[0-9]* *//g')))
   i=$((i+1))
 done
 CURSOR=$((CURSOR + $1))
}

fn_read_string()
{
#browse until \0
c=$CURSOR
R=''
while [ "$R" != '0' ]; do
 fn_read_byte
done
o=$CURSOR
l=$((CURSOR - c))
CURSOR=$c
fn_read_chars $l
# don't read final zero but skip it
fn_move_to $o
}

# write_chars : use a spaced format decimal positio
fn_write_chars()
{
echo "$@" | LANG=C  awk -F ' ' '{
 for(i=1;i<=NF;i++) {
 printf("%d ",$i) 
 }}'| LANG=C  awk -F ' ' '{
 for(i=1;i<=NF;i++) {
 printf("%c",$i) 
 }}'  >>"$OUT"
CURSORO=$((CURSORO + $(echo "$@" | awk -F ' ' '{ print NF }')))
}

fn_move_to()
{
CURSOR=$1
}

fn_write_from_source()
{
if [ "$VERBOSE" -ge '2' ] ;then
dd if=$1 ibs=1 obs=1 skip=$2 count=$3 conv=notrunc oflag=append>>"$OUT"
else
dd if=$1 ibs=1 obs=1 skip=$2 count=$3 conv=notrunc oflag=append>>"$OUT" 2>/dev/null
fi
CURSORO=$((CURSORO+$3))
[ "$VERBOSE" -ge '2' ] && debug "CURSORO=$CURSORO / $(du -b $OUT)"
}

fn_write_move_to()
{
if [ "$1" -gt "${CURSORO}" ]; then
 if [ -n "$RANDOMIZESOURCE" ]; then
	fn_write_from_source /dev/urandom 0 $(($1 - CURSORO))
 else
	fn_write_from_source $FILE $CURSORO $(($1 - CURSORO))
 fi
fi
}

fn_search_header()
{
 fd=$(od -Ad -v -a "$FILE" 2>/dev/null | grep '^[0-9]*[[:space:]]*_[[:space:]]*S[[:space:]]*M[[:space:]]*_'  | head -n 1 | cut -f 1 -d ' ' | sed '1,$s/^0*//g')
 [ -z "$fd" ] && fd=0
 fn_move_to $fd
 last=$((CURSOR + 32))
 while [ "$CURSOR" -le "$last" ]; do
   c=$CURSOR
   fn_read_byte
   if [ "$R" = '95' ]; then #_
    fn_read_byte
    if [ "$R" = '83' ]; then #S
     fn_read_byte
     if [ "$R" = '77' ]; then #M
       fn_read_byte
       if [ "$R" = '95' ]; then #_
         CURSOR=$c
         return 0
       fi
     fi
    fi
   fi
   CURSOR=$((c + 1))
 done
 return 1
}

fn_8add()
{
  res=0
  while [ "$#" -ne '0' ]; do
    if [ "$1" -le '255' ]; then
      res=$(( (res + $1)%256 ))
    else
      res=$(printf "%08x" "$1" | LANG=C awk -v "S=$res" 'BEGIN{
      printf("echo $(((%d",S)}{
        for(i=1;i<=length($0);i+=2){
          printf("+0x%s",substr($0,i,2))
        }
      }END{printf("%s",")%256))")}'|sh)
    fi
    shift 1
  done 
  echo "$res" 
}

fn_dummy_header()
{
check=0
b=$CURSOR
HEADER_ANCHOR='95 83 77 95'
HEADER_CHECKSUM=0
HEADER_LENGTH=31
HEADER_VERSION1=2
HEADER_VERSION2=0
HEADER_MAXSSIZE=150
HEADER_EPR=0
HEADER_FORMATAREA='0 0 0 0 0'
HEADER_IANCHOR='95 68 77 73 95'
HEADER_ICHECKSUM=0
HEADER_STLENGTH=12
HEADER_STRUCTTABLEADDR=0
HEADER_NBSTRUCTS=1
HEADER_SMBIOSREV=0
}

fn_read_header()
{
check=0
b=$CURSOR

#printf "Anchor "
fn_read_chars 4
HEADER_ANCHOR=${Rd}
check=$(fn_8add "$check" $S)
[ "$VERBOSE" -ge '2' ] && debug ".Anchor=${HEADER_ANCHOR}"

#printf "Checksum "
fn_read_byte
HEADER_CHECKSUM=$R
check=$(fn_8add "$check" $S)
[ "$VERBOSE" -ge '2' ] && debug ".Checksum=${HEADER_CHECKSUM}"

#printf "Length "
fn_read_byte
HEADER_LENGTH=$R
check=$(fn_8add "$check" $S)
last=$((b + R))
[ "$VERBOSE" -ge '2' ] && debug ".Length=${HEADER_LENGTH}"

#printf "SMBIOS Version "
fn_read_byte
HEADER_VERSION1=$R
check=$(fn_8add "$check" $S)
fn_read_byte
HEADER_VERSION2=$R
check=$(fn_8add "$check" $S)
[ "$VERBOSE" -ge '2' ] && debug ".SMBIOS Version=${HEADER_VERSION1}.${HEADER_VERSION2}"

#printf "Maximum Structure Size "
fn_read_word
HEADER_MAXSSIZE=$R
check=$(fn_8add "$check" $S)
[ "$VERBOSE" -ge '2' ] && debug ".Maximum Structure Size=${HEADER_MAXSSIZE}"

#printf "Entry Point Revision "
fn_read_byte
HEADER_EPR=$R
check=$(fn_8add "$check" $S)
[ "$VERBOSE" -ge '2' ] && debug ".Entry Point Revision=${HEADER_EPR}"

#printf "Formatted Area "
fn_read_chars 5
HEADER_FORMATAREA="$Rd"
check=$(fn_8add "$check" $S)
[ "$VERBOSE" -ge '2' ] && debug ".Formatted Area=${HEADER_FORMATAREA}"

checki=0
#printf "Intermediate Anchor "
fn_read_chars 5
HEADER_IANCHOR="$Rd"
check=$(fn_8add "$check" $S)
checki=$(fn_8add "$checki" $S)
[ "$VERBOSE" -ge '2' ] && debug ".Intermediate Anchor=${HEADER_IANCHOR}"

#printf "Intermediate Checksum "
fn_read_byte
HEADER_ICHECKSUM=$R
check=$(fn_8add "$check" $S)
checki=$(fn_8add "$checki" $S)
[ "$VERBOSE" -ge '2' ] && debug ".Intermediate Checksum=${HEADER_ICHECKSUM}"

#printf "Structure Table Length "
fn_read_word
HEADER_STLENGTH=$R
check=$(fn_8add "$check" $S)
checki=$(fn_8add "$checki" $S)
[ "$VERBOSE" -ge '2' ] && debug ".Structure Table Length=${HEADER_STLENGTH}"

#printf "Structure Table Address "
fn_read_dword
HEADER_STRUCTTABLEADDR=$R
check=$(fn_8add "$check" $S)
checki=$(fn_8add "$checki" $S)
[ "$VERBOSE" -ge '2' ] && debug ".Structure Table Address=${HEADER_STRUCTTABLEADDR}"


#printf "Nb SMBIOS Structures "
fn_read_word
HEADER_NBSTRUCTS=$R
check=$(fn_8add "$check" $S)
checki=$(fn_8add "$checki" $S)
[ "$VERBOSE" -ge '2' ] && debug ".Nb SMBIOS Structures=${HEADER_NBSTRUCTS}"

#printf "SMBIOS BCD Revision "
fn_read_byte
HEADER_SMBIOSREV=$R
[ "$VERBOSE" -ge '2' ] && debug ".SMBIOS BCD Revision=${HEADER_SMBIOSREV}"

check=$(fn_8add "$check" $S)
checki=$(fn_8add "$checki" $S)

[ "$VERBOSE" -ge '3' ] && debug "CURSOR=$CURSOR/CHECKSUM=$check/CI=$checki"
}

fn_write_header()
{
check=0
b=$CURSORO

#recalculate checkum
check=$(fn_8add "$check" ${HEADER_ANCHOR})
check=$(fn_8add "$check" $HEADER_LENGTH)
check=$(fn_8add "$check" $HEADER_VERSION1)
check=$(fn_8add "$check" $HEADER_VERSION2)
check=$(fn_8add "$check" $HEADER_MAXSSIZE)
check=$(fn_8add "$check" $HEADER_EPR)
check=$(fn_8add "$check" $HEADER_FORMATAREA)
HEADER_CHECKSUM=$((256-check))

#printf "Anchor "
fn_write_chars "${HEADER_ANCHOR}"
[ "$VERBOSE" -ge '2' ] && debug ".Anchor=${HEADER_ANCHOR}"

#printf "Checksum "
fn_write_byte $HEADER_CHECKSUM
[ "$VERBOSE" -ge '2' ] && debug ".Checksum=${HEADER_CHECKSUM}"

#printf "Length "
fn_write_byte $HEADER_LENGTH
[ "$VERBOSE" -ge '2' ] && debug ".Length=${HEADER_LENGTH}"

#printf "SMBIOS Version "
fn_write_byte $HEADER_VERSION1
fn_write_byte $HEADER_VERSION2
[ "$VERBOSE" -ge '2' ] && debug ".SMBIOS Version=${HEADER_VERSION1}.${HEADER_VERSION2}"

#printf "Maximum Structure Size "
fn_write_word $HEADER_MAXSSIZE
[ "$VERBOSE" -ge '2' ] && debug ".Maximum Structure Size=${HEADER_MAXSSIZE}"

#printf "Entry Point Revision "
fn_write_byte $HEADER_EPR
[ "$VERBOSE" -ge '2' ] && debug ".Entry Point Revision=${HEADER_EPR}"

#printf "Formatted Area "
fn_write_chars "$HEADER_FORMATAREA"
[ "$VERBOSE" -ge '2' ] && debug ".Formatted Area=${HEADER_FORMATAREA}"

#recalculate intermediate checkum
checki=0
checki=$(fn_8add "$checki" $HEADER_IANCHOR)
checki=$(fn_8add "$checki" $HEADER_STLENGTH)
checki=$(fn_8add "$checki" $HEADER_STRUCTTABLEADDR)
checki=$(fn_8add "$checki" $HEADER_NBSTRUCTS)
checki=$(fn_8add "$checki" $HEADER_SMBIOSREV)
HEADER_ICHECKSUM=$((256 - checki))

#printf "Intermediate Anchor "
fn_write_chars "$HEADER_IANCHOR"
[ "$VERBOSE" -ge '2' ] && debug ".Intermediate Anchor=${HEADER_IANCHOR}"

#printf "Intermediate Checksum "
fn_write_byte $HEADER_ICHECKSUM
[ "$VERBOSE" -ge '2' ] && debug ".Intermediate Checksum=${HEADER_ICHECKSUM}"

#printf "Structure Table Length "
fn_write_word $HEADER_STLENGTH
[ "$VERBOSE" -ge '2' ] && debug ".Structure Table Length=${HEADER_STLENGTH}"

#printf "Structure Table Address "
fn_write_dword $HEADER_STRUCTTABLEADDR
[ "$VERBOSE" -ge '2' ] && debug ".Structure Table Address=${HEADER_STRUCTTABLEADDR}"

#printf "Nb SMBIOS Structures "
fn_write_word $HEADER_NBSTRUCTS
[ "$VERBOSE" -ge '2' ] && debug ".Nb SMBIOS Structures=${HEADER_NBSTRUCTS}"

#printf "SMBIOS BCD Revision "
fn_write_byte $HEADER_SMBIOSREV
[ "$VERBOSE" -ge '2' ] && debug ".SMBIOS BCD Revision=${HEADER_SMBIOSREV}"

[ "$VERBOSE" -ge '3' ] && debug "CURSOR=$CURSOR/CHECKSUM=$check/CI=$checki"
}

# $1 = nb
fn_read_struct()
{
#printf "Type "
fn_read_byte
S_TYPE=$R
eval "S${1}_TYPE='${R}'"

#printf "Length "
fn_read_byte
eval "S${1}_LENGTH='${R}'"
S_LENGTH=$R
l=$((R - 4))

#printf "Handle "
fn_read_word
eval "S${1}_HANDLE='${R}'"
S_HANDLE=$R

if [ "$l" -gt '3' ]; then
#printf "Values "
fn_read_chars $l
eval "S${1}_VALUES='${Rd}'"
S_VALUES="$Rd"
else
S_VALUES=''
fi

#echo "Strings"
R=''
Ri=1
CO=$CURSOR
while [ "$R" != '\0' ]; do
 fn_read_string
 eval "S${1}_STRING${Ri}='${Rd}'"
 eval "S_STRING${Ri}='${Rd}'"
 [ "$VERBOSE" -ge '2' ] && debug ".string${Ri}=$(fn_darray2str "${Rd}")"
 Ri=$((Ri + 1))
done
Ri=$((Ri - 1))
eval "S${1}_NBSTRINGS='${Ri}'"
eval "S_NBSTRINGS='${Ri}'"
eval "S_STRLENGTH='$((CURSOR - CO))'"
#from dmidecode.c : two 0 after a struct
if [ "$Ri" -eq '1' ];then
 fn_read_byte
fi
[ "$VERBOSE" -ge '3' ] && debug ".handle=${S_HANDLE},type=${S_TYPE},length=${S_LENGTH}"
}

fn_dummy_struct1()
{
S1_TYPE=1
S_TYPE=1
S1_LENGTH=8
S_LENGTH=8
fn_read_word
S1_HANDLE=$R
S_HANDLE=$R
Ri=4
while [ "$Ri" -gt '0' ]; do
 Rd='0'
 eval "S1_STRING${Ri}='${Rd}'"
 eval "S_STRING${Ri}='${Rd}'"
 Ri=$((Ri - 1))
done
Ri=$((Ri - 1))
eval "S1_NBSTRINGS='4'"
eval "S_NBSTRINGS='4'"
eval "S_STRLENGTH='4'"
CURSOR=$((CURSOR + 17))
}


# $1 = nb
fn_write_struct()
{
#printf "Type "
fn_write_byte $S_TYPE

#printf "Length "
fn_write_byte $S_LENGTH

#printf "Handle "
fn_write_word $S_HANDLE

if [ "$S_LENGTH" -gt '3' ]; then
#printf "Values "
fn_write_chars "$S_VALUES"
fi

#echo "Strings"
R=''
Ri=1
while [ "$Ri" -le "$S_NBSTRINGS" ]; do
 fn_write_chars "$(eval "echo \$S_STRING${Ri}")"
 Ri=$((Ri + 1))
done
[ "${S_NBSTRINGS}" -eq '1' ] && fn_write_byte 0
}

fn_write_entry()
{
wei=0
while [ -d "${OUT_ENTRIES}/${S_TYPE}-${wei}" ]; do 
 wei=$((wei + 1))
done
wed="${OUT_ENTRIES}/${S_TYPE}-${wei}"
mkdir -p "${wed}"
echo ${wei} >${wed}/instance
echo "${S_TYPE}" >${wed}/type
echo "${S_LENGTH}" >${wed}/length
echo "$(($1 - 1))" >${wed}/position
echo "${S_HANDLE}" >${wed}/handle
OUTo=$OUT
OUT=${wed}/raw
fn_write_struct
OUT=$OUTo
}

#write really string
fn_write_rstring0()
{
if [ -n "$1" ]; then
  fn_write_chars "$(printf "%s" "$@" | od -v -t u1 | sed '1,$s/^[0-9]* *//g' | tr '\n' ' ')"
else
  fn_write_chars ' '
fi
fn_write_byte 0
}

debug()
{
 echo "$@" >&2
}

fn_darray2str()
{
 echo "$@" | LANG=C  awk -F ' ' '
 {
 for(i=1;i<=NF;i++) {
 printf("%d ",$i) 
 }
 }' | LANG=C  awk -F ' ' '
 {
 for(i=1;i<NF;i++) {
 printf("%c",$i) 
 }
 }' 
}

fn_writetype1()
{

  # write it
  #printf "Type "
  fn_write_byte $S_TYPE

  #printf "Length "
  fn_write_byte $S_LENGTH

  #printf "Handle "
  fn_write_word $S_HANDLE

  #Manufacturer 1
  fn_write_byte 1
  #Product 2
  fn_write_byte 2
  #Version 3
  fn_write_byte 3
  #Serial 4
  fn_write_byte 4
  if [ "$S_LENGTH" -gt '8' ]; then
  #UUID (16 bytes)
  fn_write_chars "$(echo "$OPT_UUID" | sed '1,$s/[^0-9a-fA-F]*//g' | LANG=C awk -F ' ' '{
    for(i=1;i<=length($0);i+=2){
      if (i>1) { print "printf \" \"" }
      printf("printf $((0x%s))\n",substr($0,i,2))
    }
  }' | sh | LANG=C awk -F' ' '{
   printf("%s",$0);
   for(i=NF;i<16;i++){
    printf(" 0");
   }
  }')"
  fi
  if [ "$S_LENGTH" -ge '25' ]; then
  fn_write_byte "$V6"
  fi
  if [ "$S_LENGTH" -ge '27' ]; then
    o=5
  #Family
    if [ -n "$OPT_FAMILY" ]; then
     fn_write_byte $o
     o=$((o + 1))
    else
     fn_write_byte 0
    fi
  #SKU
    if [ -n "$OPT_SKU" ]; then
     fn_write_byte $o
    else
     fn_write_byte 0
    fi
  fi
  
  #write strings
  fn_write_rstring0 "$OPT_MANUFACTURER"
  fn_write_rstring0 "$OPT_PRODUCT"
  fn_write_rstring0 "$OPT_VERSION"
  fn_write_rstring0 "$OPT_SERIAL"
  if [ "$S_LENGTH" -ge '27' ]; then
     [ -n "$OPT_FAMILY" ] && fn_write_rstring0 "$OPT_FAMILY"
     [ -n "$OPT_SKU" ] && fn_write_rstring0 "$OPT_SKU"
  fi
  fn_write_byte 0
}

DUMMY=0
if [ \( "$FILEENTRY" = '/dev/random' \) -o \( "$FILEENTRY" = '/dev/urandom' \) -o \( "$FILEENTRY" = '/dev/zero' \) ]; then
 DUMMY=1
fi

#search for a dmidecode dump
[ "$VERBOSE" -ge '1' ] && debug "Entry file=$FILEENTRY"
[ "$VERBOSE" -ge '1' ] && debug "Searching header..."
FILE=$FILEENTRY
if [ "$DUMMY" -eq '1' ]; then
 CURSOR=983040
 HEADER_CURSOR=$CURSOR
 fn_dummy_header
 HEADER_STRUCTTABLEADDR=0
else
 fn_search_header
 if [ "$?" -ne '0' ]; then
  echo 'Cannot find header' >&2
  exit 2
 fi
 HEADER_CURSOR=$CURSOR
 #[ "$VERBOSE" -ge '1' ] && debug  "Writing space before header"
 #fn_write_move_to $CURSOR
 [ "$VERBOSE" -ge '1' ] && debug  "HEADER found @$CURSOR"
 [ "$VERBOSE" -ge '1' ] && debug  "Reading Header"
 fn_read_header
 #[ "$VERBOSE" -ge '1' ] && debug  "Writing Header"
 #fn_write_header
 if [ "$FILEDMI" = "$FILEENTRY" ]; then
 [ "$VERBOSE" -ge '1' ] && debug  "STRUCTS found @${HEADER_STRUCTTABLEADDR}"
  fn_move_to ${HEADER_STRUCTTABLEADDR}
 else
  HEADER_CURSOR=983040
  HEADER_STRUCTTABLEADDR=0 #$((HEADER_CURSOR + CURSOR + 1))
  FILE=$FILEDMI
  fn_move_to 0
 fi
fi
[ "$VERBOSE" -ge '1' ] && debug "DMI file=$FILEDMI"
#fn_write_move_to ${HEADER_STRUCTTABLEADDR}
[ "$VERBOSE" -ge '1' ] && debug "Finding Type 1 structures"
ii=0
while [ "$ii" -lt "${HEADER_NBSTRUCTS}" ]; do
 ii=$((ii+1))
 [ "$VERBOSE" -ge '1' ] && debug "Reading $ii/$HEADER_NBSTRUCTS structs @$CURSOR"
 bb=$CURSOR
 if [ "$DUMMY" -eq '1' ]; then
  fn_dummy_struct1
 else
  fn_read_struct $ii
 fi
 bl=$((CURSOR-bb))

 if [ "${S_TYPE}" = '1' ]; then
  #Type 1 information. Read and Prompt for new values
  V1=$(echo "${S_VALUES}" | tr -s ' '| cut -f 1 -d ' ')
  T1=$(eval "echo \${S_STRING${V1}}")
  Td="$(fn_darray2str "$T1")"
  if [ "$BATCH" -eq '0' ] && [ -z "$OPT_MANUFACTURER" ]; then 
   echo "Manufacturer [$Td] ?"
   read -r OPT_MANUFACTURER
  fi
  [ -z "$OPT_MANUFACTURER" ] && OPT_MANUFACTURER="$Td"
  V2=$(echo "${S_VALUES}" | tr -s ' '| cut -f 2 -d ' ')
  T2=$(eval "echo \${S_STRING${V2}}")
  Td="$(fn_darray2str "$T2")"
  if [ "$BATCH" -eq '0' ] && [ -z "$OPT_PRODUCT" ]; then 
   echo "Product [$Td] ?"
   read -r OPT_PRODUCT
  fi
  [ -z "$OPT_PRODUCT" ] && OPT_PRODUCT="$Td"
  V3=$(echo "${S_VALUES}" | tr -s ' '| cut -f 3 -d ' ')
  T3=$(eval "echo \${S_STRING${V3}}")  
  Td="$(fn_darray2str "$T3")"
  if [ "$BATCH" -eq '0' ] && [ -z "$OPT_VERSION" ]; then 
   echo "Version [$Td] ?"
   read -r OPT_VERSION
  fi
  [ -z "$OPT_VERSION" ] && OPT_VERSION="$Td"
  V4=$(echo "${S_VALUES}" | tr -s ' '| cut -f 4 -d ' ')
  T4=$(eval "echo \${S_STRING${V4}}")
  Td="$(fn_darray2str "$T4")"
  if [ "$BATCH" -eq '0' ] && [ -z "$OPT_SERIAL" ]; then 
   echo "Serial [$Td] ?"
   read -r OPT_SERIAL
  fi
  [ -z "$OPT_SERIAL" ] && OPT_SERIAL="$Td"
  if [ "$S_LENGTH" -ge '9' ]; then
  V5=$(echo "${S_VALUES}" | tr -s ' '| cut -f 5-20 -d ' ' | LANG=C awk -F' ' '{
   for(i=1;i<=NF;i++){
     printf("%02x",$i)
   }
  }')
  if [ "$BATCH" -eq '0' ] && [ -z "$OPT_UUID" ]; then 
   echo "UUID [$V5] ?"
   read -r OPT_UUID
  fi
  [ -z "$OPT_UUID" ] && OPT_UUID="$V5"
  fi
  if [ "$S_LENGTH" -ge '25' ]; then
  V6=$(echo "${S_VALUES}" | tr -s ' '| cut -f 21 -d ' ')
  fi
  if [ "$S_LENGTH" -ge '27' ]; then
  V7=$(echo "${S_VALUES}" | tr -s ' '| cut -f 22 -d ' ')
  T7=$(eval "echo \${S_STRING${V7}}")
  Td="$(fn_darray2str "$T7")"
  if [ "$BATCH" -eq '0' ] && [ -z "$OPT_FAMILY" ]; then 
   echo "family [$Td] ?"
   read -r OPT_FAMILY
  fi
  [ -z "$OPT_FAMILY" ] && OPT_FAMILY="$Td"
  V8=$(echo "${S_VALUES}" | tr -s ' '| cut -f 23 -d ' ')
  T8=$(eval "echo \${S_STRING${V8}}")
  Td="$(fn_darray2str "$T8")"
  if [ "$BATCH" -eq '0' ] && [ -z "$OPT_SKU" ]; then 
   echo "SKU Number [$Td] ?"
   read -r OPT_SKU
  fi
  [ -z "$OPT_SKU" ] && OPT_SKU="$Td"
  fi

  ORIGINAL_HEADER_STLENGTH=$HEADER_STLENGTH  #keep original length for rewrite
  HEADER_STLENGTH=$((HEADER_STLENGTH - S_STRLENGTH))
  l=$(expr length "$OPT_MANUFACTURER")
  [ "$l" -eq '0' ] && l=1                       #if empty we use space
  HEADER_STLENGTH=$((HEADER_STLENGTH + l + 1)) #add 1 for \0
  l=$(expr length "$OPT_PRODUCT")
  [ "$l" -eq '0' ] && l=1
  HEADER_STLENGTH=$((HEADER_STLENGTH + l + 1)) 
  l=$(expr length "$OPT_VERSION")
  [ "$l" -eq '0' ] && l=1
  HEADER_STLENGTH=$((HEADER_STLENGTH + l + 1))
  l=$(expr length "$OPT_SERIAL")
  [ "$l" -eq '0' ] && l=1
  HEADER_STLENGTH=$((HEADER_STLENGTH + l + 1))
  if [ "$S_LENGTH" -ge '27' ]; then
    l=$(expr length "$OPT_FAMILY")
    [ "$l" -ne '0' ] && HEADER_STLENGTH=$((HEADER_STLENGTH + l + 1))
    l=$(expr length "$OPT_SKU")
    [ "$l" -ne '0' ] && HEADER_STLENGTH=$((HEADER_STLENGTH + l + 1))
  fi  
  HEADER_STLENGTH=$((HEADER_STLENGTH + 1))
  break
 fi
done


fn_write_allstructs()
{
fn_move_to ${HEADER_STRUCTTABLEADDR}
ii=0
while [ "$ii" -lt "${HEADER_NBSTRUCTS}" ]; do
 ii=$((ii+1))
 [ "$VERBOSE" -ge '1' ] && debug "Reading $ii/$HEADER_NBSTRUCTS structs @$CURSOR"
 bb=$CURSOR
 if [ "$DUMMY" -eq '1' ]; then
  fn_dummy_struct1
 else
  fn_read_struct $ii
 fi
 fn_write_entry $ii
 bl=$((CURSOR-bb))
 if [ "${S_TYPE}" = '1' ]; then
  [ "$VERBOSE" -ge '1' ] && debug "Writing Modified Type 1"
  fn_writetype1
  [ "$OPT_DMI" -ne '1' ] && break
 else
  #not type 1
  #Works?
  #fn_write_struct
  #use direct copy
  [ "$VERBOSE" -ge '1' ] && debug "Writing struct $ii"
  fn_write_from_source "$FILE" $bb $bl
 fi
done

#copy 
l=$((CURSOR - HEADER_STRUCTTABLEADDR))
l=$((ORIGINAL_HEADER_STLENGTH - l))
if [ "$l" -gt '0' ]; then
 [ "$VERBOSE" -ge '1' ] && debug  "Writing other structs ($l bytes)"
 fn_write_from_source $FILE $CURSOR $l
fi
}

if [ "$OPT_DMI" -eq '1' ]; then
 HEADER_CURSOR=0
 HEADER_STRUCTTABLEADDR=0
fi

if [ "$HEADER_CURSOR" -lt "$HEADER_STRUCTTABLEADDR" ]; then
 OUT=$OUT_DMI
 [ "$VERBOSE" -ge '1' ] && debug  "Writing space before header"
 OUT=$OUT_HEADER
 fn_write_move_to $HEADER_CURSOR
 [ "$VERBOSE" -ge '1' ] && debug  "Writing Header"
 fn_write_header
 OUT=$OUT_DMI
 [ "$VERBOSE" -ge '1' ] && debug  "Writing space before struct"
 fn_write_move_to ${HEADER_STRUCTTABLEADDR}
 fn_write_allstructs
else
 OUT=$OUT_DMI
 [ "$VERBOSE" -ge '1' ] && debug  "Writing space before struct"
 fn_write_move_to ${HEADER_STRUCTTABLEADDR}
 fn_write_allstructs
 OUT=$OUT_HEADER
 [ "$VERBOSE" -ge '1' ] && debug  "Writing space before header"
 fn_write_move_to $HEADER_CURSOR
 [ "$VERBOSE" -ge '1' ] && debug  "Writing Header"
 fn_write_header
fi
if [ "$OPT_DMI" -ne '1' ]; then
 [ "$VERBOSE" -ge '1' ] && debug "Padding"
 if [ "$CURSORO" -lt 1048576 ]; then
  fn_write_move_to 1048576
 fi
fi
#while [ "$((CURSORO % 16))" -ne '0' ]; do
#while [ : ]; do
# dmidecode -d $OUT 2>&1 >/dev/null
# [ "$?" -eq '0' ] && break
# fn_write_move_to $((CURSORO + 1))
#done