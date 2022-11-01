#date: 2022-11-01T17:04:56Z
#url: https://api.github.com/gists/2bdd6758c5fc83554f32a6ce0e594c5a
#owner: https://api.github.com/users/trdenton

#!/usr/bin/env bash
# Troy Denton 2022

# arg 1 is VCD file
# arg 2 is output vcd file
# arg 3 is percent of file you wish to keep

# Usage: ./truncate_vcd.sh lab2.vcd /tmp/lab2_out.vcd 5
# this will write 5% of the vcd to /tmp/lab2_out.vcd

if [ "$1" == "$2" ]; then
	echo "i dont think you want to do that..."
	exit
fi

if [ "$3" -lt 1 ] || [ "$3" -gt 100 ]; then
	echo "percentage should be 0-100"
	exit
fi

NUM_TSTAMPS=$(grep '^#' $1 | wc -l)

echo $NUM_TSTAMPS 
NUM_TO_KEEP=$( echo "$NUM_TSTAMPS * $3 / 100" | bc )

echo "We will keep $NUM_TO_KEEP entries"

awk "BEGIN{NT=0}{ if (\$0 ~ /^#/) {NT++}; if (NT <= $NUM_TO_KEEP) {print}}" <$1 >$2

# we need the last timestamp field... theres a better way? but this is the quick fix
SIZE_OUT=$(wc -l $2 | cut -d\  -f1)
SIZE_OUT=$( echo "$SIZE_OUT + 1" | bc )
LAST_T=$(head -n $SIZE_OUT $1 | tail -1)
echo $LAST_T >>$2
