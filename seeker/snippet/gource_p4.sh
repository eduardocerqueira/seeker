#date: 2021-10-07T17:09:42Z
#url: https://api.github.com/gists/389f71307482d647817cabeb8e34ab04
#owner: https://api.github.com/users/iUltimateLP

#!/bin/bash

# Put your P4 credentials here!
P4_IP="some.domain.com:1666"
P4_USER=John
P4_PASS=johndoe1234

rm -rf gource.log
p4 -p $P4_IP -u $P4_USER -P $P4_PASS changes |awk '{print $2}'|p4 -p $P4_IP -u $P4_USER -P $P4_PASS -x - describe -s|awk '(/^Change / || /^... /) {if ($1 == "Change") {u=substr($4,1,index($4,"@")-1); t = $(NF-1) " " $NF; gsub("/"," ",t); gsub(":"," ",t);time=mktime(t);} else {if ($NF=="add") {c="A";} else if ($NF=="delete") {c="D";} else {c="M";};f=substr($2,3,index($2,"#")-3);print time "|" u "|" c "|" f;}}'|sort -n > gource.log
