#date: 2023-12-14T16:51:42Z
#url: https://api.github.com/gists/7c3cdc91043e7bc399b5ffc592318088
#owner: https://api.github.com/users/clarkritchie

#! /bin/bash
#
# quick and dirty -- for each row in CSV 1, see if that field exists in CSV 2
#
skip_headers=1
while IFS="," read -r c1 c2 c2 c4 c5 c6 c7 # 'email' is field c2
do
	if ((skip_headers))
    then
    	echo "skipping header row"
        ((skip_headers--))
    else
    	echo "Checking $c2"
    	result=`egrep -i $c2  test2.csv`
    	if [ ! -z "$result" ]; then
			echo "*** $c2 is in both files ***"
		fi
  	fi
done < test1.csv