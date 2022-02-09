#date: 2022-02-09T16:58:22Z
#url: https://api.github.com/gists/a8555d15dd08d1e5e75acca255ac72e4
#owner: https://api.github.com/users/saulfm08

#!/bin/bash

# create a list of items
echo -e "a\nb\nc\nd"

# bash Array
array=( $(echo -e "a\nb\nc\nd") )

# Array indexes
echo ${!array[@]}

# Array values
echo ${array[@]}

# interating over the array indexes 
for index in ${!array[@]}
do
  echo "Index: $index, Value: ${array[$index]}"
done

# interating over the array indexes 
for item in ${array[@]}
do
  echo $item
done

# read csv and work with lines
while read line
do
  echo $line
done < file.csv