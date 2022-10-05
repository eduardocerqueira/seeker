#date: 2022-10-05T17:39:28Z
#url: https://api.github.com/gists/5a10dde240ee7c47c22d0212be3dc479
#owner: https://api.github.com/users/mcgheee

#!/bin/bash

sourcefile=$1

# Get amount of space used by source file & double it in human readable format
testspace=$(du -h $sourcefile | awk '{print $1}' | awk 'BEGIN{FPAT="([[:alpha:]]+)|([[:digit:]]+)"}{$1=$1*2;print}' | tr -d ' ')

# Array containing each compression command & filetype
declare -r -A zipcmds=(
[gzip]='gz'
[pbzip2]='bz2'
[xz]='xz'
)

echo "WARNING! This may take a while for large files & will consume up to $testpace more disk space during testing."

echo "Creating copy of $sourcefile named ziptest_copy ..."
cp $sourcefile ziptest_copy

echo "$sourcefile Zip Test Benchmarking Results" > ziptest_results.txt
echo -e "-------------------------------------------------------\n" >> ziptest_results.txt

echo "Beginning compression / decompression benchmarking ..."

for command in "${!zipcmds[@]}"
do
        # r is the compression ratio used. Typically 1 is fast, 9 is best, 6 is default. (xz uses 9 as default)
        for r in 1 6 9
        do
                echo "$command -$r" | tee -a ziptest_results.txt
                echo -e "---------------------------------\n" >> ziptest_results.txt
                /usr/bin/time -f "Compression:\n\t%C\n\tReal Time: %E\n\tProcessor used: %P\n\tPeak Memory Usage: %M" -- $command -$r ziptest_copy 2>&1 | tee -a ziptest_results.txt
                du ziptest_copy.${zipcmds[$command]} | awk 'BEGIN {printf "\tDisk Usage: "} {print $1}' | tee -a ziptest_results.txt
                /usr/bin/time -f "Decompression:\n\t%C\n\tReal Time: %E\n\tProcessor used: %P\n\tPeak Memory Usage: %M" -- $command -d ziptest_copy.${zipcmds[$command]} 2>&1 | tee -a ziptest_results.txt
                echo -e "\n" | tee -a ziptest_results.txt
        done
done

echo "Cleaning up ziptest_copy ..."
rm -f ziptest_copy

echo "Done! Results stored in ziptest_results.txt"
