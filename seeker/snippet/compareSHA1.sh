#date: 2023-03-31T17:10:15Z
#url: https://api.github.com/gists/436c227439c45a12ab49cc51d0f3e8be
#owner: https://api.github.com/users/geugenm

#!/bin/bash

# Check if the script is called with exactly two file arguments
if [ $# -ne 2 ]; then
    echo "Usage: $(basename "$0") file1 file2"
    exit 1
fi

# Iterate over each file argument
for file in "$@"; do
    # Check if the file exists
    if [ ! -f "$file" ]; then
        echo "Error: $file does not exist"
        exit 1
    fi
done

# Get the SHA1 checksums of the files
sha1_1=$(sha1sum "$1" | cut -d ' ' -f 1)
sha1_2=$(sha1sum "$2" | cut -d ' ' -f 1)

# Print the checksums with file names and highlighted characters
echo -n -e "\e[33m$1:\e[0m "
for i in $(seq 1 ${#sha1_1}); do
    if [ "${sha1_1:$i-1:1}" == "${sha1_2:$i-1:1}" ]; then
        echo -n -e "\e[32m${sha1_1:$i-1:1}\e[0m"
    else
        echo -n -e "\e[31m${sha1_1:$i-1:1}\e[0m"
    fi
done
echo

echo -n -e "\e[33m$2:\e[0m "
for i in $(seq 1 ${#sha1_2}); do
    if [ "${sha1_2:$i-1:1}" == "${sha1_1:$i-1:1}" ]; then
        echo -n -e "\e[32m${sha1_2:$i-1:1}\e[0m"
    else
        echo -n -e "\e[31m${sha1_2:$i-1:1}\e[0m"
    fi
done
echo

# Check if the SHA1 checksums are matching or not
if [ "$sha1_1" == "$sha1_2" ]; then
    echo "Hashes match"
else
    echo "Hashes do not match"
fi
