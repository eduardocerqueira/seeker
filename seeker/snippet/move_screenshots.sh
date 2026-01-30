#date: 2026-01-30T17:10:32Z
#url: https://api.github.com/gists/1f13d891c81bd25483ecf36b86e03203
#owner: https://api.github.com/users/ShaunOfTheLive

#!/bin/sh
onboard="/mnt/onboard/"
out_dir="${onboard}.screenshots/"
ext=".png"  # extension is always .png, that's the format Kobo uses for screenshots

for file in "$onboard"screen_*"$ext"; do
    crc32="$(cat $file | busybox gzip -c | busybox tail -c 8 | busybox od -An -N 4 -tx4 | sed 's/^ //' | tr '[:lower:]' '[:upper:]')"
    #ext=${file##*.}
    temp_name="[${crc32}]${ext}"
    if [ -e "${out_dir}${temp_name}" ] && cmp -s "$file" "${out_dir}${temp_name}"; then
        rm "$file"
        #echo "deleting $file, $crc32 same as ${out_dir}${temp_name}"
    else
        mv -n "$file" "${out_dir}${temp_name}"
    fi
done

for file in "$out_dir"[*"$ext"; do
    timestamp="$(stat -c %y $file | cut -d' ' -f1,2 | sed 's/ /T/' | sed 's/:/./g' | sed -E 's/.{10}$//')"
    final_name="${timestamp} $(basename $file)"
    mv -n "$file" "${out_dir}${final_name}"
done
