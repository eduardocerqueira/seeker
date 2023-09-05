#date: 2023-09-05T17:08:00Z
#url: https://api.github.com/gists/32cab88e6a865b5842957c84e7b6ad7e
#owner: https://api.github.com/users/Winterhuman

#!/bin/sh

# Requires pngquant, gifsicle, gif2apng, and optionally exiftool.

pquit() { printf "\033[1m\033[31m%b\033[0;39m" "$1"; exit 1; }
pstat() { printf "\033[1m\033[34m%b\033[0;39m\033[1m%b\033[0;39m\n" "$1" "$2"; }
clean() { if ! rm -r "$tmp"; then pquit "Failed to delete '$tmp'!\n"; fi }

trap clean EXIT
tmp="$(mktemp -d)"


# Input

if [ -z "$1" ]; then
	pquit "No arguments given!\n"; fi

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
	pstat "\tArgument 1: " "/path/to/input{.png,.gif}" 
	pstat "\tArgument 2: " "/path/to/output{.png,.apng,.gif}"
	printf "\t\t\033[1mNote:\033[0;39m The output's extension is replaced with the ideal format's extension.\n"
	exit 0
fi

if [ ! -f "$1" ]; then
	pquit "'$1' doesn't exist, or isn't a file!\n"; fi

input="$1"


# Mimetype detection

find_mime() {
	if ! file -ib "$1"; then
		pquit "Couldn't determine mimetype of '$1'!\n"; fi
}

mime="$(find_mime "$input")"
mime="${mime%;*}"
pstat "Input: " "$input ($mime)"


# Output

if [ -z "$2" ]; then
	pquit "No output given!\n"; fi

for output_exists in "$2"*; do
	if [ "${output_exists%.*}" = "${2%.*}" ]; then
		pquit "'$output_exists' shares a filename with '$2'!\n"; fi
done

given_output="${2%.*}"
output_filename="${given_output##*/}"

pstat "Output template: " "${given_output}.???"


# In case of PNG

input_is_png() {
	final_output="${given_output}.png"
	if ! pngquant --quality 100 --speed 1 --strip "$input" --output "$final_output"; then
		pquit "'pngquant' failed!\n"; fi
}


# In case of GIF

togif() {
	tmp_gifsicle_output="${tmp}/${output_filename}-gifsicle"
	if ! gifsicle --optimize="$1" --optimize=keep-empty "$input" -o "${tmp_gifsicle_output}${1}.gif"; then
		pquit "'gifsicle' failed!\n"; fi
}

toapng() {
	# GIF2APNG can't handle absolute paths, so we convert them to relative paths.
	# Source: https://sourceforge.net/p/gif2apng/discussion/1022150/thread/8ec5e7e288
	tmp_apng_input="$(realpath --relative-to="$PWD" "$input")"
	tmp_apng_output="$(realpath --relative-to="$PWD" "${tmp}/${output_filename}.apng")"
	if ! gif2apng -z2 -i10 "$tmp_apng_input" "$tmp_apng_output" > /dev/null 2>&1; then
		pquit "'gif2apng' failed!\n"; fi
}

input_is_gif() {
	opt_level="1"
	while [ "$opt_level" -le "3" ]; do
		togif "$opt_level" &
		opt_level="$(( "$opt_level" + 1 ))"
	done

	toapng &

	wait

	for tmp_file in "$tmp"/*; do
		size="$(wc -c < "$tmp_file")"
		sizepathlist="${size} TRUNCATETOHERE${tmp_file}\n$sizepathlist"
	done

	smallest="$(printf "%b" "$sizepathlist" | sort -n | head -n1)"
	smallest="${smallest#*TRUNCATETOHERE}"
	final_output="${given_output}.${smallest##*.}"

	if ! mv "$smallest" "$final_output"; then
		pquit "Failed to move $smallest to $final_output!\n"; fi
}


# File selection (functions have to be defined first)

case "$mime" in
	"image/png") input_is_png;;
	"image/gif") input_is_gif;;
	*) pquit "Mimetype of '$input' is neither PNG nor GIF!\n";;
esac

pstat "Final output: " "$final_output"

if ! exiftool -overwrite_original_in_place -all= "$final_output" >/dev/null 2>&1; then
		printf "Failed to remove EXIF metadata for '%s'!\n" "$final_output"
	else
		pstat "Size diff: " "$(wc -c < "$input") bytes -> $(wc -c < "$final_output") bytes"
fi