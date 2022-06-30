#date: 2022-06-30T21:13:52Z
#url: https://api.github.com/gists/ec939188eb5dd2c01f58cfbe1be924f4
#owner: https://api.github.com/users/alisterk

#!/bin/bash
workdir=$(pwd)
filepath=$1
format=$2
outdir=$3
book_convert(){
    filepath=$1
    format=$2
    tmp_dir="$HOME/Downloads/eCoreCmdtmp/$(uuidgen)"
    mkdir -p $tmp_dir
    # rm -rf "$HOME/Downloads/eCoreCmdtmp"
    dirname=$(dirname "$filepath")
    input_file_type=$(echo ".${filepath##*.}")
    basename=$(basename -s $input_file_type "$filepath")
    origin_name=$(basename "$filepath")
    newfile_neme=$basename"."$format
    echo "- Convert $basename to $format"
    echo "$workdir/eCoreCmd c $dirname/$origin_name $outdir/$newfile_neme $tmp_dir"
    $workdir/eCoreCmd c "$dirname/$origin_name" "$outdir/$newfile_neme" "$tmp_dir"
    rm -rf $tmp_dir
    echo "- Done! $origin_name , output file: $newfile_neme"
}

book_convert "$1" $2
