#date: 2022-03-01T16:48:04Z
#url: https://api.github.com/gists/9e9e3dd483ae07dee7938feff593d5a8
#owner: https://api.github.com/users/johnrobertmcc

function ctwp() {
  extension=$1
  params=$2
      	
  for i in *.$extension
   	do
   	  file=$i
   	  convert $file $params ${file%.*}.webp
   	done
}
