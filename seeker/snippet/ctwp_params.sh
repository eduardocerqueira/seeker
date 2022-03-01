#date: 2022-03-01T16:57:25Z
#url: https://api.github.com/gists/6d8bcebfa1c1692f265ff79365376a72
#owner: https://api.github.com/users/johnrobertmcc

function ctwp() {
      extension=$1
      params=$2 # Add this
      for i in *.$extension
      do
        file=$i
        convert $file $params ${file%.*}.webp # Add params here
      done
}
