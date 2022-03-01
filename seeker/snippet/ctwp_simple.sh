#date: 2022-03-01T16:56:39Z
#url: https://api.github.com/gists/fbb6d4e2cba7803eaf629013e1dfe84c
#owner: https://api.github.com/users/johnrobertmcc

function ctwp() {
      extension=$1 # take the extension you provided the function
      for i in *.$extension # iterate through the files in the directory based on the extension
      do
        file=$i # declare the file as an easy-to-read variable
        convert $file ${file%.*}.webp 
# The above line executes the ImageMagick command, removes the basename from the file to grab only the file name (so ‘bull.jpg’ becomes ‘bull’)
      done
}
