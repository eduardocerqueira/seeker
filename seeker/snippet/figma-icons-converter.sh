#date: 2023-08-24T16:32:24Z
#url: https://api.github.com/gists/9b86358f2525cc3d0a1113ca2b9ffc03
#owner: https://api.github.com/users/Git-I985

in="./e1"
out="./export"

rm -rf ${out}
mkdir -p ${out}

for file in ${in}/*; do
  dirname=$(basename $file '.svg' | tr '+' '_');
  dest=$out/$dirname/default.svg
  echo $dest
  mkdir $(dirname $dest)
  cp $file $dest
done


