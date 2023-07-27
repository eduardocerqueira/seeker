#date: 2023-07-27T16:30:20Z
#url: https://api.github.com/gists/c075d0b681307e356482d77621d86866
#owner: https://api.github.com/users/s3rgeym

for filename in *.tar.bz2; do
  target=~/workspace/data/breaches/yandex/"$(basename $filename .tar.bz2)"
  mkdir -p "$target"
  tar xf "$filename" -C "$target"
done