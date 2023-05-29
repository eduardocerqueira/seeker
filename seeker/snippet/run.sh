#date: 2023-05-29T17:00:36Z
#url: https://api.github.com/gists/bee56dfd4a10e0e20580a38193e957bf
#owner: https://api.github.com/users/dpaluy

bundle add dockerfile-rails
bin/rails g dockerfile
brew add flyctl
flyctl launch
fly open