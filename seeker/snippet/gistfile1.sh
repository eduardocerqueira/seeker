#date: 2023-08-23T17:01:09Z
#url: https://api.github.com/gists/0fc5e1a934c5818fa48112c6f99bde74
#owner: https://api.github.com/users/alex-salnikov

# install cdparanoia e.g apt-get install cdparanoia

# rip tracks individually

cdparanoia -B

# convert to mp3 in a loop

for t in track{01..18}*.wav; do lame $t; done