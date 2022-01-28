#date: 2022-01-28T16:56:32Z
#url: https://api.github.com/gists/c73a75f8cbc8613b50c3b4bfc8cad2e4
#owner: https://api.github.com/users/aaronpenne

# Combine multiple images into one (three singles into one row of three)
ffmpeg -i IMG_0021.jpeg -i IMG_0029.jpeg -i IMG_0026.jpeg -filter_complex "[0][2][1]hstack=inputs=3" combined.png

# Expand image by adding colored padding
convert unpadded.png -background "#ECE7DD" -gravity center -extent 7200x4000 padded.png