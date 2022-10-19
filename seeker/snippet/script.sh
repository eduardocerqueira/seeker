#date: 2022-10-19T17:30:48Z
#url: https://api.github.com/gists/ef061d458f638397bcba05d8ed7757ed
#owner: https://api.github.com/users/nxjosephofficial

#!/bin/bash

# Learn today's date.
date=$(date "+%Y-%m-%d")
# Example: 2022-10-19

# Put a dash for filename.
date="$date-"

# Get the filename.
echo "Enter filename: "
read -r filename

# Create the file.
file="$date""$filename"
posts_path="_posts/"
touch "$posts_path""$file"
file="$posts_path""$file"

# Add front matter to created post.
echo "---" > "$file"
echo "layout: post" >> "$file"
echo "Enter the post's title: "
read -r title
echo "title: \"""$title""\"" >> "$file"
echo "---" >> "$file"
echo "" >> "$file"