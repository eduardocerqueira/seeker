#date: 2024-04-23T16:55:04Z
#url: https://api.github.com/gists/1abb334d64536e038dfa40b50f475832
#owner: https://api.github.com/users/SahilMahadwar

#!/bin/bash

# Fetch wallpaper from the API - https://github.com/TimothyYe/bing-wallpaper
api_url="https://bing.biturl.top/?resolution=3840&format=json&index=0&mkt=zh-CN"
response=$(curl -s "$api_url")

# Parse JSON response
start_date=$(echo "$response" | jq -r '.start_date')
end_date=$(echo "$response" | jq -r '.end_date')
image_url=$(echo "$response" | jq -r '.url')
copyright=$(echo "$response" | jq -r '.copyright')
copyright_link=$(echo "$response" | jq -r '.copyright_link')

# Directory to store images and config file
home_dir="$HOME"
relative_path="/Pictures/bing-wallpapers"
image_dir="${home_dir}${relative_path}"

# Config file
config_file="$image_dir/bing-wallpaper-config.json"

# Image filename
image_filename="$start_date"_"$end_date.jpg"

# Creating directory if not exists
mkdir -p "$image_dir"

# Check if image already exists in config file
if grep -q "$image_url" "$config_file"; then
    echo "Wallpaper already exists. Skipping download."
    
    # Apply wallpaper   
    # Dark mode
    gsettings set org.gnome.desktop.background picture-uri-dark file://$image_dir/$image_filename
    # Light mode
    gsettings set org.gnome.desktop.background picture-uri file://$image_dir/$image_filename

else
    # Downloading the image
    curl -s -o "$image_dir/$image_filename" "$image_url"

    echo "New bing wallpaper downloaded."

    # Updating config file
    if [ -f "$config_file" ]; then
        # If config file exists, append new entry to the "wallpapers" array
        jq --arg image_url "$image_url" --arg start_date "$start_date" --arg end_date "$end_date" \
            --arg copyright "$copyright" --arg copyright_link "$copyright_link" \
            '.wallpapers += [{url: $image_url, start_date: $start_date, end_date: $end_date, copyright: $copyright, copyright_link: $copyright_link}]' "$config_file" > "$config_file.tmp" && \
            mv "$config_file.tmp" "$config_file"
    else
        # If config file doesn't exist, create new with the entry
        echo "{\"wallpapers\": [{\"url\": \"$image_url\", \"start_date\": \"$start_date\", \"end_date\": \"$end_date\", \"copyright\": \"$copyright\", \"copyright_link\": \"$copyright_link\"}]}" \
            > "$config_file"
    fi

    # Apply wallpaper   
    # Dark mode
    gsettings set org.gnome.desktop.background picture-uri-dark file://$image_dir/$image_filename
    # Light mode
    gsettings set org.gnome.desktop.background picture-uri file://$image_dir/$image_filename
    
fi
