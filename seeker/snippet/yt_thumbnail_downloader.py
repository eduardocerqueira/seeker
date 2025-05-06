#date: 2025-05-06T16:55:23Z
#url: https://api.github.com/gists/0b3fe14bb4f5cf6818711cfb884cbc6c
#owner: https://api.github.com/users/kelson8

# Copyright 2025 kelson8 - KCNet-ThumbnailDownloader GPLV3
# Basic YouTube thumbnail downloader
# This only took me about 10 minutes to create

import requests
from pathlib import Path

# I got the idea from here: https://www.reddit.com/r/letsplay/comments/2yo1ca/guide_how_do_download_any_thumbnail_from_youtube/

# Saving image:
# https://stackoverflow.com/questions/30229231/python-save-image-from-url

# TODO Setup this to get the user input for the video_id

# The video_id can be changed here, I will add user input for it later.
# Video: I Tried Your Darkest PC Fantasies - Linus Tech Tips
video_id = "t52UW5bXkbs"

# This gets the url of the thumbnail with the video_id
# Such as: https://img.youtube.com/vi/t52UW5bXkbs/maxresdefault.jpg
thumbnail_url = "https://img.youtube.com/vi/" + video_id + "/maxresdefault.jpg"

# Download the thumbnail from the url.
# Will give errors on 404 and 403 and cancel the download if it fails.
def download_thumbnail(url):
    file_name = video_id + ".jpg"
    file_path = Path(file_name)

    # https://stackoverflow.com/questions/82831/how-do-i-check-whether-a-file-exists-without-exceptions
    # Check if the file exists, if so do nothing.
    if file_path.exists():
        print("File already exists!")
        return
    else:
        img_url = requests.get(url)

        # Check if the file is on the server, do nothing if there is an error.
        # https://stackoverflow.com/questions/15258728/requests-how-to-tell-if-youre-getting-a-404
        if img_url.status_code == 404:
            print("File not found!")
            return
        elif img_url.status_code == 403:
            print("Access forbidden to resource!")
            return

        img_data = requests.get(url).content

        with open(file_name, 'wb') as handler:
            handler.write(img_data)

# I haven't completed this part of the command yet
def video_id_input():
    # This only takes the video id, not the url.
    video_id_test = input("Enter the video id to download the thumbnail: ")

def print_link():
    print(thumbnail_url)

def main():
    download_thumbnail(thumbnail_url)

# Debug to print the url
# print_link()

if __name__ == '__main__':
    main()