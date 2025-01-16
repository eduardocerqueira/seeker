#date: 2025-01-16T17:05:39Z
#url: https://api.github.com/gists/249e54197c33eabf1e3c31906a9cf897
#owner: https://api.github.com/users/jeffreygarc

import json

# load the json data
with open('data.json', 'r') as file:
    data = json.load(file)

# extract the urls for "Favorite VideoList"
saved_links = [video["Link"] for video in data.get("Activity", {}).get("Favorite Videos", {}).get("FavoriteVideoList", [])]

# extract urls from "ItemFavoriteList"
liked_links = [item["link"] for item in data.get("Activity", {}).get("Like List", {}).get("ItemFavoriteList", [])]

# count all urls
all_links = saved_links + liked_links

# save the links
with open('saved.txt', 'w') as saved_file:
    for link in saved_links:
        saved_file.write(link + '\n')

with open('likes.txt', 'w') as likes_file:
    for link in liked_links:
        likes_file.write(link + '\n')

# print the total urls
total_urls = len(all_links)
print(f"Total number of URLs found: {total_urls}")
