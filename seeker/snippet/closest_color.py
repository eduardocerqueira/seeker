#date: 2025-07-29T16:55:08Z
#url: https://api.github.com/gists/9cb880dc0c9198aa239ea8865553bc95
#owner: https://api.github.com/users/maikgreubel

import webcolors
import sys
import urllib.request
import os

# Make use of xkcd 954 color name mappings in one name to rgb file (thank you dudes!)
if not os.path.exists(os.path.join(os.path.realpath(__file__), "rgb.txt")):
    urllib.request.urlretrieve("https://xkcd.com/color/rgb.txt", "rgb.txt")

def closest_color(requested_color:str):
    """
    Return the three most matching color names for given color
    """
    xkcd_colors = {}    

    with open("rgb.txt") as f:
        next(f)
        for line in f:
            name, color = line.split("#")
            xkcd_colors[name.strip()] = "#" + color.strip()

    if not requested_color.startswith('#'):
        requested_color = "#" + requested_color
    
    min_distance = float("inf")
    closest_names = []

    requested_rgb = webcolors.hex_to_rgb(requested_color)
    for name in webcolors.names():
        rgb = webcolors.name_to_rgb(name)

        distance = (
            (rgb.red - requested_rgb.red) ** 2 +
            (rgb.green - requested_rgb.green) ** 2 +
            (rgb.blue - requested_rgb.blue) ** 2
        )

        if distance < min_distance:
            min_distance = distance
            closest_names.insert(0, name)
            if len(closest_names) > 3:
                closest_names.pop()

    for name, color in xkcd_colors.items():
        rgb = webcolors.hex_to_rgb(color)
        distance = (
            (rgb.red - requested_rgb.red) ** 2 +
            (rgb.green - requested_rgb.green) ** 2 +
            (rgb.blue - requested_rgb.blue) ** 2
        )

        if distance < min_distance:
            min_distance = distance
            closest_names.insert(0, name)
            if len(closest_names) > 3:
                closest_names.pop()

    return closest_names

if __name__ == '__main__':
    print("Closest color name is ", closest_color(sys.argv[1]))