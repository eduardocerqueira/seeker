#date: 2022-07-26T16:46:22Z
#url: https://api.github.com/gists/49eac209ffdcbfea3f4907fe53e085eb
#owner: https://api.github.com/users/austinyen56

"""

Made by @austinyen56
Allows you to download music from most streaming services (YT, SoundCloud... etc) in multiple formats(mp3, wav, flac)
Supports thumbnail embedding for mp3 option only

"""
import os
url = str(input("Enter URL: "))
# Can change --audio-format to wav if needed
fmt = str(input("In what format (flac/wav/mp3[support thumbnail embedding]): "))

cmd = "youtube-dl -i -x --audio-format "+ fmt +" --embed-thumbnail --postprocessor-args \"-write_id3v1 1 -id3v2_version 3\" --audio-quality 0 -o C:/Users/Austin/Desktop/YoutubeDL_Downloads/%(title)s.%(ext)s"
#cmd = "youtube-dl -i -x --audio-format "+ fmt +" --embed-thumbnail --postprocessor-args \"-write_id3v1 1 -id3v2_version 3\" --audio-quality 0 -o C:/Users/Austin/Desktop/ref/%(title)s.%(ext)s"
cmdurl = cmd + " " + url
print(cmdurl)
os.system(cmdurl)
input("Successfully downloaded, hit ENTER to exit... ")