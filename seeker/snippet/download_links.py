#date: 2024-06-24T16:42:11Z
#url: https://api.github.com/gists/ddd1e9c6e33ab1f42d3357623a91e6bc
#owner: https://api.github.com/users/burak-yildizoz

import os
from pytube import YouTube
import subprocess
import sys

# https://stackoverflow.com/a/9130405
os.chdir(sys.path[0])

# https://stackoverflow.com/questions/78160027/how-to-solve-http-error-400-bad-request-in-pytube
# https://github.com/pytube/pytube/issues/1894
from pytube.innertube import _default_clients
_default_clients["ANDROID"]["context"]["client"]["clientVersion"] = "19.08.35"
_default_clients["ANDROID_MUSIC"] = _default_clients["ANDROID"]

with open('links.txt') as f:
    links = f.readlines()

for link in links:
    link = link.strip()
    print('Link:', link)
    # https://stackoverflow.com/questions/76129007/pytube-keyerror-streamdata-while-downloading-a-video
    yt = YouTube(link, use_oauth=True)
    # video = yt.streams.get_highest_resolution()

    name = yt.title
    print('Name:', name)
    # https://stackoverflow.com/a/3939381
    name = name.translate(str.maketrans('', '', '\/:*?"<>|'))

    # https://stackoverflow.com/a/63269663
    name_mp4 = name + '.mp4'
    audio = yt.streams.get_audio_only()
    audio.download()
    name_mp3 = name + '.mp3'
    ffmpeg = 'ffmpeg -i "%s" "%s"' % (name_mp4, name_mp3)

    print(ffmpeg)
    res = subprocess.run(ffmpeg, shell=True)
    if res.returncode == 0:
        os.remove(name_mp4)
    print('--------------------------------------------------------------------------------')
