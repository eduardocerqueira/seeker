#date: 2021-11-30T17:12:50Z
#url: https://api.github.com/gists/ab83fa66c36d27f6f87e2b711241c73c
#owner: https://api.github.com/users/elydev01

import time, os
from pydub import AudioSegment
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


VIDEO_EXTENSIONS = ["MP4","MOV","WMV","AVI","AVCHD","FLV","F4V","SWF","MKV","WEBM"]
AUDIO_EXTENSIONS = ["MP3","M4A","WAV","WMA","AAC","FLAC"]


files_path = input('Indicate source file: ')
file_path, file = os.path.split(files_path)
file_name, _extension = os.path.splitext(file)
extension = _extension.lstrip(".")

duration = int(input("Duration (in seconds): "))

startTime = 0

targetname = f'{file_name}-extract-{time.strftime("%d%Y%m%M%H%S")}.{extension}'


if extension.upper() in AUDIO_EXTENSIONS:
    endTime = duration * 1000
    song = AudioSegment.from_file(files_path)
    extract = song[startTime:endTime]

    if extension.lower() == "m4a":
        extension = "ipod"
    extract.export(targetname, format=extension.lower())
else:
    endTime = duration / 60
    ffmpeg_extract_subclip(files_path, startTime, endTime, targetname=targetname)
