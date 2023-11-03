#date: 2023-11-03T16:48:02Z
#url: https://api.github.com/gists/c0d6d601b83028410d0749382daa34c8
#owner: https://api.github.com/users/lostquix

from pytube import YouTube
from pathlib import Path
from os import *
from pywebio.input import *
from pywebio.output import *

def video_dowload():
    while True:
        video_link = input("Insira o link do video: ")
        if video_link.split("//")[0] == "https:":

            put_text("Download em andamento...".title()).style('color: blue; front-size: 50px')
            video_url = YouTube(video_link)
            video = video_url.streams.get_highest_resolution()
            path_to_dowaload = (r'C:\Users\Dias\Downloads')
            video.download(path_to_dowaload)
            put_text("Video Baixado com sucesso.".title()).style('color: orange; front-size: 50px')
            startfile(r'C:\\Users\\Dias\\Downloads')

if __name__ == "__main__":
    video_dowload()
