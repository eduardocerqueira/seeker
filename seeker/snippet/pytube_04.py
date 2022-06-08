#date: 2022-06-08T16:47:02Z
#url: https://api.github.com/gists/990cc7a0f8b34c9f8ff3d3b251d08d19
#owner: https://api.github.com/users/SoftSAR

from pytube import YouTube
from multiprocessing import Pool

def savevid(url_video):
    try:
        YouTube(url_video).streams.first().download()
    except:
        print(f"ссылка {url_video} не валидна")
        
if __name__ == '__main__':   
    url_videos = input("Укажите ссылки на видео для скачивания через запятую.").split(",")
    processes_count = 3 #Количество запускаемых процессов
    processes = Pool(processes=processes_count)        
    processes.map(savevid, url_videos)