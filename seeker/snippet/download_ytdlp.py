#date: 2025-01-29T16:58:21Z
#url: https://api.github.com/gists/10b356d38767561a9d5d86fbfe0f029e
#owner: https://api.github.com/users/Trevahok

from swiftshadow.classes import Proxy
from pytubefix import YouTube
import os.path as osp
import pytubefix
import os

swift = Proxy(autoRotate=True)
print(swift.proxy())


save_dir = './test'


video_ids = ['8qIl-0XOguM']
import time
import random

done = set()

for i in video_ids: 
    try:
       
        url = osp.join(f"https://www.youtube.com/watch?v={i}")
        if url in done: 
            continue
        print(url)

        ip, protocol = swift.proxy()
        yt = pytubefix.YouTube(url, proxies={ protocol: ip} )
        # yt = "**********"=True )
    except Exception as e :
        print(f"Connection Error: {i}")

  
    try: 
        yt.streams\
        .filter(progressive=True, file_extension="mp4")\
        .order_by("resolution")\
        .desc()\
        .first()\
        .download(osp.join(save_dir, f"{i}.mp4"))

        done.add(url)
        time.sleep(random.randint(1, 10))
    except Exception as e : 
        print('ERROR downloading: ')
        print(e)
        if 'BotDetection' in str(type(e)): 
            time.sleep(6)

         time.sleep(6)

