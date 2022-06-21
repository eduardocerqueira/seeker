#date: 2022-06-21T17:16:58Z
#url: https://api.github.com/gists/296c2c2010d11ba4254f54f89a6cc11d
#owner: https://api.github.com/users/bwees

from defisheye import Defisheye
import cv2 
import subprocess
from tqdm import tqdm
import sys
import threading

dtype = 'linear'
format = 'fullframe'
fov = 180
pfov = 120

threads = 4

frame_count = int(cv2.VideoCapture(sys.argv[1]).get(7))
frame_ids = range(1,frame_count+1)

# delete contents of process folder
subprocess.call(['rm','-rf','process'])
subprocess.call(['mkdir','process'])

with tqdm(total=len(frame_ids)) as pbar:

    # run 4 threads that will do the defisheye
    def run_thread(i):
        cap = cv2.VideoCapture(sys.argv[1])
        my_ids = frame_ids[i::threads]
        for id in my_ids:
            cap.set(1,id)
            succ, img = cap.read()

            if not succ:
                break

            defi = Defisheye(img,dtype=dtype,format=format,fov=fov,pfov = pfov)
            defi.convert('process/outframe_%d.png' % id)
            pbar.update(1)
            
        
    # create threads
    thread_objs = []
    for i in range(threads):
        t = threading.Thread(target=run_thread,args=(i,))
        thread_objs.append(t)

    # start threads and wait for completion
    for t in thread_objs:
        t.start()

    # wait for threads to finish
    for t in thread_objs:
        t.join()

subprocess.call(['ffmpeg', '-framerate', '30', '-i', 'process/outframe_%d.png', '-i', sys.argv[1], '-c:a', 'aac', '-c:v', 'libx264', '-profile:v', 'high', '-crf', '20', '-pix_fmt', 'yuv420p', '-map', '0:v:0', '-map', '1:a:0', '-shortest', '-y', sys.argv[2]])
subprocess.call(['rm','-rf','process'])
