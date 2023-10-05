#date: 2023-10-05T17:08:01Z
#url: https://api.github.com/gists/62a078e2c219fd1db030c03d47664bc8
#owner: https://api.github.com/users/d-roman-halliday

# Link proxies
# Bug in current Resolve release links clips wrongly to one or two proxies only.
# Don't forget to update proxy_path_root below.

import os, sys
import traceback
import tkinter
import tkinter.messagebox
from tkinter import filedialog

from python_get_resolve import GetResolve
from datetime import datetime

proxy_path_root = "S:\\ProxyMedia"
acceptable_exts = [".mov",".mp4",".mxf",".avi"]


def get_proxy_path():
    root = tkinter.Tk()
    root.withdraw()
    f = filedialog.askdirectory(initialdir = proxy_path_root, title = "Link proxies")
    if f is None:
        print("User cancelled dialog. Exiting.")
        exit(0)
    return f


def filter_videos(dir):
    videos = []

    # Walk directory to match files
    for root, dirs, files in os.walk(dir, topdown=False):
        for name in files:
            file = os.path.join(root, name)
            # print(file)

            # Check extension is allowed
            if os.path.splitext(file)[1].lower() in acceptable_exts:
                videos.append(file)
    return videos




def match_proxies(potential_proxies):

    linked = []

    track_len = timeline.GetTrackCount("video")
    print(f"Video track count: {track_len}")
    for i in range(track_len):
        items = timeline.GetItemListInTrack("video", i)
        if items is None:
            continue

        for potential_proxy in potential_proxies:
            proxy_name = os.path.splitext(os.path.basename(potential_proxy))[0]

            for item in items:
                for ext in acceptable_exts:
                    if ext.lower() in item.GetName().lower():
                        try:
                            media = item.GetMediaPoolItem()
                            name = media.GetName()
                            path = media.GetClipProperty("File Path")

                        except:
                            print(f"Skipping {name}, no linked media pool item.")
                            continue
                        
                        clip_name = os.path.splitext(os.path.basename(path))[0]
                        if proxy_name.lower() in clip_name.lower():
                            if name not in linked:
                                linked.append(name)
                                print(f"Found match: {proxy_name} & {clip_name}")
                                media.LinkProxyMedia(potential_proxy)
            



if __name__ == "__main__":

    try:
        
        # Get global variables
        resolve = GetResolve()
        
        global project
        project = resolve.GetProjectManager().GetCurrentProject()

        global timeline
        timeline = project.GetCurrentTimeline()     

        global media_pool
        media_pool = project.GetMediaPool()

        proxy_dir = get_proxy_path()
        print(f"Passed directory: '{proxy_dir}'\n")

        potential_proxies = filter_videos(proxy_dir)
        print(potential_proxies)
    
        match_proxies(potential_proxies)


    
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        tkinter.messagebox.showinfo("ERROR", tb)
        print("ERROR - " + str(e))