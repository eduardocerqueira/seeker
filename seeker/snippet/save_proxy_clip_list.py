#date: 2023-10-05T17:08:01Z
#url: https://api.github.com/gists/62a078e2c219fd1db030c03d47664bc8
#owner: https://api.github.com/users/d-roman-halliday


# Save proxy clip list
# Currently no way exists to create proxies of clips used in active timeline.
# Can't right click or create smart bin to filter by usage and timeline.
# This creates a text file of clips from active timeline that encode_resolve_proxies can read. 
# May change it to csv/json/yml someday.
# Don't forget to update clip_list_path

import os, sys
import traceback
import tkinter
import tkinter.messagebox
from tkinter import filedialog

from python_get_resolve import GetResolve
from datetime import datetime

clip_list_path = "S:\\ProxyMedia\\ClipLists"


def save_clip_list(clips):
    
    # initial_dir = os.path.join(clip_list_path, project.GetName())
    os.makedirs(clip_list_path, exist_ok=True)
    
    root = tkinter.Tk()
    root.withdraw()
    f = filedialog.asksaveasfile(initialdir = clip_list_path, initialfile = f"{project.GetName()}_{timeline.GetName()}_cliplist.txt", title="Generate clip list", 
                                        filetypes = (("txt file", "*.txt"), ("all files", "*.*")))
    if f is None:
        exit(0)

    # write project name as header
    # f.write(f"{str(project.GetName())}\n") 

    # write clip list
    for clip in clips:
        f.write(f"{str(clip)}\n")

    f.close()


def get_media_paths():
    
    acceptable_exts = [".mov",".mp4",".mxf",".avi"]
    media_paths = []

    track_len = timeline.GetTrackCount("video")
    print(f"Video track count: {track_len}")
    for i in range(track_len):
        items = timeline.GetItemListInTrack("video", i)
        if items is None:
            continue

        for item in items:
            for ext in acceptable_exts:
                if ext.lower() in item.GetName().lower():
                    try:
                        media = item.GetMediaPoolItem()
                        path = media.GetClipProperty("File Path")
                    except:
                        print(f"Skipping {item.GetName()}, no linked media pool item.")    
                        continue
                    
                    if path not in media_paths:
                        media_paths.append(path)
                        media_paths = list(dict.fromkeys(media_paths)) 
                        
                # else:
                #     print(f"Skipping {item.GetName()}, not of type {ext}")
       
    return media_paths
    


if __name__ == "__main__":

    try:
        
        # Get global variables
        resolve = GetResolve()
        
        global project
        project = resolve.GetProjectManager().GetCurrentProject()

        global timeline
        timeline = project.GetCurrentTimeline()     

        clips = get_media_paths()
        save_clip_list(clips)

    
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)

        root = tkinter.Tk()
        root.withdraw()
        tkinter.messagebox.showinfo("ERROR", tb)
        print("ERROR - " + str(e))