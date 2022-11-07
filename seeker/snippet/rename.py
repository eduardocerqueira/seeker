#date: 2022-11-07T17:09:44Z
#url: https://api.github.com/gists/a6afb0c584769515b94a0146fad59573
#owner: https://api.github.com/users/funkey7dan

import os
import argparse
import re

# main
parser = argparse.ArgumentParser()
parser.add_argument('--path','-p', type=str,
                    help='the working path')
args = vars(parser.parse_args())
if args['path'] is None:
    root = input("No path given, please provide a path!\n")
else:
    root = os.path.abspath(args['path'])
if not os.path.isdir(root):
    print("Error! the passed path is not a folder.")
    exit(-1)
    
for path, subdirs, files in os.walk(root):
    for folder in subdirs:
        r = re.compile('.*\.(mp4|mkv|avi)')
        temp = list(filter(r.match,files)) #list of video files in folder
        if temp != [] : change_to = os.path.splitext(temp[0])[0] #if there is a movie, get its file name
        if re.search('sub',folder,re.IGNORECASE): # check if the subfolder we are in the subtitle folder
            contents = os.listdir(os.path.join(path,folder)) 
            for i in range(0,len(contents)): # check the contents of the subs folder
                if contents[i].endswith(".srt"):
                    prev_name = os.path.splitext(contents[i])[0] # check if the name needs changing
                    if change_to not in prev_name :
                        os.rename(os.path.join(path,folder,contents[i]),os.path.join(path,folder,change_to+"."+prev_name+".srt"))