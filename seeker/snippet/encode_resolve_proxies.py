#date: 2023-10-05T17:08:01Z
#url: https://api.github.com/gists/62a078e2c219fd1db030c03d47664bc8
#owner: https://api.github.com/users/d-roman-halliday

# Windows: Create a shortcut to a .bat file in 'shell:sendto' to call this script with files passed as arguments
# Use with 'save_proxy_clip_list.py' gist to quickly pull used timeline clips into FFMPEG.
# Use 'link_proxies.py' gist to relink proxies correctly. 
# Bug in current Resolve release links clips wrongly to one or two proxies only.
# This assumes FFMPEG is on path.

import os, sys, shutil
import subprocess
import argparse
import pathlib
import winsound
from winsound import Beep

from datetime import datetime

ap = argparse.ArgumentParser(description='Watchfolder or manually queued Resolve proxy encodes',
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

ap.add_argument('path', nargs='+',
                help='Path of a file or a folder of files.')
ap.add_argument("-d", "--dryrun", required=False,
                help="use to skip encoding for debug purposes")
ap.add_argument("-y", "--yes", required=False,
                help="assume yes at any prompts")
args = ap.parse_args()

# Globals:
###################################################

# Vars
acceptable_exts = ['.mp4', '.mov', '.mxf']
encodable = []
skipped = []
cwd = os.getcwd()
proxy_path_root = "S:\\ProxyMedia"

###################################################


# Logging path
if not os.path.exists(proxy_path_root):
    raise Exception(f"{proxy_path_root} does not exist. Please check write path.")
log_path = os.path.join(proxy_path_root, "ProxyEncoder.log")

def confirm(message):
    answer = input(message + "\n")
    print("\n")
    if "y" in answer.lower():
        return True
    elif "n" in answer.lower():
        return False
    else:
        print(f"Invalid response, '{answer}'. Please answer 'yes' or 'no'")
        confirm(message)


def print_and_log(message, log_only=False, add_time=True):
    if not log_only:
        print(message)
    
    with open(log_path, "a+") as changelog:
        changelog.write(f"[{datetime.now()}] {message} \n")

def get_vids(filepaths):
    try:
        for path in filepaths:
            print_and_log(f"Checking path: '{path}'", log_only=True)
            if os.path.isfile(path):
                if os.path.splitext(path)[1].casefold() in acceptable_exts:
                    print_and_log(f"Queued {path}")
                    
                    # Get folder structure
                    p = pathlib.Path(path)
                    output_dir = os.path.join(proxy_path_root, os.path.dirname(p.relative_to(*p.parts[:1])))
                    # Make folder structure
                    os.makedirs(output_dir, exist_ok=True)
                    # Return full output file path
                    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(path))[0])
                    encodable.append({"source_media":path, "output_file":output_file})
                else:
                    print_and_log(f"Skipping {path}") # todo: Directory scanning not working yet. Only works on files. "Too many values to unpack"
                    skipped.append(path)

            elif os.path.isdir(path):
                print(f"Scanning directory: {path}")
                for root, files in os.walk(path, topdown=True):
                    for file in files:
                        get_vids(os.path.join(root, file))
                        

    except Exception as e:
        print_and_log(e)

    return skipped, encodable

def parse_list(file):
    
    f = open(file, 'r')
    lines = f.readlines()
    lines = [i.strip() for i in lines]
    return lines

def encode(src_path, output_path):
    try:
        print("\n")
        filename = os.path.basename(src_path)
        print_and_log(f"Encoding '{filename}'")

        #-pix_fmt yuv422p10le -c:v prores_ks -profile:v 3 -vendor ap10 -vf scale=1280:720,fps=50/1
        subprocess.run((f"ffmpeg.exe -y -i \"{src_path}\" -c:v dnxhd -profile:v dnxhr_sq -vf scale=1280:720,fps=50/1,format=yuv422p -c:a pcm_s16le -ar 48000 -v warning -stats -hide_banner \"{output_path}.mxf\""),
                        shell=True, stdout=subprocess.PIPE)
        
        winsound.Beep(600, 200) # Success beep
        return True

    except Exception as e:
        print_and_log("\n\n----------------------------", add_time=False)
        print_and_log(e)
        print_and_log("----------------------------\n\n", add_time=False)
        print_and_log(f"Failed encoding: {src_path}")
        winsound.Beep(375, 150) # Fail beep








if __name__ == "__main__":

    new_encode = f"################# {datetime.now().strftime('%A, %d, %B, %y, %I:%M %p')} #################"
    print_and_log(new_encode, log_only=True, add_time=False)

    filepaths = args.path

    for file in filepaths:
        if ".txt" in os.path.splitext(file)[1]:
            print(f"Parsing list from file '{file}'\n")
            txt_file_paths = parse_list(file)
            print(txt_file_paths)
            filepaths.remove(file) # Remove the text file for processing 


    # Get encodable files from text clip list
    skipped, encodable_from_txt = get_vids(txt_file_paths)

    # Get any dirs, files passed for processing
    skipped, encodable_loose = get_vids(filepaths)

    # Combine encode lists
    encodable = encodable_from_txt + encodable_loose
    print(encodable)

    for video in encodable:
        print_and_log(f"Queued {video['source_media']}")


    # Confirm encode
    if not args.yes:
        if not confirm("Encode the above files?"):
            print("Aborting encode.")
            sys.exit(1)

    # Encode loose files
    for file in encodable:
        print(type(file))
        if not args.dryrun:
            if encode(file['source_media'], file['output_file']):
                print_and_log(f"Successfully encoded: {file}")

    
    print(f"Done encoding. Check log file: '{log_path}'")

    # Finished jingle
    for i in range(1, 10):
        winsound.Beep(i * 100, 200)
    