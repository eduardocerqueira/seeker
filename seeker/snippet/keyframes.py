#date: 2024-04-26T17:04:17Z
#url: https://api.github.com/gists/3d822b82c41c0e9236742a4422278caf
#owner: https://api.github.com/users/scrpr

import sys, time, pathlib, argparse, time, os
import vapoursynth as vs
from vapoursynth import core

"""
    Dependencies:
        * VapourSynth
        * wwxd [https://github.com/dubhater/vapoursynth-wwxd]
        * vapoursynth-scxvid (If using --use-scxvid) [https://github.com/dubhater/vapoursynth-scxvid]

    ~Add this to your windows path so you can call it from anywhere... :_
"""

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s | %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

parser = argparse.ArgumentParser()
parser.add_argument('--use-scxvid', action='store_true', help="use Scxvid instead of WWXD to detect scene changes")
parser.add_argument('--use-slices', action='store_true', help="when using Scxvid, speeds things up at the cost of differences in scene detection")
parser.add_argument('--sushi', action='store_true', help="sushi compatible (pseudo-XviD 2pass stat file) format")
parser.add_argument('--out-file', help="the file to write scene changes to (Aegisub format); defaults to 'SourceFileName_keyframes.txt' in the same directory as the input video file")
parser.add_argument('clip', help="the input video file")
args = parser.parse_args()

get_name = os.path.splitext(os.path.basename(args.clip)) # defaults to "Source Filename_keyframe.txt"
get_name = get_name[0]+"_keyframes.txt"

if args.out_file:
    get_name = args.out_file

out_path = args.out_file or str(pathlib.Path(args.clip).parent / get_name)
use_scxvid = args.use_scxvid

clip = core.ffms2.Source(source=args.clip)
clip = core.resize.Bilinear(clip, 640, 360, format=vs.YUV420P8)  # speed up the analysis by resizing first
clip = core.scxvid.Scxvid(clip, use_slices=args.use_slices) if use_scxvid else core.wwxd.WWXD(clip)

out_txt = []
if args.sushi:
    out_txt.append("# XviD 2pass stat file\n\n")
else:
    out_txt.append("# keyframe format v1\nfps 0")

print("\nGenerating keyframes : %s\n" %get_name)

start_time = time.time()
for i in range(clip.num_frames):
    props = clip.get_frame(i).props
    scenechange = props._SceneChangePrev if use_scxvid else props.Scenechange
    if args.sushi:
        out_txt.append("i" if scenechange else "b")
    elif scenechange:
        out_txt.append(str(i))
        elapsed_time = (time.time() - start_time) # It's something :/
    progress(i, clip.num_frames, status=" [%d / %d] | %s " %(i,clip.num_frames,time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

out_txt.append("") # trailing newline just in case

with open(out_path, 'w') as f:
    f.write("\n".join(out_txt))

print("\n")
