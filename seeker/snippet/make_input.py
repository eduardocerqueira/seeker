#date: 2023-09-20T16:59:26Z
#url: https://api.github.com/gists/3fac47498cc715cc8162040fe9b70dba
#owner: https://api.github.com/users/hyperlogic

#
# python make_input.py --fps 1 movie01 movie02
#
# Will create an input folder in the curren directory then
# invoke ffmpeg to extract frames from each video and copy
# the resulting frames into that input folder.
#

import argparse
import os
import tempfile

parser = argparse.ArgumentParser(
    prog="make_input",
    description="Produce a sequence of image frames ready for gaussian-splatting",
)
parser.add_argument(
    "-f",
    "--fps",
    default="1",
    help="rate to sample video frames in frames per second",
)

parser.add_argument("filename", nargs="+", help="list of movies to convert to images")
args = parser.parse_args()

print(args)

os.mkdir("input")
input_dir = os.path.abspath("input")
video_filenames = [os.path.abspath(f) for f in args.filename]

print(input_dir)

with tempfile.TemporaryDirectory() as tmpdirname:
    for filename in video_filenames:
        basename = os.path.basename(filename).replace(".", "_")
        cmd = f"ffmpeg -i {filename} -qscale:v 1 -qmin 1 -vf fps={args.fps} {os.path.join(tmpdirname, basename)}%05d.jpg"
        print(cmd)
        os.system(cmd)

    images = []
    for filename in video_filenames:
        basename = os.path.basename(filename).replace(".", "_")
        video_images = []
        for f in os.listdir(tmpdirname):
            if f.startswith(basename) and f.endswith(".jpg"):
                video_images.append(f)
        video_images.sort()
        images.extend(video_images)

    for i, image in enumerate(images):
        basename = "{:05d}.jpg".format(i)
        os.rename(os.path.join(tmpdirname, image), os.path.join(input_dir, basename))
