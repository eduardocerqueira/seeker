#date: 2022-06-30T21:14:30Z
#url: https://api.github.com/gists/112066a78c103f8610703d44335b9d51
#owner: https://api.github.com/users/hldr4

import sys
from fractions import Fraction

"""
Calculates the SAR required to achieve the desired new DAR

SAR = Desired DAR / Current DAR

eg. (4/3)/(391/284) = 1136/1173

usage: sar.py [current_dar] [desired_dar]
(input as fractions)
"""

current = sys.argv[1].split('/')
wanted = sys.argv[2].split('/')

fs = list(map(Fraction, wanted+current))

sar = Fraction((fs[0]/fs[1])/(fs[2]/fs[3]))

print(f'\nRequired SAR: {sar}')
print(f'\nExample command: ffmpeg -i old.h264 -c copy -bsf:v "h264_metadata=sample_aspect_ratio={sar}" new.h264')
print(f'\n + Tip: to extract the bistream, use something like ffmpeg -i video.mkv -c:v copy -bsf:v h264_mp4toannexb old.h264') 