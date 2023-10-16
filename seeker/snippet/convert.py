#date: 2023-10-16T17:06:42Z
#url: https://api.github.com/gists/218c9011d9c78b914fdff738c12b08d3
#owner: https://api.github.com/users/unbelauscht

#!/usr/bin/env python3

from PIL import Image

from os import stat
from glob import glob
import subprocess

try:
    subprocess.check_output('rm *.gs.png temp_no_ocr.pdf output.pdf', shell=True, executable="/bin/bash")
except:
    pass

files = sorted(glob('*.png'))

for idx, file in enumerate(files):
    print(f"Processing {idx+1} of {len(files)}")
    filename_parts = file.split('.')

    with Image.open(file) as img:
        img = img.convert("L", dither=None)  # to grayscale
        img = img.point(lambda x: 0 if x < 128 else 255, '1')

    img.save(f"{filename_parts[0]}.gs.png")

files = sorted(glob('*.gs.png'))

subprocess.check_output('convert ' + ' '.join(files) + ' temp_no_ocr.pdf', shell=True, executable="/bin/bash")

subprocess.check_call(['ocrmypdf', 'temp_no_ocr.pdf', 'output.pdf'])

try:
    subprocess.check_output('rm *.gs.png temp_no_ocr.pdf', shell=True, executable="/bin/bash")
except:
    pass

size = round(
        stat('output.pdf').st_size / pow(1024, 2),
        2
    )

print(f"Filesize: {size}MB")