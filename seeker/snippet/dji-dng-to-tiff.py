#date: 2022-10-19T17:35:00Z
#url: https://api.github.com/gists/c9be9be7c7ccf9de07bad44c890b10c9
#owner: https://api.github.com/users/weaming

#!/usr/bin/env python3
# For M1:
#!/usr/bin/arch -x86_64 /Library/Frameworks/Python.framework/Versions/3.10/bin/python3

"""
Batch convert DNG to TIFF.

Install dependencies:
python3 -m pip install numpy cython imageio

For M1:
1. download & install "macOS 64-bit universal2 installer" via https://www.python.org/downloads/release/python-3108/
2. use the universal2 version python to install dependencies:
    /usr/bin/arch -x86_64 /Library/Frameworks/Python.framework/Versions/3.10/bin/python3 -m pip install numpy cython imageio
    /usr/bin/arch -x86_64 /Library/Frameworks/Python.framework/Versions/3.10/bin/python3 -m pip install rawpy
"""

import os
import rawpy
import imageio

for infile in os.listdir("./"):
    if infile[-3:] == "DNG":
        outfile = infile[:-3] + "tiff"
        raw = rawpy.imread(infile)
        print(infile, '==>', outfile)

        # https://letmaik.github.io/rawpy/api/rawpy.Params.html#rawpy.Params
        rgb = raw.postprocess(
            use_camera_wb=True,
            half_size=False,
            no_auto_bright=False,
            output_bps=16,
            output_color=rawpy.ColorSpace.ProPhoto,
        )
        imageio.imsave(outfile, rgb)
