#date: 2022-09-12T17:01:24Z
#url: https://api.github.com/gists/7b239ca5a9e80163704891c58b3f8a82
#owner: https://api.github.com/users/jmuhlich

# usage: ome-tiff-set-resolution.py [-h] image size
#
# Set physical pixel dimensions in an OME-TIFF
#
# positional arguments:
#   image       OME-TIFF file to modify
#   size        pixel width in microns
#
# optional arguments:
#   -h, --help  show this help message and exit

import argparse
import ome_types
import sys
import tifffile

parser = argparse.ArgumentParser(
    description="Set physical pixel dimensions in an OME-TIFF",
)
parser.add_argument("image", help="OME-TIFF file to modify")
parser.add_argument("size", type=float, help="pixel width in microns")
args = parser.parse_args()

ome = ome_types.from_tiff(args.image)
pixels = ome.images[0].pixels
pixels.physical_size_x_unit = "µm"
pixels.physical_size_x = args.size
pixels.physical_size_y_unit = "µm"
pixels.physical_size_y = args.size

tifffile.tiffcomment(args.image, ome.to_xml())

print(f"Successfully set physical pixel dimensions in {args.image} to {args.size}")
